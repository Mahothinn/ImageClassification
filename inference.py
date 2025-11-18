#!/usr/bin/env python3
"""
保存されたモデルを使用してテストデータで推論を実行するスクリプト
"""

from __future__ import annotations

import argparse
from pathlib import Path

import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def load_model(model_path: Path, device: torch.device) -> tuple[nn.Module, list[str], int]:
    """
    保存されたモデルを読み込む。

    Parameters
    ----------
    model_path : Path
        モデルファイルのパス
    device : torch.device
        使用デバイス

    Returns
    -------
    tuple[nn.Module, list[str], int]
        (model, class_names, num_classes)
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    num_classes = checkpoint['num_classes']
    class_names = checkpoint['class_names']
    
    # モデルを作成
    model = efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_features, num_classes)
    )
    
    # 重みを読み込む
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, class_names, num_classes


def get_test_loader(
    data_dir: Path,
    class_names: list[str],
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    splits_path: Path | None = None,
) -> DataLoader:
    """
    テストデータのDataLoaderを作成する。

    Parameters
    ----------
    data_dir : Path
        データセットのルートディレクトリ
    class_names : list[str]
        クラス名のリスト（順序が重要）
    batch_size : int
        バッチサイズ
    num_workers : int
        DataLoaderのワーカー数
    image_size : int
        画像サイズ

    Returns
    -------
    DataLoader
        テストデータのDataLoader
    """
    dataset_dir = data_dir / "NA_Fish_Dataset"
    
    # テスト用のtransform
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # データセットを読み込み
    full_dataset = datasets.ImageFolder(
        root=str(dataset_dir),
        transform=test_transform,
    )
    test_dataset: torch.utils.data.Dataset
    subset_info = None

    if splits_path and splits_path.exists():
        with splits_path.open("r", encoding="utf-8") as f:
            subset_info = json.load(f)
        test_indices = subset_info.get("test")
        if test_indices is None:
            print("警告: splits.json に test インデックスが含まれていません。全データを使用します。")
            test_dataset = full_dataset
        else:
            print(f"保存されたテスト分割（{len(test_indices)}枚）を使用します。")
            test_dataset = Subset(full_dataset, test_indices)
    else:
        if splits_path:
            print(f"警告: {splits_path} が見つかりません。全データを使用します。")
        test_dataset = full_dataset
    
    # クラス名の順序を確認
    dataset_class_names = full_dataset.classes
    if dataset_class_names != class_names:
        print(f"警告: データセットのクラス順序が異なります。")
        print(f"  モデル: {class_names}")
        print(f"  データセット: {dataset_class_names}")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return test_loader


def inference(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    class_names: list[str],
) -> tuple[list[int], list[int], list[float]]:
    """
    テストデータで推論を実行する。

    Parameters
    ----------
    model : nn.Module
        モデル
    test_loader : DataLoader
        テストデータのDataLoader
    device : torch.device
        使用デバイス
    class_names : list[str]
        クラス名のリスト

    Returns
    -------
    tuple[list[int], list[int], list[float]]
        (true_labels, pred_labels, confidences)
    """
    true_labels = []
    pred_labels = []
    confidences = []
    
    model.eval()
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="推論中")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())
            confidences.extend(confidence.cpu().numpy())
            
            # 進捗表示
            correct = (predicted == labels).sum().item()
            total = labels.size(0)
            pbar.set_postfix({"acc": f"{100 * correct / total:.2f}%"})
    
    return true_labels, pred_labels, confidences


def plot_confusion_matrix(
    true_labels: list[int],
    pred_labels: list[int],
    class_names: list[str],
    output_path: Path,
) -> None:
    """
    混同行列をプロットして保存する。

    Parameters
    ----------
    true_labels : list[int]
        正解ラベル
    pred_labels : list[int]
        予測ラベル
    class_names : list[str]
        クラス名のリスト
    output_path : Path
        保存先パス
    """
    cm = confusion_matrix(true_labels, pred_labels)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"混同行列を保存しました: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="保存されたモデルでテストデータの推論を実行する"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("./outputs/best_model.pth"),
        help="モデルファイルのパス（デフォルト: ./outputs/best_model.pth）",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data"),
        help="データセットディレクトリ（デフォルト: ./data）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="バッチサイズ（デフォルト: 32）",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="画像サイズ（デフォルト: 224）",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoaderのワーカー数（デフォルト: 4）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./outputs"),
        help="結果保存先ディレクトリ（デフォルト: ./outputs）",
    )
    parser.add_argument(
        "--plot-cm",
        action="store_true",
        help="混同行列をプロットして保存する",
    )
    parser.add_argument(
        "--splits-path",
        type=Path,
        default=Path("./outputs/splits.json"),
        help="訓練時に保存したデータ分割情報（デフォルト: ./outputs/splits.json）",
    )

    args = parser.parse_args()

    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # モデルを読み込む
    print(f"\nモデルを読み込み中: {args.model_path}")
    model, class_names, num_classes = load_model(args.model_path, device)
    print(f"クラス数: {num_classes}")
    print(f"クラス名: {class_names}")

    # テストデータのDataLoaderを作成
    print("\nテストデータを読み込み中...")
    test_loader = get_test_loader(
        data_dir=args.data_dir,
        class_names=class_names,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        splits_path=args.splits_path,
    )
    print(f"テストデータ: {len(test_loader.dataset)}枚")

    # 推論を実行
    print("\n推論を実行中...")
    true_labels, pred_labels, confidences = inference(
        model, test_loader, device, class_names
    )

    # 結果を計算
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    confidences = np.array(confidences)
    
    accuracy = np.mean(true_labels == pred_labels) * 100
    
    print(f"\n{'='*60}")
    print(f"推論結果")
    print(f"{'='*60}")
    print(f"総画像数: {len(true_labels)}枚")
    print(f"正解数: {np.sum(true_labels == pred_labels)}枚")
    print(f"誤分類数: {np.sum(true_labels != pred_labels)}枚")
    print(f"精度: {accuracy:.2f}%")
    print(f"平均信頼度: {np.mean(confidences):.4f}")

    # クラスごとの精度
    print(f"\n{'='*60}")
    print(f"クラスごとの精度")
    print(f"{'='*60}")
    for i, class_name in enumerate(class_names):
        mask = true_labels == i
        if np.sum(mask) > 0:
            class_acc = np.mean(pred_labels[mask] == i) * 100
            class_count = np.sum(mask)
            print(f"{class_name:20s}: {class_acc:6.2f}% ({np.sum((true_labels == i) & (pred_labels == i))}/{class_count})")

    # 分類レポート
    print(f"\n{'='*60}")
    print(f"詳細な分類レポート")
    print(f"{'='*60}")
    report = classification_report(
        true_labels,
        pred_labels,
        target_names=class_names,
        digits=4,
    )
    print(report)

    # 混同行列をプロット
    if args.plot_cm:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        cm_path = args.output_dir / "confusion_matrix.png"
        plot_confusion_matrix(true_labels, pred_labels, class_names, cm_path)

    # 誤分類の詳細
    incorrect_mask = true_labels != pred_labels
    if np.sum(incorrect_mask) > 0:
        print(f"\n{'='*60}")
        print(f"誤分類の詳細 ({np.sum(incorrect_mask)}件)")
        print(f"{'='*60}")
        incorrect_indices = np.where(incorrect_mask)[0]
        for idx in incorrect_indices[:10]:  # 最初の10件を表示
            true_class = class_names[true_labels[idx]]
            pred_class = class_names[pred_labels[idx]]
            confidence = confidences[idx]
            print(f"  正解: {true_class:20s} → 予測: {pred_class:20s} (信頼度: {confidence:.4f})")
        if np.sum(incorrect_mask) > 10:
            print(f"  ... 他 {np.sum(incorrect_mask) - 10} 件")

    print("\n推論が完了しました！")


if __name__ == "__main__":
    main()

