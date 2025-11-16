#!/usr/bin/env python3
"""
NA_Fish_DatasetをEfficientNetで画像分類する訓練スクリプト
"""

from __future__ import annotations

import argparse
import os
import random
import ssl
import urllib.request
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from tqdm import tqdm


class TransformDataset(torch.utils.data.Dataset):
    """transformを変更可能なデータセットラッパー"""
    def __init__(self, dataset: torch.utils.data.Dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.dataset[index]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.dataset)


def get_data_loaders(
    data_dir: Path,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    num_workers: int = 4,
    image_size: int = 224,
) -> tuple[DataLoader, DataLoader, DataLoader, int, list[str]]:
    """
    データセットを読み込み、train/val/testに分割してDataLoaderを作成する。

    Parameters
    ----------
    data_dir : Path
        データセットのルートディレクトリ（NA_Fish_Datasetを含む）
    batch_size : int
        バッチサイズ
    train_ratio : float
        訓練データの割合
    val_ratio : float
        検証データの割合
    num_workers : int
        DataLoaderのワーカー数
    image_size : int
        リサイズ後の画像サイズ

    Returns
    -------
    tuple[DataLoader, DataLoader, DataLoader, int, list[str]]
        (train_loader, val_loader, test_loader, num_classes, class_names)
    """
    dataset_dir = data_dir / "NA_Fish_Dataset"
    if not dataset_dir.exists():
        raise FileNotFoundError(f"データセットが見つかりません: {dataset_dir}")

    # より強力なデータ拡張と正規化（過学習防止のため）
    train_transform = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),  # 少し大きめにリサイズ
        transforms.RandomCrop(image_size),  # ランダムクロップ
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),  # 上下反転も追加（魚の向きが変わるため）
        transforms.RandomRotation(degrees=30),  # 回転角度を増やす
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # ランダムアフィン変換
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # より強い色調整
        transforms.RandomGrayscale(p=0.1),  # たまにグレースケール化
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.2),  # ぼかし
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.33)),  # ランダム消去
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # データセットを読み込み（transformなしで一度読み込む）
    full_dataset = datasets.ImageFolder(
        root=str(dataset_dir),
        transform=None,
    )

    # クラス名を取得
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"クラス数: {num_classes}")
    print(f"クラス名: {class_names}")

    # クラスごとにインデックスをグループ化（層化分割のため）
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(full_dataset):
        class_indices[label].append(idx)

    print("\nクラスごとの画像数:")
    for class_idx, indices in class_indices.items():
        print(f"  {class_names[class_idx]}: {len(indices)}枚")

    # 各クラスごとにtrain/val/testに分割（層化分割）
    train_indices = []
    val_indices = []
    test_indices = []

    for class_idx, indices in class_indices.items():
        # 各クラスのデータをシャッフル
        random.seed(42)
        shuffled_indices = indices.copy()
        random.shuffle(shuffled_indices)

        # クラスごとに分割
        class_size = len(shuffled_indices)
        class_train_size = int(train_ratio * class_size)
        class_val_size = int(val_ratio * class_size)
        class_test_size = class_size - class_train_size - class_val_size

        train_indices.extend(shuffled_indices[:class_train_size])
        val_indices.extend(shuffled_indices[class_train_size:class_train_size + class_val_size])
        test_indices.extend(shuffled_indices[class_train_size + class_val_size:])

    # インデックスをシャッフル（クラス間の順序をランダム化）
    random.seed(42)
    random.shuffle(train_indices)
    random.shuffle(val_indices)
    random.shuffle(test_indices)

    # Subsetを作成
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    test_subset = Subset(full_dataset, test_indices)

    print(f"\n層化分割後のクラス分布:")
    print(f"訓練データ: {len(train_subset)}枚")
    print(f"検証データ: {len(val_subset)}枚")
    print(f"テストデータ: {len(test_subset)}枚")

    # 各データセットにtransformを適用
    train_dataset = TransformDataset(
        train_subset,
        train_transform,
    )
    val_dataset = TransformDataset(
        val_subset,
        val_test_transform,
    )
    test_dataset = TransformDataset(
        test_subset,
        val_test_transform,
    )

    # DataLoaderを作成
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"訓練データ: {len(train_dataset)}枚")
    print(f"検証データ: {len(val_dataset)}枚")
    print(f"テストデータ: {len(test_dataset)}枚")

    return train_loader, val_loader, test_loader, num_classes, class_names


def create_model(
    num_classes: int,
    pretrained: bool = True,
    disable_ssl_verify: bool = False,
) -> nn.Module:
    """
    EfficientNet-B0モデルを作成する。

    Parameters
    ----------
    num_classes : int
        分類クラス数
    pretrained : bool
        事前学習済み重みを使用するか
    disable_ssl_verify : bool
        SSL証明書の検証を無効にする（開発環境のみ）

    Returns
    -------
    nn.Module
        EfficientNetモデル
    """
    if pretrained:
        # SSL設定用の変数を初期化
        original_https_context = None
        original_opener = None
        original_ssl_verify = None
        
        try:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            print("事前学習済み重みをダウンロード中...")
            
            # SSL証明書の問題を回避するための設定
            if disable_ssl_verify:
                warnings.warn(
                    "SSL証明書の検証を無効にしています。"
                    "本番環境では使用しないでください。"
                )
                # urllibのSSL検証を無効化
                original_https_context = ssl._create_default_https_context
                ssl._create_default_https_context = ssl._create_unverified_context
                
                # urllib.requestのopenerも設定
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                https_handler = urllib.request.HTTPSHandler(context=ssl_context)
                opener = urllib.request.build_opener(https_handler)
                # 既存のopenerを保存（Noneの可能性がある）
                original_opener = getattr(urllib.request, '_opener', None)
                urllib.request.install_opener(opener)
                
                # 環境変数も設定（torchvisionが使用する可能性がある）
                original_ssl_verify = os.environ.get('PYTHONHTTPSVERIFY', '')
                os.environ['PYTHONHTTPSVERIFY'] = '0'
            
            try:
                model = efficientnet_b0(weights=weights)
                print("事前学習済み重みの読み込みが完了しました。")
            finally:
                # SSL設定を元に戻す
                if disable_ssl_verify and original_https_context is not None:
                    ssl._create_default_https_context = original_https_context
                    if original_opener is not None:
                        urllib.request.install_opener(original_opener)
                    else:
                        # デフォルトのopenerを再構築
                        urllib.request._opener = None
                    if original_ssl_verify:
                        os.environ['PYTHONHTTPSVERIFY'] = original_ssl_verify
                    elif 'PYTHONHTTPSVERIFY' in os.environ:
                        del os.environ['PYTHONHTTPSVERIFY']
        except Exception as e:
            warnings.warn(
                f"事前学習済み重みのダウンロードに失敗しました: {e}\n"
                "事前学習なしでモデルを初期化します。"
            )
            model = efficientnet_b0(weights=None)
    else:
        model = efficientnet_b0(weights=None)

    # 最終層を置き換え（ドロップアウトを追加して過学習を防止）
    num_features = model.classifier[1].in_features
    # ドロップアウト層を追加
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),  # 50%のドロップアウト
        nn.Linear(num_features, num_classes)
    )

    return model


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """1エポックの訓練を実行する。"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="訓練中")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100 * correct / total:.2f}%",
        })

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """検証を実行する。"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="検証中")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100 * correct / total:.2f}%",
            })

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NA_Fish_DatasetをEfficientNetで画像分類する"
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
        "--epochs",
        type=int,
        default=50,
        help="エポック数（デフォルト: 50）",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="学習率（デフォルト: 0.001）",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="画像サイズ（デフォルト: 224）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./outputs"),
        help="モデル保存先ディレクトリ（デフォルト: ./outputs）",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="事前学習済み重みを使用しない",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoaderのワーカー数（デフォルト: 4）",
    )
    parser.add_argument(
        "--disable-ssl-verify",
        action="store_true",
        help="SSL証明書の検証を無効にする（開発環境のみ、セキュリティリスクあり）",
    )

    args = parser.parse_args()

    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # データローダー作成
    print("\nデータローダーを作成中...")
    train_loader, val_loader, test_loader, num_classes, class_names = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    # モデル作成
    print("\nモデルを作成中...")
    model = create_model(
        num_classes=num_classes,
        pretrained=not args.no_pretrained,
        disable_ssl_verify=args.disable_ssl_verify,
    )
    model = model.to(device)

    # 損失関数とオプティマイザー
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # 出力ディレクトリ作成
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 訓練ループ
    print("\n訓練を開始します...")
    best_val_acc = 0.0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nエポック {epoch}/{args.epochs}")
        print("-" * 50)

        # 訓練
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 検証
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # 学習率スケジューラー更新
        scheduler.step(val_loss)

        print(f"\n訓練 - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        print(f"検証 - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

        # ベストモデルを保存
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = args.output_dir / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "num_classes": num_classes,
                "class_names": class_names,
            }, model_path)
            print(f"ベストモデルを保存しました: {model_path}")

    # 最終モデルを保存
    final_model_path = args.output_dir / "final_model.pth"
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "num_classes": num_classes,
        "class_names": class_names,
    }, final_model_path)
    print(f"\n最終モデルを保存しました: {final_model_path}")

    # テストデータで評価
    print("\nテストデータで評価中...")
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"テスト - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")

    print("\n訓練が完了しました！")


if __name__ == "__main__":
    main()

