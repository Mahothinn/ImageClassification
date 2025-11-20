#!/usr/bin/env python3
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_model(model_path, num_classes):
    """学習済みモデルを読み込む"""
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    checkpoint = torch.load(model_path, map_location="cpu")
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model

def create_dataloader(test_dir, batch_size, image_size):
    """テストフォルダ → DataLoader 作成"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader, test_dataset.classes

def run_inference(model, dataloader):
    """推論実行"""
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)

def plot_confusion_matrix(labels, preds, class_names, output_path):
    """混同行列のプロット"""
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names,
                yticklabels=class_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="outputs/best_model.pth", help="学習済みモデルのパス")
    parser.add_argument("--test-dir", required=True, help="外部テストデータフォルダ")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output-dir", default="inference_results")
    parser.add_argument("--plot-cm", action="store_true", help="混同行列を保存")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("🔍 テストデータ読み込み中:", args.test_dir)
    test_loader, class_names = create_dataloader(args.test_dir, args.batch_size, args.image_size)

    print("🧠 モデル読み込み中:", args.model_path)
    model = load_model(args.model_path, num_classes=len(class_names))

    print("🚀 推論実行中…")
    labels, preds = run_inference(model, test_loader)

    print("\n📘 Classification Report")
    print(classification_report(labels, preds, target_names=class_names))

    if args.plot_cm:
        cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
        plot_confusion_matrix(labels, preds, class_names, cm_path)
        print("📊 混同行列を保存しました:", cm_path)

    print("\n✨ 推論完了!")

if __name__ == "__main__":
    main()
