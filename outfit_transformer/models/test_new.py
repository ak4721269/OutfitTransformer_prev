# -*- coding:utf-8 -*-
"""
Test Script for OutfitTransformer
"""
import os
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from outfit_transformer.models.outfit_transformer import OutfitTransformer
from outfit_transformer.utils.utils import load_dataloader  # Assumes you have a utility to load test dataloader.

def load_model(checkpoint_path, device):
    """Load the trained model from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = OutfitTransformer()
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    return model

def evaluate_model(model, dataloader, device):
    """Evaluate the model on the test data."""
    all_targets = []
    all_predictions = []
    all_scores = []

    for batch in tqdm(dataloader, desc="Testing"):
        inputs = {key: value.to(device) for key, value in batch['inputs'].items()}
        targets = batch['targets'].view(-1).cpu().numpy()
        
        with torch.no_grad():
            logits = model.cp_forward(inputs, do_encode=True)
            scores = logits.view(-1).cpu().numpy()
            predictions = (scores >= 0.5).astype(int)
        
        all_targets.extend(targets)
        all_predictions.extend(predictions)
        all_scores.extend(scores)

    return np.array(all_targets), np.array(all_predictions), np.array(all_scores)

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot and save the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    # Paths and settings
    checkpoint_path = "path/to/your/model_checkpoint.pth"  # Update with your model path
    test_data_path = "path/to/your/test_data"  # Update with your test data path
    confusion_matrix_path = "confusion_matrix.png"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and test dataloader
    model = load_model(checkpoint_path, device)
    test_dataloader = load_dataloader(test_data_path, mode='test')  # Implement this function as needed.

    # Evaluate the model
    y_true, y_pred, y_scores = evaluate_model(model, test_dataloader, device)

    # Compute metrics
    acc = np.mean(y_true == y_pred)
    auc = roc_auc_score(y_true, y_scores)
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    # Plot and save confusion matrix
    plot_confusion_matrix(y_true, y_pred, save_path=confusion_matrix_path)
    print(f"Confusion matrix saved to {confusion_matrix_path}")

if __name__ == "__main__":
    main()
