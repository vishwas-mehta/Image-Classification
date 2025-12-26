"""
Evaluation module for trained models.
Includes accuracy calculation, confusion matrix, and classification report.
"""

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm

import sys
sys.path.append('..')
from config import DEVICE, CLASSES


def evaluate_model(model, test_loader, device=DEVICE):
    """
    Evaluate model on test dataset.
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to use
    
    Returns:
        accuracy, all_predictions, all_labels, all_images
    """
    model.eval()
    model = model.to(device)
    
    all_predictions = []
    all_labels = []
    all_images = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Evaluating')
        
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Store some images for visualization
            if len(all_images) < 100:
                all_images.extend(images.cpu())
    
    accuracy = 100. * correct / total
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    
    return accuracy, all_predictions, all_labels, all_images[:100]


def get_confusion_matrix(labels, predictions):
    """
    Generate confusion matrix.
    
    Args:
        labels: True labels
        predictions: Model predictions
    
    Returns:
        Confusion matrix as numpy array
    """
    cm = confusion_matrix(labels, predictions)
    return cm


def get_classification_report(labels, predictions, output_dict=False):
    """
    Generate classification report.
    
    Args:
        labels: True labels
        predictions: Model predictions
        output_dict: Return as dictionary
    
    Returns:
        Classification report string or dict
    """
    report = classification_report(
        labels, 
        predictions, 
        target_names=CLASSES,
        output_dict=output_dict
    )
    return report


def get_per_class_accuracy(labels, predictions):
    """
    Calculate accuracy for each class.
    
    Args:
        labels: True labels
        predictions: Model predictions
    
    Returns:
        List of per-class accuracies
    """
    class_accuracies = []
    
    for i in range(len(CLASSES)):
        mask = labels == i
        if mask.sum() > 0:
            class_acc = 100. * (predictions[mask] == i).sum() / mask.sum()
            class_accuracies.append(class_acc)
        else:
            class_accuracies.append(0.0)
    
    return class_accuracies


def evaluate_and_report(model, test_loader, device=DEVICE, verbose=True):
    """
    Complete evaluation with all metrics.
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to use
        verbose: Print detailed report
    
    Returns:
        Dictionary with all evaluation results
    """
    # Get predictions
    accuracy, predictions, labels, sample_images = evaluate_model(
        model, test_loader, device
    )
    
    # Confusion matrix
    cm = get_confusion_matrix(labels, predictions)
    
    # Classification report
    report = get_classification_report(labels, predictions, output_dict=True)
    report_str = get_classification_report(labels, predictions, output_dict=False)
    
    # Per-class accuracy
    class_accuracies = get_per_class_accuracy(labels, predictions)
    
    if verbose:
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"\nOverall Test Accuracy: {accuracy:.2f}%")
        print("\nPer-Class Accuracy:")
        print("-" * 40)
        for i, (class_name, class_acc) in enumerate(zip(CLASSES, class_accuracies)):
            print(f"  {class_name:15} : {class_acc:.2f}%")
        print("-" * 40)
        print(f"\nClassification Report:\n{report_str}")
    
    results = {
        'accuracy': accuracy,
        'predictions': predictions,
        'labels': labels,
        'sample_images': sample_images,
        'confusion_matrix': cm,
        'classification_report': report,
        'per_class_accuracy': class_accuracies
    }
    
    return results


def compare_models(models_dict, test_loader, device=DEVICE):
    """
    Compare multiple models on the same test set.
    
    Args:
        models_dict: Dictionary with model_name: model pairs
        test_loader: Test data loader
        device: Device to use
    
    Returns:
        Dictionary with comparison results
    """
    comparison = {}
    
    for name, model in models_dict.items():
        print(f"\nEvaluating {name}...")
        print("-" * 40)
        
        accuracy, _, _, _ = evaluate_model(model, test_loader, device)
        comparison[name] = accuracy
    
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    for name, acc in sorted(comparison.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name:20} : {acc:.2f}%")
    
    return comparison


def get_misclassified_samples(model, test_loader, device=DEVICE, max_samples=20):
    """
    Get misclassified samples for error analysis.
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to use
        max_samples: Maximum number of samples to return
    
    Returns:
        List of (image, true_label, predicted_label) tuples
    """
    model.eval()
    model = model.to(device)
    
    misclassified = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            # Find misclassified samples
            mask = predicted != labels
            wrong_images = images[mask]
            wrong_labels = labels[mask]
            wrong_preds = predicted[mask]
            
            for img, true_l, pred_l in zip(wrong_images, wrong_labels, wrong_preds):
                misclassified.append((
                    img.cpu(),
                    true_l.item(),
                    pred_l.item()
                ))
                
                if len(misclassified) >= max_samples:
                    return misclassified
    
    return misclassified


if __name__ == "__main__":
    # Test evaluation
    from src.data_loader import get_data_loaders
    from src.models.custom_cnn import CustomCNN
    
    # Get test loader
    _, _, test_loader = get_data_loaders(batch_size=128)
    
    # Create untrained model (just for testing the pipeline)
    model = CustomCNN()
    
    # Evaluate
    results = evaluate_and_report(model, test_loader)
    
    print("\n Test completed!")
    print(f"Confusion matrix shape: {results['confusion_matrix'].shape}")
