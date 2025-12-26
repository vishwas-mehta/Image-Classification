"""
Main entry point for training and evaluating models.
Run this script to train models from command line.
"""

import argparse
import torch
import sys
sys.path.append('.')

from config import set_seed, get_device, EPOCHS, LEARNING_RATE
from src.data_loader import get_data_loaders
from src.models.custom_cnn import CustomCNN
from src.models.resnet import ResNet18
from src.train import train_model
from src.evaluate import evaluate_and_report
from src.utils import plot_training_history, plot_confusion_matrix, plot_model_comparison


def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 Image Classification')
    parser.add_argument('--model', type=str, default='cnn', 
                        choices=['cnn', 'resnet'],
                        help='Model to train (cnn or resnet)')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--no-augmentation', action='store_true',
                        help='Disable data augmentation')
    parser.add_argument('--eval-only', action='store_true',
                        help='Only evaluate a saved model')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to saved model for evaluation')
    
    args = parser.parse_args()
    
    # Set seed and device
    set_seed(42)
    device = get_device()
    
    # Load data
    print("\n" + "=" * 60)
    print("Loading CIFAR-10 dataset...")
    print("=" * 60)
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=args.batch_size,
        use_augmentation=not args.no_augmentation
    )
    
    # Create model
    if args.model == 'cnn':
        model = CustomCNN(num_classes=10, dropout_rate=0.5)
        model_name = 'custom_cnn'
    else:
        model = ResNet18(num_classes=10)
        model_name = 'resnet18'
    
    print(f"\nModel: {model_name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    if args.eval_only:
        # Load and evaluate model
        if args.model_path:
            model.load_state_dict(torch.load(args.model_path))
        else:
            model.load_state_dict(torch.load(f'models/{model_name}_best.pth'))
        
        print("\nEvaluating model...")
        results = evaluate_and_report(model, test_loader, device=device)
        plot_confusion_matrix(results['confusion_matrix'], 
                              save_path=f'results/{model_name}_confusion_matrix.png')
    else:
        # Train model
        print("\n" + "=" * 60)
        print(f"Training {model_name}...")
        print("=" * 60)
        
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            learning_rate=args.lr,
            device=device,
            use_scheduler=True,
            use_early_stopping=True,
            model_name=model_name
        )
        
        # Plot training history
        plot_training_history(history, save_path=f'results/{model_name}_training.png')
        
        # Evaluate on test set
        print("\n" + "=" * 60)
        print("Evaluating on test set...")
        print("=" * 60)
        
        model.load_state_dict(torch.load(f'models/{model_name}_best.pth'))
        results = evaluate_and_report(model, test_loader, device=device)
        plot_confusion_matrix(results['confusion_matrix'], 
                              save_path=f'results/{model_name}_confusion_matrix.png')
    
    print("\n" + "=" * 60)
    print("Done! Check the 'results/' folder for visualizations.")
    print("=" * 60)


if __name__ == '__main__':
    main()
