"""
Training runner — trains the ResNet18 medicine name classifier.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --epochs 15 --batch-size 64
"""
import sys
import os
import argparse

# Make sure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train medicine name classifier")
    parser.add_argument("--epochs",      type=int,   default=10,
                        help="Number of training epochs (default: 10)")
    parser.add_argument("--batch-size",  type=int,   default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--lr",          type=float, default=1e-3,
                        help="Initial learning rate (default: 0.001)")
    parser.add_argument("--unfreeze-at", type=int,   default=4,
                        help="Epoch at which to unfreeze backbone (default: 4)")
    args = parser.parse_args()

    val_acc, test_acc = train(
        epochs        = args.epochs,
        batch_size    = args.batch_size,
        lr            = args.lr,
        unfreeze_epoch= args.unfreeze_at,
    )
    print(f"\nFinal — val_acc={val_acc:.3f}  test_acc={test_acc:.3f}")
