"""
Train Neural ZUPT Model
========================
  python tools/train_zupt.py                     # Default
  python tools/train_zupt.py --epochs 50 --lr 0.001
  python tools/train_zupt.py --data data/zupt_training --output models/zupt_net.pt
"""
import argparse, json, logging, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from airwriting_imu.ml.neural_zupt import ZUPTNet
except ImportError:
    print("❌ PyTorch is required: pip install torch")
    sys.exit(1)


def load_dataset(data_dir: str) -> tuple:
    p = Path(data_dir)
    X = np.load(p / "X.npy")
    y = np.load(p / "y.npy")
    return X, y


def train(args):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    # ── Load data ──
    log.info(f"Loading from {args.data}")
    X, y = load_dataset(args.data)
    log.info(f"Dataset: {X.shape[0]:,} samples, window={X.shape[1]}")

    # Train/val split
    n = len(X)
    idx = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx, val_idx = idx[:split], idx[split:]

    X_train = torch.from_numpy(X[train_idx])
    y_train = torch.from_numpy(y[train_idx])
    X_val = torch.from_numpy(X[val_idx])
    y_val = torch.from_numpy(y[val_idx])

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size * 2)

    log.info(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}")

    # ── Model ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ZUPTNet(input_dim=6, hidden_dim=args.hidden, num_layers=2,
                    dropout=args.dropout).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model: {n_params:,} params on {device}")

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    # ── Training loop ──
    best_val_loss = float("inf")
    best_val_acc = 0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * len(xb)
            predicted = (pred > 0.5).float()
            train_correct += (predicted == yb).sum().item()
            train_total += len(xb)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_tp = val_fp = val_tn = val_fn = 0

        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)

                val_loss += loss.item() * len(xb)
                predicted = (pred > 0.5).float()
                val_correct += (predicted == yb).sum().item()
                val_total += len(xb)

                val_tp += ((predicted == 1) & (yb == 1)).sum().item()
                val_fp += ((predicted == 1) & (yb == 0)).sum().item()
                val_tn += ((predicted == 0) & (yb == 0)).sum().item()
                val_fn += ((predicted == 0) & (yb == 1)).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total
        precision = val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 0
        recall = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        log.info(
            f"E{epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.3f} "
            f"P={precision:.3f} R={recall:.3f} F1={f1:.3f} | "
            f"{elapsed:.1f}s"
        )

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0

            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_path)
            log.info(f"  💾 Saved → {out_path} (val_acc={val_acc:.3f}, F1={f1:.3f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                log.info(f"Early stopping at epoch {epoch}")
                break

    log.info(f"\n✅ Training complete. Best val_acc={best_val_acc:.3f}")
    log.info(f"   Model saved to: {args.output}")


def main():
    ap = argparse.ArgumentParser(description="Train Neural ZUPT")
    ap.add_argument("--data", default="data/zupt_training")
    ap.add_argument("--output", default="models/zupt_net.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=10)
    args = ap.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)

    train(args)


if __name__ == "__main__":
    main()
