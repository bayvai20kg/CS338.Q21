"""
train_ltc_lstm.py
=================
Training pipeline cho Gated-LTC-LSTM model.

Features:
  - Dataset class với sliding window (seq_len ngày → dự báo ngày tiếp theo)
  - Training loop với Adam optimizer + CosineAnnealingLR scheduler
  - Early stopping (patience=20)
  - Gradient clipping (max_norm=1.0) — quan trọng với RNN/ODE
  - Lưu best model checkpoint
  - Loss curves + prediction plots

Chạy: python train_ltc_lstm.py
"""

import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from gated_ltc_lstm import GatedLTCLSTM

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
CFG = {
    # Data
    'data_dir': '../../dataset_processed',
    'output_dir' : 'outputs',
    'seq_len'    : 14,          # 2 tuần lịch sử
    'batch_size' : 32,

    # Model
    'hidden_size': 64,
    'n_layers'   : 2,
    'dropout'    : 0.2,
    'tau_min'    : 1.0,
    'n_ode_steps': 6,

    # Training
    'epochs'         : 200,
    'lr'             : 1e-3,
    'weight_decay'   : 1e-4,
    'patience'       : 20,      # early stopping
    'grad_clip'      : 1.0,     # gradient clipping
    'scheduler_T_max': 100,

    # Misc
    'seed'  : 42,
    'device': 'cpu',
}

# ══════════════════════════════════════════════════════════════════════════════
# REPRODUCIBILITY
# ══════════════════════════════════════════════════════════════════════════════
torch.manual_seed(CFG['seed'])
np.random.seed(CFG['seed'])
os.makedirs(CFG['output_dir'], exist_ok=True)

plt.rcParams.update({
    'figure.facecolor': '#0f1117', 'axes.facecolor': '#1a1d2e',
    'axes.edgecolor': '#3a3d5c',   'axes.labelcolor': '#c8cce8',
    'xtick.color': '#8890c8',      'ytick.color': '#8890c8',
    'text.color': '#e8eaf6',       'grid.color': '#2a2d4a',
    'grid.linewidth': 0.6,         'figure.dpi': 120,
    'savefig.dpi': 150,            'savefig.facecolor': '#0f1117',
    'axes.titlecolor': '#a0c4ff',  'axes.titleweight': 'bold',
    'legend.facecolor': '#1a1d2e','legend.edgecolor': '#3a3d5c',
    'legend.labelcolor': '#c8cce8',
})

# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════
class PACDataset(Dataset):
    """
    Sliding window dataset cho bài toán PAC forecasting.

    Mỗi sample: (X, y) trong đó
        X : (seq_len, n_features) — lịch sử seq_len ngày
        y : scalar              — giá trị PAC ngày tiếp theo (scaled)
    """

    def __init__(self, csv_path: str, feature_cols: list, target: str, seq_len: int):
        df = pd.read_csv(csv_path, index_col='Ngay', parse_dates=True)
        self.X_data = df[feature_cols].values.astype(np.float32)
        self.y_data = df[target].values.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X_data) - self.seq_len

    def __getitem__(self, idx):
        X = self.X_data[idx : idx + self.seq_len]           # (seq_len, n_feat)
        y = self.y_data[idx + self.seq_len]                  # scalar
        return torch.from_numpy(X), torch.tensor(y)


# ══════════════════════════════════════════════════════════════════════════════
# METRICS (trên scaled values, rồi inverse transform để tính real metrics)
# ══════════════════════════════════════════════════════════════════════════════
def inverse_transform(y_scaled, scaler_params, target):
    mn = scaler_params[target]['min']
    mx = scaler_params[target]['max']
    return y_scaled * (mx - mn) + mn


def compute_metrics(y_true_real, y_pred_real):
    mae  = np.mean(np.abs(y_true_real - y_pred_real))
    rmse = np.sqrt(np.mean((y_true_real - y_pred_real) ** 2))
    ss_res = np.sum((y_true_real - y_pred_real) ** 2)
    ss_tot = np.sum((y_true_real - np.mean(y_true_real)) ** 2)
    r2   = 1 - ss_res / (ss_tot + 1e-10)
    mask = y_true_real != 0
    mape = np.mean(np.abs((y_true_real[mask] - y_pred_real[mask]) / y_true_real[mask])) * 100
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════
def train_epoch(model, loader, optimizer, criterion, device, grad_clip):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X).squeeze(-1)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds, trues = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X).squeeze(-1)
        total_loss += criterion(pred, y).item() * len(y)
        preds.append(pred.cpu().numpy())
        trues.append(y.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    return total_loss / len(loader.dataset), preds, trues


# ══════════════════════════════════════════════════════════════════════════════
# PLOT FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def plot_loss_curves(train_losses, val_losses, save_path):
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle('Training & Validation Loss — Gated-LTC-LSTM', fontsize=13,
                 color='#e8eaf6', fontweight='bold')
    ax.plot(train_losses, color='#00b4d8', linewidth=1.5, label='Train Loss (MSE)')
    ax.plot(val_losses,   color='#f72585', linewidth=1.5, label='Val Loss (MSE)')
    best_ep = int(np.argmin(val_losses))
    ax.axvline(best_ep, color='#ffbe0b', linestyle='--', linewidth=1.2,
               label=f'Best Val @ epoch {best_ep+1}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.legend()
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


def plot_predictions(y_true_real, y_pred_real, metrics, split_name, save_path):
    fig, axes = plt.subplots(2, 1, figsize=(16, 9))
    fig.suptitle(
        f'Gated-LTC-LSTM — PAC Forecasting ({split_name})  '
        f'MAE={metrics["MAE"]:.2f} mg/L  RMSE={metrics["RMSE"]:.2f}  R²={metrics["R2"]:.4f}',
        fontsize=12, color='#e8eaf6', fontweight='bold'
    )

    # Time-series plot
    ax1 = axes[0]
    x_idx = np.arange(len(y_true_real))
    ax1.plot(x_idx, y_true_real, color='#00b4d8', linewidth=1.2, label='Actual PAC (mg/L)', alpha=0.9)
    ax1.plot(x_idx, y_pred_real, color='#f72585', linewidth=1.2, label='Predicted PAC (mg/L)', alpha=0.85)
    ax1.fill_between(x_idx,
                     np.minimum(y_true_real, y_pred_real),
                     np.maximum(y_true_real, y_pred_real),
                     alpha=0.15, color='#ffbe0b', label='Error band')
    ax1.set_xlabel('Day index')
    ax1.set_ylabel('PAC Concentration (mg/L)')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.35)

    # Scatter plot
    ax2 = axes[1]
    mn_val = min(y_true_real.min(), y_pred_real.min())
    mx_val = max(y_true_real.max(), y_pred_real.max())
    ax2.scatter(y_true_real, y_pred_real, alpha=0.35, s=20, color='#7209b7')
    ax2.plot([mn_val, mx_val], [mn_val, mx_val], 'w--', linewidth=1.2, label='Perfect prediction')
    ax2.set_xlabel('Actual PAC (mg/L)')
    ax2.set_ylabel('Predicted PAC (mg/L)')
    ax2.set_title(f'Scatter: Actual vs Predicted  (R² = {metrics["R2"]:.4f})', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.35)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 62)
    print("  Gated-LTC-LSTM — PAC Forecasting Training")
    print("=" * 62)

    # ── 0. Preprocessing (nếu chưa có) ────────────────────────────────────────
    meta_path = os.path.join(CFG['data_dir'], 'metadata.json')
    if True:
        print(f"\n[0] Dữ liệu đã có tại {CFG['data_dir']}/")

    with open(meta_path, encoding='utf-8') as f:
        meta = json.load(f)
    with open(os.path.join(CFG['data_dir'], 'scaler_params.json'), encoding='utf-8') as f:
        scaler_params = json.load(f)

    feature_cols = meta['feature_cols']
    target       = meta['target']
    n_features   = meta['n_features']
    device       = torch.device(CFG['device'])

    print(f"    Features : {n_features}")
    print(f"    Target   : {target}")
    print(f"    Seq len  : {CFG['seq_len']} days")
    print(f"    Device   : {device}")

    # ── 1. DataLoaders ────────────────────────────────────────────────────────
    train_ds = PACDataset(f"{CFG['data_dir']}/train.csv", feature_cols, target, CFG['seq_len'])
    val_ds   = PACDataset(f"{CFG['data_dir']}/val.csv",   feature_cols, target, CFG['seq_len'])
    test_ds  = PACDataset(f"{CFG['data_dir']}/test.csv",  feature_cols, target, CFG['seq_len'])

    train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'], shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG['batch_size'], shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=CFG['batch_size'], shuffle=False)

    print(f"\n[1] Datasets → Train:{len(train_ds)} Val:{len(val_ds)} Test:{len(test_ds)}")

    # ── 2. Model ──────────────────────────────────────────────────────────────
    model = GatedLTCLSTM(
        input_size  = n_features,
        hidden_size = CFG['hidden_size'],
        n_layers    = CFG['n_layers'],
        dropout     = CFG['dropout'],
        tau_min     = CFG['tau_min'],
        n_ode_steps = CFG['n_ode_steps'],
    ).to(device)

    n_params = model.count_parameters()
    print(f"\n[2] Model: GatedLTCLSTM | Params: {n_params:,}")
    print(f"    hidden={CFG['hidden_size']} layers={CFG['n_layers']} "
          f"ode_steps={CFG['n_ode_steps']} tau_min={CFG['tau_min']}")

    # ── 3. Optimizer & Scheduler ──────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=CFG['lr'],
                                  weight_decay=CFG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG['scheduler_T_max'], eta_min=1e-5
    )
    criterion = nn.MSELoss()

    # ── 4. Training loop ──────────────────────────────────────────────────────
    best_val_loss = float('inf')
    patience_cnt  = 0
    train_losses, val_losses = [], []
    best_ckpt = os.path.join(CFG['output_dir'], 'best_model.pt')

    print(f"\n[3] Training (max {CFG['epochs']} epochs, patience={CFG['patience']})...")
    print(f"    {'Epoch':>6} | {'Train Loss':>11} | {'Val Loss':>10} | {'LR':>10} | {'Time':>8}")
    print("    " + "-" * 56)

    t_start = time.time()
    for epoch in range(1, CFG['epochs'] + 1):
        t0 = time.time()
        tr_loss  = train_epoch(model, train_loader, optimizer, criterion,
                               device, CFG['grad_clip'])
        val_loss, _, _ = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        lr_now = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        # Log mỗi 10 epochs hoặc khi có cải thiện
        if epoch % 10 == 0 or val_loss < best_val_loss:
            marker = ' *' if val_loss < best_val_loss else ''
            print(f"    {epoch:6d} | {tr_loss:11.6f} | {val_loss:10.6f} | "
                  f"{lr_now:10.2e} | {elapsed:6.1f}s{marker}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt  = 0
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'val_loss'   : best_val_loss,
                'cfg'        : CFG,
            }, best_ckpt)
        else:
            patience_cnt += 1
            if patience_cnt >= CFG['patience']:
                print(f"\n    Early stopping at epoch {epoch} "
                      f"(no improvement for {CFG['patience']} epochs)")
                break

    total_time = time.time() - t_start
    print(f"\n    Training complete in {total_time:.1f}s | "
          f"Best val loss: {best_val_loss:.6f}")

    # ── 5. Plot loss curves ───────────────────────────────────────────────────
    plot_loss_curves(train_losses, val_losses,
                     os.path.join(CFG['output_dir'], 'loss_curves.png'))

    # ── 6. Evaluation ─────────────────────────────────────────────────────────
    print(f"\n[4] Loading best model from {best_ckpt}...")
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    print(f"    Best epoch: {ckpt['epoch']} | Val MSE: {ckpt['val_loss']:.6f}")

    results = {}
    for split_name, loader in [('Validation', val_loader), ('Test', test_loader)]:
        _, preds_s, trues_s = eval_epoch(model, loader, criterion, device)
        preds_real = inverse_transform(preds_s, scaler_params, target)
        trues_real = inverse_transform(trues_s, scaler_params, target)
        metrics    = compute_metrics(trues_real, preds_real)
        results[split_name] = {
            'metrics': metrics,
            'preds'  : preds_real.tolist(),
            'trues'  : trues_real.tolist(),
        }

        fname = 'val_predictions.png' if split_name == 'Validation' else 'test_predictions.png'
        plot_predictions(trues_real, preds_real, metrics, split_name,
                         os.path.join(CFG['output_dir'], fname))

        print(f"\n  [{split_name}]")
        print(f"    MAE  = {metrics['MAE']:.4f} mg/L")
        print(f"    RMSE = {metrics['RMSE']:.4f} mg/L")
        print(f"    R²   = {metrics['R2']:.4f}")
        print(f"    MAPE = {metrics['MAPE']:.2f}%")

    # ── 7. Save results ───────────────────────────────────────────────────────
    results_path = os.path.join(CFG['output_dir'], 'results.json')
    def to_native(d):
        return {k: float(v) for k, v in d.items()}

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'config'      : CFG,
            'n_params'    : n_params,
            'best_epoch'  : int(ckpt['epoch']),
            'best_val_mse': float(ckpt['val_loss']),
            'validation'  : to_native(results['Validation']['metrics']),
            'test'        : to_native(results['Test']['metrics']),
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[5] Results saved → {results_path}")
    print("\n" + "=" * 62)
    print("  TRAINING PIPELINE COMPLETE!")
    print("=" * 62)
    print(f"""
  Output files:
    outputs/best_model.pt       — best model checkpoint
    outputs/loss_curves.png     — training curves
    outputs/val_predictions.png — validation predictions
    outputs/test_predictions.png— test predictions
    outputs/results.json        — metrics summary
""")


if __name__ == '__main__':
    main()
