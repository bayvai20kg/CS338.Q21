"""
train_lstm_baseline.py
======================
Baseline LSTM model để so sánh với Gated-LTC-LSTM.

Dùng CÙNG:
  - Dataset (dataset_processed/)
  - Seq_len, batch_size, optimizer, scheduler
  - Metrics: MAE, RMSE, R², MAPE
  - Plot style

Chạy: python -X utf8 train_lstm_baseline.py
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


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — giữ nguyên các thông số giống LTC-LSTM để so sánh công bằng
# ══════════════════════════════════════════════════════════════════════════════
CFG = {
    'data_dir': '../../dataset_processed',
    'output_dir'     : 'outputs_lstm',
    'seq_len'        : 14,
    'batch_size'     : 32,
    # Model
    'hidden_size'    : 64,
    'n_layers'       : 2,
    'dropout'        : 0.2,
    # Training
    'epochs'         : 200,
    'lr'             : 1e-3,
    'weight_decay'   : 1e-4,
    'patience'       : 20,
    'grad_clip'      : 1.0,
    'scheduler_T_max': 100,
    'seed'           : 42,
    'device'         : 'cpu',
}

torch.manual_seed(CFG['seed'])
np.random.seed(CFG['seed'])
os.makedirs(CFG['output_dir'], exist_ok=True)

# ── Dark style (giống LTC-LSTM) ───────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f1117', 'axes.facecolor': '#1a1d2e',
    'axes.edgecolor': '#3a3d5c',   'axes.labelcolor': '#c8cce8',
    'xtick.color': '#8890c8',      'ytick.color': '#8890c8',
    'text.color': '#e8eaf6',       'grid.color': '#2a2d4a',
    'grid.linewidth': 0.6,         'figure.dpi': 120,
    'savefig.dpi': 150,            'savefig.facecolor': '#0f1117',
    'axes.titlecolor': '#a0c4ff',  'axes.titleweight': 'bold',
    'legend.facecolor': '#1a1d2e', 'legend.edgecolor': '#3a3d5c',
    'legend.labelcolor': '#c8cce8',
})

# ══════════════════════════════════════════════════════════════════════════════
# MODEL — Standard LSTM (PyTorch built-in)
# ══════════════════════════════════════════════════════════════════════════════
class LSTMBaseline(nn.Module):
    """
    Vanilla stacked LSTM với output head giống Gated-LTC-LSTM.

    Architecture:
        Input  : (batch, seq_len, input_size)
        LSTM   : n_layers=2, hidden=64, dropout=0.2
        Output : Linear(64→32) → SiLU → Dropout → Linear(32→1)
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size   = input_size,
            hidden_size  = hidden_size,
            num_layers   = n_layers,
            dropout      = dropout if n_layers > 1 else 0.0,
            batch_first  = True,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        """
        x   : (batch, seq_len, input_size)
        out : (batch, 1)
        """
        lstm_out, _ = self.lstm(x)         # (batch, seq_len, hidden)
        last_h = lstm_out[:, -1, :]        # lấy timestep cuối
        last_h = self.layer_norm(last_h)
        return self.fc_out(last_h)         # (batch, 1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════════════
# DATASET (dùng lại y hệt từ train_ltc_lstm.py)
# ══════════════════════════════════════════════════════════════════════════════
class PACDataset(Dataset):
    def __init__(self, csv_path, feature_cols, target, seq_len):
        df = pd.read_csv(csv_path, index_col='Ngay', parse_dates=True)
        self.X_data = df[feature_cols].values.astype(np.float32)
        self.y_data = df[target].values.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X_data) - self.seq_len

    def __getitem__(self, idx):
        X = self.X_data[idx: idx + self.seq_len]
        y = self.y_data[idx + self.seq_len]
        return torch.from_numpy(X), torch.tensor(y)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def inverse_transform(y_scaled, scaler_params, target):
    mn = scaler_params[target]['min']
    mx = scaler_params[target]['max']
    return y_scaled * (mx - mn) + mn


def compute_metrics(y_true, y_pred):
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2   = float(1 - ss_res / (ss_tot + 1e-10))
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}


def train_epoch(model, loader, optimizer, criterion, device, grad_clip):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X).squeeze(-1)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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
    return total_loss / len(loader.dataset), np.concatenate(preds), np.concatenate(trues)


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════
def plot_loss_curves(train_losses, val_losses, save_path):
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle('Training & Validation Loss — LSTM Baseline',
                 fontsize=13, color='#e8eaf6', fontweight='bold')
    ax.plot(train_losses, color='#06d6a0', linewidth=1.5, label='Train Loss (MSE)')
    ax.plot(val_losses,   color='#ffbe0b', linewidth=1.5, label='Val Loss (MSE)')
    best_ep = int(np.argmin(val_losses))
    ax.axvline(best_ep, color='#f72585', linestyle='--', linewidth=1.2,
               label=f'Best Val @ epoch {best_ep+1}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.legend()
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_predictions(y_true, y_pred, metrics, split_name, save_path):
    fig, axes = plt.subplots(2, 1, figsize=(16, 9))
    fig.suptitle(
        f'LSTM Baseline — PAC Forecasting ({split_name})  '
        f'MAE={metrics["MAE"]:.2f} mg/L  RMSE={metrics["RMSE"]:.2f}  R²={metrics["R2"]:.4f}',
        fontsize=12, color='#e8eaf6', fontweight='bold'
    )
    ax1 = axes[0]
    x_idx = np.arange(len(y_true))
    ax1.plot(x_idx, y_true, color='#06d6a0', linewidth=1.2, label='Actual PAC (mg/L)')
    ax1.plot(x_idx, y_pred, color='#ffbe0b', linewidth=1.2, label='Predicted PAC (mg/L)', alpha=0.85)
    ax1.fill_between(x_idx,
                     np.minimum(y_true, y_pred),
                     np.maximum(y_true, y_pred),
                     alpha=0.15, color='#fb5607', label='Error band')
    ax1.set_xlabel('Day index')
    ax1.set_ylabel('PAC Concentration (mg/L)')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.35)

    ax2 = axes[1]
    mn_val = min(y_true.min(), y_pred.min())
    mx_val = max(y_true.max(), y_pred.max())
    ax2.scatter(y_true, y_pred, alpha=0.35, s=20, color='#3a86ff')
    ax2.plot([mn_val, mx_val], [mn_val, mx_val], 'w--', linewidth=1.2, label='Perfect prediction')
    ax2.set_xlabel('Actual PAC (mg/L)')
    ax2.set_ylabel('Predicted PAC (mg/L)')
    ax2.set_title(f'Scatter: Actual vs Predicted  (R² = {metrics["R2"]:.4f})', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON PLOT — gọi sau khi cả 2 model đã chạy xong
# ══════════════════════════════════════════════════════════════════════════════
def plot_comparison(lstm_results, ltc_results, save_path):
    """Vẽ bảng so sánh metrics của 2 model."""
    metrics_order = ['MAE', 'RMSE', 'R2', 'MAPE']
    x = np.arange(len(metrics_order))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Model Comparison: LSTM Baseline vs Gated-LTC-LSTM',
                 fontsize=14, color='#e8eaf6', fontweight='bold')

    for ax_idx, (split, lstm_m, ltc_m) in enumerate([
        ('Validation (2021)', lstm_results['validation'], ltc_results['validation']),
        ('Test (2022)',        lstm_results['test'],       ltc_results['test']),
    ]):
        ax = axes[ax_idx]
        lstm_vals = [lstm_m[k] for k in metrics_order]
        ltc_vals  = [ltc_m[k]  for k in metrics_order]

        bars1 = ax.bar(x - width/2, lstm_vals, width, label='LSTM Baseline',
                       color='#06d6a0', alpha=0.85, edgecolor='none')
        bars2 = ax.bar(x + width/2, ltc_vals,  width, label='Gated-LTC-LSTM',
                       color='#f72585', alpha=0.85, edgecolor='none')

        ax.set_xticks(x)
        ax.set_xticklabels(metrics_order, fontsize=10)
        ax.set_title(split, fontsize=11, color='#a0c4ff')
        ax.legend(fontsize=9)
        ax.grid(True, axis='y', alpha=0.35)

        # Annotate bars
        for bar in bars1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=8, color='#e8eaf6')
        for bar in bars2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=8, color='#e8eaf6')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 62)
    print("  LSTM Baseline — PAC Forecasting")
    print("=" * 62)

    # ── 0. Preprocessing check ────────────────────────────────────────────────
    meta_path = os.path.join(CFG['data_dir'], 'metadata.json')
    if True:
        print(f"\n[0] Dùng lại data từ {CFG['data_dir']}/")

    with open(meta_path, encoding='utf-8') as f:
        meta = json.load(f)
    with open(os.path.join(CFG['data_dir'], 'scaler_params.json'), encoding='utf-8') as f:
        scaler_params = json.load(f)

    feature_cols = meta['feature_cols']
    target       = meta['target']
    n_features   = meta['n_features']
    device       = torch.device(CFG['device'])

    print(f"    Features: {n_features}  |  Seq len: {CFG['seq_len']}  |  Device: {device}")

    # ── 1. DataLoaders ────────────────────────────────────────────────────────
    train_ds = PACDataset(f"{CFG['data_dir']}/train.csv", feature_cols, target, CFG['seq_len'])
    val_ds   = PACDataset(f"{CFG['data_dir']}/val.csv",   feature_cols, target, CFG['seq_len'])
    test_ds  = PACDataset(f"{CFG['data_dir']}/test.csv",  feature_cols, target, CFG['seq_len'])

    train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'], shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG['batch_size'], shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=CFG['batch_size'], shuffle=False)

    print(f"\n[1] Datasets → Train:{len(train_ds)}  Val:{len(val_ds)}  Test:{len(test_ds)}")

    # ── 2. Model ──────────────────────────────────────────────────────────────
    model = LSTMBaseline(
        input_size  = n_features,
        hidden_size = CFG['hidden_size'],
        n_layers    = CFG['n_layers'],
        dropout     = CFG['dropout'],
    ).to(device)

    n_params = model.count_parameters()
    print(f"\n[2] Model: LSTM Baseline | Params: {n_params:,}")
    print(f"    hidden={CFG['hidden_size']}  layers={CFG['n_layers']}  dropout={CFG['dropout']}")

    # ── 3. Optimizer & Scheduler ──────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG['scheduler_T_max'], eta_min=1e-5)
    criterion = nn.MSELoss()

    # ── 4. Training loop ──────────────────────────────────────────────────────
    best_val_loss = float('inf')
    patience_cnt  = 0
    train_losses, val_losses = [], []
    best_ckpt = os.path.join(CFG['output_dir'], 'best_model_lstm.pt')

    print(f"\n[3] Training (max {CFG['epochs']} epochs, patience={CFG['patience']})...")
    print(f"    {'Epoch':>6} | {'Train Loss':>11} | {'Val Loss':>10} | {'LR':>10} | {'Time':>8}")
    print("    " + "-" * 56)

    t_start = time.time()
    for epoch in range(1, CFG['epochs'] + 1):
        t0 = time.time()
        tr_loss          = train_epoch(model, train_loader, optimizer, criterion,
                                       device, CFG['grad_clip'])
        val_loss, _, _   = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        lr_now  = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        if epoch % 10 == 0 or val_loss < best_val_loss:
            marker = ' *' if val_loss < best_val_loss else ''
            print(f"    {epoch:6d} | {tr_loss:11.6f} | {val_loss:10.6f} | "
                  f"{lr_now:10.2e} | {elapsed:6.1f}s{marker}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt  = 0
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'val_loss'   : best_val_loss,
            }, best_ckpt)
        else:
            patience_cnt += 1
            if patience_cnt >= CFG['patience']:
                print(f"\n    Early stopping at epoch {epoch}")
                break

    total_time = time.time() - t_start
    print(f"\n    Training done in {total_time:.1f}s | Best val loss: {best_val_loss:.6f}")

    # Plot loss
    plot_loss_curves(train_losses, val_losses,
                     os.path.join(CFG['output_dir'], 'loss_curves_lstm.png'))
    print(f"    Saved: {CFG['output_dir']}/loss_curves_lstm.png")

    # ── 5. Evaluation ─────────────────────────────────────────────────────────
    print(f"\n[4] Loading best model...")
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    print(f"    Best epoch: {ckpt['epoch']} | Val MSE: {ckpt['val_loss']:.6f}")

    lstm_results = {}
    for split_name, loader in [('Validation', val_loader), ('Test', test_loader)]:
        _, preds_s, trues_s = eval_epoch(model, loader, criterion, device)
        preds_real = inverse_transform(preds_s, scaler_params, target)
        trues_real = inverse_transform(trues_s, scaler_params, target)
        metrics    = compute_metrics(trues_real, preds_real)
        lstm_results[split_name.lower()] = metrics

        fname = f"{'val' if split_name=='Validation' else 'test'}_predictions_lstm.png"
        plot_predictions(trues_real, preds_real, metrics, split_name,
                         os.path.join(CFG['output_dir'], fname))
        print(f"    Saved: {CFG['output_dir']}/{fname}")

        print(f"\n  [{split_name}]")
        print(f"    MAE  = {metrics['MAE']:.4f} mg/L")
        print(f"    RMSE = {metrics['RMSE']:.4f} mg/L")
        print(f"    R²   = {metrics['R2']:.4f}")
        print(f"    MAPE = {metrics['MAPE']:.2f}%")

    # ── 6. Comparison với LTC-LSTM ─────────────────────────────────────────────
    ltc_results_path = 'outputs/results.json'
    if os.path.exists(ltc_results_path):
        with open(ltc_results_path, encoding='utf-8') as f:
            ltc_data = json.load(f)

        ltc_results = {
            'validation': ltc_data['validation'],
            'test'      : ltc_data['test'],
        }

        plot_comparison(
            lstm_results, ltc_results,
            os.path.join(CFG['output_dir'], 'model_comparison.png')
        )

        print("\n" + "=" * 62)
        print("  COMPARISON: LSTM Baseline vs Gated-LTC-LSTM")
        print("=" * 62)
        print(f"\n  {'Metric':<8} | {'LSTM Val':>10} | {'LTC Val':>10} | {'LSTM Test':>10} | {'LTC Test':>10}")
        print("  " + "-" * 58)
        for m in ['MAE', 'RMSE', 'R2', 'MAPE']:
            lv = lstm_results['validation'][m]
            tv = ltc_results['validation'][m]
            lt = lstm_results['test'][m]
            tt = ltc_results['test'][m]
            # Đánh dấu model tốt hơn (nhỏ hơn với MAE/RMSE/MAPE, lớn hơn với R2)
            if m == 'R2':
                tag_v = '← LTC' if tv > lv else '← LSTM'
                tag_t = '← LTC' if tt > lt else '← LSTM'
            else:
                tag_v = '← LTC' if tv < lv else '← LSTM'
                tag_t = '← LTC' if tt < lt else '← LSTM'
            print(f"  {m:<8} | {lv:10.4f} | {tv:10.4f} {tag_v:<8} | "
                  f"{lt:10.4f} | {tt:10.4f} {tag_t}")
    else:
        print("\n  (Không tìm thấy outputs/results.json — bỏ qua comparison chart)")

    # Save LSTM results
    with open(os.path.join(CFG['output_dir'], 'results_lstm.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'model'       : 'LSTM Baseline',
            'n_params'    : n_params,
            'best_epoch'  : int(ckpt['epoch']),
            'best_val_mse': float(ckpt['val_loss']),
            'validation'  : lstm_results['validation'],
            'test'        : lstm_results['test'],
        }, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 62)
    print("  LSTM BASELINE DONE!")
    print("=" * 62)
    print(f"""
  Output files (outputs_lstm/):
    best_model_lstm.pt
    loss_curves_lstm.png
    val_predictions_lstm.png
    test_predictions_lstm.png
    model_comparison.png      ← so sánh 2 model
    results_lstm.json
""")


if __name__ == '__main__':
    main()
