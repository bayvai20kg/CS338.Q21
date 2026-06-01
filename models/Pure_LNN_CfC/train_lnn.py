"""
train_lnn.py
============
Training pipeline cho Pure LNN (Closed-form CfC).

So sánh công bằng với LSTM baseline:
  - Cùng dataset, cùng seq_len, cùng batch_size
  - Cùng hidden_size=64, n_layers=2, dropout=0.2
  - LNN dùng lr thấp hơn (5e-4) vì CfC dynamics cần optimizer cẩn thận hơn
  - patience=30 (cho LNN thêm thời gian hội tụ)

Chạy: python -X utf8 train_lnn.py
"""

import os, json, time
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

from lnn_model import LNNModel

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
CFG = {
    'data_dir': '../../dataset_processed',
    'output_dir'     : 'outputs_lnn',
    'seq_len'        : 14,
    'batch_size'     : 32,
    # Model — giữ nguyên để so sánh công bằng với LSTM
    'hidden_size'    : 64,
    'n_layers'       : 2,
    'dropout'        : 0.2,
    'delta_t'        : 1.0,    # 1 ngày
    # Training — LNN cần lr nhỏ hơn, patience dài hơn
    'epochs'         : 300,
    'lr'             : 5e-4,   # thấp hơn LSTM (1e-3) → CfC cần landscape phẳng hơn
    'weight_decay'   : 1e-4,
    'patience'       : 30,     # dài hơn LSTM (20) để LNN đủ hội tụ
    'grad_clip'      : 1.0,
    'scheduler_T_max': 150,
    'seed'           : 42,
    'device'         : 'cpu',
}

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
    'legend.facecolor': '#1a1d2e', 'legend.edgecolor': '#3a3d5c',
    'legend.labelcolor': '#c8cce8',
})

PALETTE = {'lnn': '#f72585', 'lstm': '#06d6a0', 'actual': '#00b4d8'}

# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════
class PACDataset(Dataset):
    def __init__(self, csv_path, feature_cols, target, seq_len):
        df = pd.read_csv(csv_path, index_col='Ngay', parse_dates=True)
        self.X = df[feature_cols].values.astype(np.float32)
        self.y = df[target].values.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        return (torch.from_numpy(self.X[idx: idx + self.seq_len]),
                torch.tensor(self.y[idx + self.seq_len]))

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def inverse_transform(y_s, params, target):
    mn, mx = params[target]['min'], params[target]['max']
    return y_s * (mx - mn) + mn

def compute_metrics(y_true, y_pred):
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2   = float(1 - np.sum((y_true-y_pred)**2) / (np.sum((y_true-np.mean(y_true))**2)+1e-10))
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100)
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}

def train_epoch(model, loader, optimizer, criterion, device, grad_clip):
    model.train(); total = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X).squeeze(-1), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item() * len(y)
    return total / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval(); total = 0.0; ps, ts = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        p = model(X).squeeze(-1)
        total += criterion(p, y).item() * len(y)
        ps.append(p.cpu().numpy()); ts.append(y.cpu().numpy())
    return total/len(loader.dataset), np.concatenate(ps), np.concatenate(ts)

# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════
def plot_loss(train_l, val_l, path):
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle('Training & Validation Loss — Pure LNN (CfC)',
                 fontsize=13, color='#e8eaf6', fontweight='bold')
    ax.plot(train_l, color=PALETTE['lnn'], lw=1.5, label='Train Loss')
    ax.plot(val_l,   color='#ffbe0b',      lw=1.5, label='Val Loss')
    best = int(np.argmin(val_l))
    ax.axvline(best, color='#3a86ff', ls='--', lw=1.2, label=f'Best @ epoch {best+1}')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
    ax.legend(); ax.grid(True, alpha=0.35)
    plt.tight_layout(); plt.savefig(path, bbox_inches='tight'); plt.close()

def plot_preds(y_true, y_pred, metrics, split, path):
    fig, axes = plt.subplots(2, 1, figsize=(16, 9))
    fig.suptitle(
        f'Pure LNN (CfC) — PAC Forecasting ({split})  '
        f'MAE={metrics["MAE"]:.2f} mg/L  RMSE={metrics["RMSE"]:.2f}  R²={metrics["R2"]:.4f}',
        fontsize=12, color='#e8eaf6', fontweight='bold')
    ax1 = axes[0]
    xi = np.arange(len(y_true))
    ax1.plot(xi, y_true, color=PALETTE['actual'], lw=1.2, label='Actual PAC')
    ax1.plot(xi, y_pred, color=PALETTE['lnn'],    lw=1.2, label='LNN Predicted', alpha=0.85)
    ax1.fill_between(xi, np.minimum(y_true, y_pred), np.maximum(y_true, y_pred),
                     alpha=0.15, color='#ffbe0b', label='Error band')
    ax1.set_ylabel('PAC (mg/L)'); ax1.legend(fontsize=9); ax1.grid(True, alpha=0.35)
    ax2 = axes[1]
    lo = min(y_true.min(), y_pred.min()); hi = max(y_true.max(), y_pred.max())
    ax2.scatter(y_true, y_pred, alpha=0.35, s=20, color='#7209b7')
    ax2.plot([lo, hi], [lo, hi], 'w--', lw=1.2, label='Perfect')
    ax2.set_xlabel('Actual PAC'); ax2.set_ylabel('Predicted PAC')
    ax2.set_title(f'R² = {metrics["R2"]:.4f}', fontsize=10)
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.35)
    plt.tight_layout(); plt.savefig(path, bbox_inches='tight'); plt.close()

def plot_final_comparison(lnn_r, lstm_r, path):
    """Bar chart so sánh LNN vs LSTM trên cả val và test."""
    metrics = ['MAE', 'RMSE', 'R2', 'MAPE']
    splits  = [('Validation (2021)', 'validation'), ('Test (2022)', 'test')]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('🏁 Final Comparison: Pure LNN (CfC) vs LSTM Baseline',
                 fontsize=14, color='#e8eaf6', fontweight='bold')
    x = np.arange(len(metrics)); w = 0.35
    for ax, (title, key) in zip(axes, splits):
        lnn_v  = [lnn_r[key][m]  for m in metrics]
        lstm_v = [lstm_r[key][m] for m in metrics]
        b1 = ax.bar(x - w/2, lnn_v,  w, label='LNN (CfC)',     color=PALETTE['lnn'],  alpha=0.85, edgecolor='none')
        b2 = ax.bar(x + w/2, lstm_v, w, label='LSTM Baseline', color=PALETTE['lstm'], alpha=0.85, edgecolor='none')
        ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=10)
        ax.set_title(title, fontsize=11, color='#a0c4ff')
        ax.legend(fontsize=9); ax.grid(True, axis='y', alpha=0.35)
        for b in list(b1) + list(b2):
            h = b.get_height()
            ax.text(b.get_x()+b.get_width()/2, h+0.01, f'{h:.3f}',
                    ha='center', va='bottom', fontsize=8, color='#e8eaf6')
    plt.tight_layout(); plt.savefig(path, bbox_inches='tight'); plt.close()
    print(f"    Saved: {path}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 62)
    print("  Pure LNN (Closed-form CfC) — PAC Forecasting")
    print("=" * 62)

    # ── 0. Data ───────────────────────────────────────────────────────────────
    meta_path = f"{CFG['data_dir']}/metadata.json"
    if True:
        print(f"\n[0] Using cached data from {CFG['data_dir']}/")

    meta         = json.load(open(meta_path, encoding='utf-8'))
    scaler_params= json.load(open(f"{CFG['data_dir']}/scaler_params.json", encoding='utf-8'))
    feat_cols    = meta['feature_cols']
    target       = meta['target']
    n_feat       = meta['n_features']
    device       = torch.device(CFG['device'])

    print(f"    Features:{n_feat}  seq_len:{CFG['seq_len']}  device:{device}")

    # ── 1. DataLoaders ────────────────────────────────────────────────────────
    mk = lambda split, shuf: DataLoader(
        PACDataset(f"{CFG['data_dir']}/{split}.csv", feat_cols, target, CFG['seq_len']),
        batch_size=CFG['batch_size'], shuffle=shuf,
        drop_last=(shuf))
    train_loader = mk('train', True)
    val_loader   = mk('val',   False)
    test_loader  = mk('test',  False)
    print(f"\n[1] Datasets → Train:{len(train_loader.dataset)}  "
          f"Val:{len(val_loader.dataset)}  Test:{len(test_loader.dataset)}")

    # ── 2. Model ──────────────────────────────────────────────────────────────
    model = LNNModel(
        input_size  = n_feat,
        hidden_size = CFG['hidden_size'],
        n_layers    = CFG['n_layers'],
        dropout     = CFG['dropout'],
        delta_t     = CFG['delta_t'],
    ).to(device)

    n_params = model.count_parameters()
    print(f"\n[2] Model: Pure LNN (CfC) | Params: {n_params:,}")
    print(f"    hidden={CFG['hidden_size']}  layers={CFG['n_layers']}  "
          f"delta_t={CFG['delta_t']}  lr={CFG['lr']}")

    # ── 3. Optimizer + Scheduler ──────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG['scheduler_T_max'], eta_min=1e-6)
    criterion = nn.MSELoss()

    # ── 4. Training ───────────────────────────────────────────────────────────
    best_val = float('inf'); patience_cnt = 0
    train_ls, val_ls = [], []
    ckpt_path = f"{CFG['output_dir']}/best_model_lnn.pt"

    print(f"\n[3] Training (max {CFG['epochs']} epochs, patience={CFG['patience']})...")
    print(f"    {'Epoch':>6} | {'Train':>10} | {'Val':>10} | {'LR':>10} | {'Time':>7}")
    print("    " + "-" * 52)

    t0 = time.time()
    for ep in range(1, CFG['epochs'] + 1):
        t1 = time.time()
        tr = train_epoch(model, train_loader, optimizer, criterion, device, CFG['grad_clip'])
        vl, _, _ = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        train_ls.append(tr); val_ls.append(vl)
        lr_now = optimizer.param_groups[0]['lr']

        if ep % 10 == 0 or vl < best_val:
            tag = ' *' if vl < best_val else ''
            print(f"    {ep:6d} | {tr:10.6f} | {vl:10.6f} | "
                  f"{lr_now:10.2e} | {time.time()-t1:5.1f}s{tag}")

        if vl < best_val:
            best_val = vl; patience_cnt = 0
            torch.save({'epoch': ep, 'state': model.state_dict(), 'val_loss': vl}, ckpt_path)
        else:
            patience_cnt += 1
            if patience_cnt >= CFG['patience']:
                print(f"\n    Early stopping at epoch {ep}")
                break

    print(f"\n    Done in {time.time()-t0:.1f}s | Best val MSE: {best_val:.6f}")

    plot_loss(train_ls, val_ls, f"{CFG['output_dir']}/loss_curves_lnn.png")
    print(f"    Saved: {CFG['output_dir']}/loss_curves_lnn.png")

    # ── 5. Evaluation ─────────────────────────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['state'])
    print(f"\n[4] Best model: epoch {ckpt['epoch']} | Val MSE: {ckpt['val_loss']:.6f}")

    lnn_results = {}
    for sname, loader in [('Validation', val_loader), ('Test', test_loader)]:
        _, ps, ts = eval_epoch(model, loader, criterion, device)
        pr = inverse_transform(ps, scaler_params, target)
        tr = inverse_transform(ts, scaler_params, target)
        m  = compute_metrics(tr, pr)
        lnn_results[sname.lower()] = m

        fname = f"{'val' if sname=='Validation' else 'test'}_predictions_lnn.png"
        plot_preds(tr, pr, m, sname, f"{CFG['output_dir']}/{fname}")
        print(f"    Saved: {CFG['output_dir']}/{fname}")

        print(f"\n  [{sname}]")
        for k, v in m.items():
            unit = 'mg/L' if k in ('MAE','RMSE') else ('%' if k=='MAPE' else '')
            print(f"    {k:<6} = {v:.4f} {unit}")

    # ── 6. Comparison với LSTM ────────────────────────────────────────────────
    lstm_path = 'outputs_lstm/results_lstm.json'
    if os.path.exists(lstm_path):
        lstm_r = json.load(open(lstm_path, encoding='utf-8'))

        plot_final_comparison(
            lnn_results, lstm_r,
            f"{CFG['output_dir']}/comparison_lnn_vs_lstm.png"
        )

        print("\n" + "=" * 62)
        print("  COMPARISON: Pure LNN (CfC) vs LSTM Baseline")
        print("=" * 62)
        header = f"  {'Metric':<7}| {'LNN Val':>10} | {'LSTM Val':>10} | {'LNN Test':>10} | {'LSTM Test':>10} | Winner"
        print(header)
        print("  " + "-" * 68)
        for m_name in ['MAE', 'RMSE', 'R2', 'MAPE']:
            lv  = lnn_results['validation'][m_name]
            ltv = lstm_r['validation'][m_name]
            lt  = lnn_results['test'][m_name]
            ltt = lstm_r['test'][m_name]
            # R2: higher is better; others: lower is better
            better_val  = lv < ltv if m_name != 'R2' else lv > ltv
            better_test = lt < ltt if m_name != 'R2' else lt > ltt
            winner = '🏆 LNN' if (better_val and better_test) else \
                     '🏆 LSTM' if (not better_val and not better_test) else '~Tie'
            print(f"  {m_name:<7}| {lv:10.4f} | {ltv:10.4f} | {lt:10.4f} | {ltt:10.4f} | {winner}")

    # ── 7. Save ───────────────────────────────────────────────────────────────
    with open(f"{CFG['output_dir']}/results_lnn.json", 'w', encoding='utf-8') as f:
        json.dump({'model':'Pure LNN (CfC)', 'n_params': n_params,
                   'best_epoch': int(ckpt['epoch']),
                   'validation': lnn_results['validation'],
                   'test': lnn_results['test']}, f, indent=2)

    print(f"\n  Output files saved to: {CFG['output_dir']}/")
    print("=" * 62)


if __name__ == '__main__':
    main()
