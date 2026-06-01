"""
train_lnn_lstm.py
=================
Training pipeline: LNN + LSTM (không Gated Input) vs LSTM Baseline.

Chạy: python -X utf8 train_lnn_lstm.py
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

from lnn_lstm_model import LNNLSTMModel

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
CFG = {
    'data_dir': '../../dataset_processed',
    'output_dir'      : 'outputs_lnn_lstm',
    'seq_len'         : 14,
    'batch_size'      : 32,
    # Model — cùng hidden_size với LSTM để so sánh công bằng
    'hidden_size'     : 64,
    'n_layers'        : 2,
    'dropout'         : 0.2,
    'delta_t'         : 1.0,   # CfC: timestep (1 ngày)
    # Training
    'epochs'          : 300,
    'lr'              : 8e-4,   # CfC hội tụ tốt hơn Euler → có thể dùng lr cao hơn
    'weight_decay'    : 1e-4,
    'patience'        : 30,
    'grad_clip'       : 1.0,
    'scheduler_T_max' : 150,
    'seed'            : 42,
    'device'          : 'cpu',
}

torch.manual_seed(CFG['seed']); np.random.seed(CFG['seed'])
os.makedirs(CFG['output_dir'], exist_ok=True)

plt.rcParams.update({
    'figure.facecolor': '#0f1117', 'axes.facecolor': '#1a1d2e',
    'axes.edgecolor':   '#3a3d5c', 'axes.labelcolor': '#c8cce8',
    'xtick.color':      '#8890c8', 'ytick.color':     '#8890c8',
    'text.color':       '#e8eaf6', 'grid.color':      '#2a2d4a',
    'grid.linewidth':   0.6,       'figure.dpi':      120,
    'savefig.dpi':      150,       'savefig.facecolor':'#0f1117',
    'axes.titlecolor':  '#a0c4ff', 'axes.titleweight': 'bold',
    'legend.facecolor': '#1a1d2e', 'legend.edgecolor': '#3a3d5c',
    'legend.labelcolor':'#c8cce8',
})

# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════
class PACDataset(Dataset):
    def __init__(self, csv_path, feature_cols, target, seq_len):
        df = pd.read_csv(csv_path, index_col='Ngay', parse_dates=True)
        self.X = df[feature_cols].values.astype(np.float32)
        self.y = df[target].values.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self): return len(self.X) - self.seq_len
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i:i+self.seq_len]), torch.tensor(self.y[i+self.seq_len])

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def inv(y, params, target):
    mn, mx = params[target]['min'], params[target]['max']
    return y * (mx - mn) + mn

def metrics(y_true, y_pred):
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    r2   = float(1 - np.sum((y_true-y_pred)**2) / (np.sum((y_true-np.mean(y_true))**2)+1e-10))
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100)
    return {'MAE':mae, 'RMSE':rmse, 'R2':r2, 'MAPE':mape}

def train_one(model, loader, opt, crit, dev, clip):
    model.train(); tot = 0.0
    for X, y in loader:
        X, y = X.to(dev), y.to(dev); opt.zero_grad()
        loss = crit(model(X).squeeze(-1), y); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step(); tot += loss.item() * len(y)
    return tot / len(loader.dataset)

@torch.no_grad()
def eval_one(model, loader, crit, dev):
    model.eval(); tot = 0.0; ps, ts = [], []
    for X, y in loader:
        X, y = X.to(dev), y.to(dev); p = model(X).squeeze(-1)
        tot += crit(p, y).item() * len(y)
        ps.append(p.cpu().numpy()); ts.append(y.cpu().numpy())
    return tot/len(loader.dataset), np.concatenate(ps), np.concatenate(ts)

# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════
def plot_loss(tr, vl, path, title):
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(title, fontsize=13, color='#e8eaf6', fontweight='bold')
    ax.plot(tr, color='#f72585', lw=1.5, label='Train Loss')
    ax.plot(vl, color='#ffbe0b', lw=1.5, label='Val Loss')
    best = int(np.argmin(vl))
    ax.axvline(best, color='#3a86ff', ls='--', lw=1.2, label=f'Best @ epoch {best+1}')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
    ax.legend(); ax.grid(True, alpha=0.35)
    plt.tight_layout(); plt.savefig(path, bbox_inches='tight'); plt.close()
    print(f"    Saved: {path}")

def plot_preds(y_true, y_pred, m, split, path, color):
    fig, axes = plt.subplots(2, 1, figsize=(16, 9))
    fig.suptitle(
        f'LNN+LSTM — PAC Forecasting ({split})  '
        f'MAE={m["MAE"]:.2f} mg/L  RMSE={m["RMSE"]:.2f}  R²={m["R2"]:.4f}',
        fontsize=12, color='#e8eaf6', fontweight='bold')
    xi = np.arange(len(y_true))
    axes[0].plot(xi, y_true, color='#00b4d8', lw=1.2, label='Actual PAC')
    axes[0].plot(xi, y_pred, color=color, lw=1.2, label='LNN+LSTM Predicted', alpha=0.85)
    axes[0].fill_between(xi, np.minimum(y_true,y_pred), np.maximum(y_true,y_pred),
                         alpha=0.15, color='#ffbe0b', label='Error band')
    axes[0].set_ylabel('PAC (mg/L)'); axes[0].legend(fontsize=9); axes[0].grid(True,alpha=0.35)
    lo=min(y_true.min(),y_pred.min()); hi=max(y_true.max(),y_pred.max())
    axes[1].scatter(y_true, y_pred, alpha=0.35, s=20, color=color)
    axes[1].plot([lo,hi],[lo,hi],'w--',lw=1.2,label='Perfect')
    axes[1].set_xlabel('Actual PAC'); axes[1].set_ylabel('Predicted PAC')
    axes[1].set_title(f'R² = {m["R2"]:.4f}', fontsize=10)
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.35)
    plt.tight_layout(); plt.savefig(path, bbox_inches='tight'); plt.close()
    print(f"    Saved: {path}")

def plot_comparison(lnn_r, lstm_r, path):
    metric_list = ['MAE','RMSE','R2','MAPE']
    splits = [('Validation (2021)','validation'), ('Test (2022)','test')]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('🏁 Comparison: LNN+LSTM  vs  LSTM Baseline',
                 fontsize=14, color='#e8eaf6', fontweight='bold')
    x = np.arange(len(metric_list)); w = 0.35
    for ax, (title, key) in zip(axes, splits):
        lnn_v  = [lnn_r[key][m]  for m in metric_list]
        lstm_v = [lstm_r[key][m] for m in metric_list]
        b1 = ax.bar(x-w/2, lnn_v,  w, label='LNN+LSTM', color='#f72585', alpha=0.85, edgecolor='none')
        b2 = ax.bar(x+w/2, lstm_v, w, label='LSTM',     color='#06d6a0', alpha=0.85, edgecolor='none')
        ax.set_xticks(x); ax.set_xticklabels(metric_list, fontsize=10)
        ax.set_title(title, fontsize=11, color='#a0c4ff')
        ax.legend(fontsize=9); ax.grid(True, axis='y', alpha=0.35)
        for b in list(b1)+list(b2):
            h = b.get_height()
            ax.text(b.get_x()+b.get_width()/2, h+0.005, f'{h:.3f}',
                    ha='center', va='bottom', fontsize=8, color='#e8eaf6')
    plt.tight_layout(); plt.savefig(path, bbox_inches='tight'); plt.close()
    print(f"    Saved: {path}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 62)
    print("  LNN + LSTM (không Gated Input) — PAC Forecasting")
    print("=" * 62)

    # ── Data ──────────────────────────────────────────────────────────────────
    meta_path = f"{CFG['data_dir']}/metadata.json"
    if True:
        print(f"\n[0] Dùng lại data từ {CFG['data_dir']}/")

    meta    = json.load(open(meta_path, encoding='utf-8'))
    scaler  = json.load(open(f"{CFG['data_dir']}/scaler_params.json", encoding='utf-8'))
    feats   = meta['feature_cols']
    target  = meta['target']
    n_feat  = meta['n_features']
    device  = torch.device(CFG['device'])
    print(f"    Features:{n_feat}  seq_len:{CFG['seq_len']}  device:{device}")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    mk = lambda s, sh: DataLoader(
        PACDataset(f"{CFG['data_dir']}/{s}.csv", feats, target, CFG['seq_len']),
        batch_size=CFG['batch_size'], shuffle=sh, drop_last=sh)
    tr_l = mk('train', True); va_l = mk('val', False); te_l = mk('test', False)
    print(f"\n[1] Train:{len(tr_l.dataset)}  Val:{len(va_l.dataset)}  Test:{len(te_l.dataset)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = LNNLSTMModel(
        input_size  = n_feat,
        hidden_size = CFG['hidden_size'],
        n_layers    = CFG['n_layers'],
        dropout     = CFG['dropout'],
        delta_t     = CFG['delta_t'],
    ).to(device)

    n_params = model.count_parameters()
    print(f"\n[2] Model: LNN+LSTM (CfC) | Params: {n_params:,}")
    print(f"    hidden={CFG['hidden_size']}  layers={CFG['n_layers']}  "
          f"delta_t={CFG['delta_t']}  lr={CFG['lr']}")

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    opt   = torch.optim.Adam(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG['scheduler_T_max'], eta_min=1e-6)
    crit  = nn.MSELoss()

    # ── Training ──────────────────────────────────────────────────────────────
    best_val = float('inf'); pat = 0; tr_ls = []; va_ls = []
    ckpt = f"{CFG['output_dir']}/best_model_lnn_lstm.pt"

    print(f"\n[3] Training (max {CFG['epochs']} epochs, patience={CFG['patience']})...")
    print(f"    {'Epoch':>6} | {'Train':>10} | {'Val':>10} | {'LR':>10} | {'Time':>7}")
    print("    " + "-" * 52)

    t0 = time.time()
    for ep in range(1, CFG['epochs']+1):
        t1 = time.time()
        tr = train_one(model, tr_l, opt, crit, device, CFG['grad_clip'])
        vl, _, _ = eval_one(model, va_l, crit, device)
        sched.step(); tr_ls.append(tr); va_ls.append(vl)
        lr_now = opt.param_groups[0]['lr']

        if ep % 10 == 0 or vl < best_val:
            tag = ' *' if vl < best_val else ''
            print(f"    {ep:6d} | {tr:10.6f} | {vl:10.6f} | {lr_now:10.2e} | {time.time()-t1:5.1f}s{tag}")

        if vl < best_val:
            best_val = vl; pat = 0
            torch.save({'epoch':ep,'state':model.state_dict(),'val_loss':vl}, ckpt)
        else:
            pat += 1
            if pat >= CFG['patience']:
                print(f"\n    Early stopping at epoch {ep}"); break

    print(f"\n    Done in {time.time()-t0:.1f}s | Best val MSE: {best_val:.6f}")
    plot_loss(tr_ls, va_ls, f"{CFG['output_dir']}/loss_lnn_lstm.png",
              'Training & Validation Loss — LNN+LSTM')

    # ── Evaluation ────────────────────────────────────────────────────────────
    saved = torch.load(ckpt, map_location=device)
    model.load_state_dict(saved['state'])
    print(f"\n[4] Best epoch:{saved['epoch']}  Val MSE:{saved['val_loss']:.6f}")

    results = {}
    for sname, loader, col in [
        ('Validation', va_l, '#f72585'),
        ('Test',       te_l, '#7209b7'),
    ]:
        _, ps, ts = eval_one(model, loader, crit, device)
        pr = inv(ps, scaler, target); tr_ = inv(ts, scaler, target)
        m  = metrics(tr_, pr)
        results[sname.lower()] = m
        fname = f"{'val' if sname=='Validation' else 'test'}_preds_lnn_lstm.png"
        plot_preds(tr_, pr, m, sname, f"{CFG['output_dir']}/{fname}", col)
        print(f"\n  [{sname}]")
        for k, v in m.items():
            unit = 'mg/L' if k in ('MAE','RMSE') else ('%' if k=='MAPE' else '')
            print(f"    {k:<6}= {v:.4f} {unit}")

    # ── Comparison ────────────────────────────────────────────────────────────
    lstm_path = 'outputs_lstm/results_lstm.json'
    if os.path.exists(lstm_path):
        lstm_r = json.load(open(lstm_path, encoding='utf-8'))
        plot_comparison(results, lstm_r, f"{CFG['output_dir']}/comparison_lnn_lstm_vs_lstm.png")

        print("\n" + "=" * 65)
        print("  COMPARISON: LNN+LSTM  vs  LSTM Baseline")
        print("=" * 65)
        print(f"  {'':7}| {'LNN+LSTM Val':>13} | {'LSTM Val':>9} | {'LNN+LSTM Test':>13} | {'LSTM Test':>9} | Winner")
        print("  " + "-" * 72)
        for mn in ['MAE','RMSE','R2','MAPE']:
            lv  = results['validation'][mn]
            ltv = lstm_r['validation'][mn]
            lt  = results['test'][mn]
            ltt = lstm_r['test'][mn]
            better_v = lv < ltv if mn != 'R2' else lv > ltv
            better_t = lt < ltt if mn != 'R2' else lt > ltt
            win = '🏆 LNN+LSTM' if (better_v and better_t) else \
                  '🏆 LSTM' if (not better_v and not better_t) else '~ Tie'
            print(f"  {mn:<7}| {lv:13.4f} | {ltv:9.4f} | {lt:13.4f} | {ltt:9.4f} | {win}")

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(f"{CFG['output_dir']}/results_lnn_lstm.json", 'w', encoding='utf-8') as f:
        json.dump({'model':'LNN+LSTM','n_params':n_params,
                   'best_epoch':int(saved['epoch']),
                   'validation':results['validation'],
                   'test':results['test']}, f, indent=2)
    print(f"\n  Files saved → {CFG['output_dir']}/")
    print("=" * 62)

if __name__ == '__main__':
    main()
