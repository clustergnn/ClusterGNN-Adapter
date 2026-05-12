import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args, scheduler=None, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate * 1}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
        
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.plot(true, label='GroundTruth', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def visual_forecast(
    input_1d,
    upgrade_input,
    true_1d,
    pred_1d,
    save_path,
    title=None,
):
    import numpy as np
    import os
    from matplotlib.patches import FancyBboxPatch
    
    # --- [1] Paper-style visualization settings ---
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif", "serif"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.unicode_minus": False
    })

    S, P = len(input_1d), len(true_1d)
    x_input = np.arange(S)
    x_future = np.arange(S, S + P)

    fig, ax = plt.subplots(figsize=(6.5, 2.6))  

    # --- [2] Color palette ---
    c_actual = '#e74c3c'   # Red
    c_cluster = '#2980b9'  # Blue
    c_pred = '#0000FF'     # Pure Blue
    c_divider = '#555555'  # Gray

    # --- [3] Ground Truth (History + Future) ---
    ax.plot(x_input, input_1d, color=c_actual, linewidth=1.2)
    ax.plot(
        x_future,
        true_1d,
        color=c_actual,
        linewidth=1.2,
        label='Ground Truth',
        zorder=3
    )

    # Connection line (History -> Future GT)
    ax.plot(
        [S - 1, S],
        [input_1d[-1], true_1d[0]],
        color=c_actual,
        linewidth=1.2,
        alpha=0.4
    )

    # --- [4] Cluster-Enhanced Input ---
    if upgrade_input is not None:
        ax.plot(
            x_input,
            upgrade_input,
            color=c_cluster,
            linewidth=1.2,
            label='Cluster-Fused Input',
            zorder=2
        )

    # --- [5] Forecast Divider ---
    ax.axvline(
        x=S - 0.5,
        color=c_divider,
        linestyle=':',
        linewidth=1.0
    )

    # --- [6] Prediction ---
    ax.plot(
        x_future,
        pred_1d,
        color=c_pred,
        linewidth=1.8,
        linestyle='-',
        label='Prediction',
        zorder=4
    )

    # Connection line (History -> Prediction)
    ax.plot(
        [S - 1, S],
        [input_1d[-1], pred_1d[0]],
        color=c_pred,
        linewidth=1.8,
        alpha=0.5
    )

    # --- [7] Styling ---
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle=':', alpha=0.4)

    ax.set_xlabel('Time Step', fontsize=9)
    ax.set_ylabel('Value', fontsize=9)

    # Highlight title
    if title is not None:
        ax.set_title(
            title,
            fontsize=14,       # ↑
            fontweight='bold', # bold
            pad=8
        )

    # --- [8] Force legend order ---
    handles, labels = ax.get_legend_handles_labels()

    desired_order = [
        'Ground Truth',
        'Prediction',
        'Cluster-Fused Input'
    ]

    ordered_handles = []
    ordered_labels = []

    for name in desired_order:
        if name in labels:
            idx = labels.index(name)
            ordered_handles.append(handles[idx])
            ordered_labels.append(labels[idx])

    ax.legend(
        ordered_handles,
        ordered_labels,
        loc='lower left',
        frameon=True,
        fontsize=8,        
        ncol=1,
        handlelength=1.0,  
        handletextpad=0.4,  
        borderpad=0.2,
        framealpha=0.5
    )

    plt.tight_layout()

    
    # --- [9] Save figure ---
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    plt.savefig(
        save_path,
        format='pdf',
        bbox_inches='tight'
    )
    plt.close()
    
def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, -1, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

