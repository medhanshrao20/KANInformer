"""
KANInformer — Informer with KAN layers replacing Feed-Forward (MLP) blocks.

Architecture from paper Section 2.2.3 and source: https://github.com/375330014/lzy
Based on Informer: https://github.com/zhouhaoyi/Informer2020

Run as main script to train and evaluate on all 4 seasons:
  python KANInformer.py [--season spring] [--force]
"""

import argparse
import math
import os
import pickle
import random
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from device_utils import get_device
from metrics_utils import compute_all_metrics

warnings.filterwarnings('ignore')


# ============================================================
# KAN import
# ============================================================

def get_kan_class():
    """Import KAN from pykan (kan package)."""
    try:
        from kan import KAN
        return KAN
    except ImportError:
        pass
    try:
        from pykan import KAN
        return KAN
    except ImportError:
        pass
    raise ImportError(
        'Could not import KAN. Install pykan: pip install pykan\n'
        'or: pip install git+https://github.com/KindXiaoming/pykan.git'
    )


# ============================================================
# Positional Embedding
# ============================================================

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return self.pe[:, :x.size(1), :]


# ============================================================
# Token Embedding
# ============================================================

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode='circular',
            bias=False,
        )
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # x: (batch, seq_len, c_in)
        x = self.conv(x.permute(0, 2, 1))  # -> (batch, d_model, seq_len)
        return x.transpose(1, 2)            # -> (batch, seq_len, d_model)


# ============================================================
# Data Embedding
# ============================================================

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.05):
        super().__init__()
        self.token_embed = TokenEmbedding(c_in, d_model)
        self.pos_embed = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        tok = self.token_embed(x)
        pos = self.pos_embed(x)
        return self.dropout(tok + pos)


# ============================================================
# Attention Masks
# ============================================================

class TriangularCausalMask:
    def __init__(self, B, L, device='cpu'):
        mask_shape = (B, 1, L, L)
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask:
    def __init__(self, B, H, L, index, scores, device='cpu'):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[
            torch.arange(B)[:, None, None],
            torch.arange(H)[None, :, None],
            index,
            :
        ]
        self._mask = indicator.view(scores.shape)

    @property
    def mask(self):
        return self._mask


# ============================================================
# ProbSparse Attention
# ============================================================

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None,
                 attention_dropout=0.05, output_attention=False):
        super().__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # Sample K keys for sparsity measure
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k), device=Q.device)
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # Sparsity measure M
        M = Q_K_sample.max(-1)[0] - Q_K_sample.sum(-1) / L_K
        M_top = M.topk(n_top, sorted=False)[1]

        # Top-u queries full attention
        Q_reduce = Q[
            torch.arange(B)[:, None, None],
            torch.arange(H)[None, :, None],
            M_top, :
        ]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))
        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        V_mean = V.mean(dim=-2)
        contex = V_mean.unsqueeze(-2).expand(B, H, L_Q, D).clone()
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)

        context_in[
            torch.arange(B)[:, None, None],
            torch.arange(H)[None, :, None],
            index, :
        ] = torch.matmul(attn, V).type_as(context_in)

        if self.output_attention:
            attns = (torch.ones(B, H, L_Q, L_V) / L_V).type_as(attn)
            attns[
                torch.arange(B)[:, None, None],
                torch.arange(H)[None, :, None],
                index, :
            ] = attn
            return context_in, attns
        return context_in, None

    def forward(self, queries, keys, values, attn_mask=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)  # (B, H, L_Q, D)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * int(math.ceil(math.log(L_K + 1)))
        u = self.factor * int(math.ceil(math.log(L_Q + 1)))

        U_part = min(U_part, L_K)
        u = min(u, L_Q)

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        scale = self.scale or (1.0 / math.sqrt(D))
        scores_top = scores_top * scale

        context = self._get_initial_context(values, L_Q)
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous().transpose(2, 1), attn


# ============================================================
# Full (Standard) Attention
# ============================================================

class FullAttention(nn.Module):
    def __init__(self, mask_flag=False, scale=None,
                 attention_dropout=0.05, output_attention=False):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        scale = self.scale or (1.0 / math.sqrt(D))

        scores = torch.einsum('blhd,bshd->bhls', queries, keys) * scale

        if self.mask_flag and attn_mask is not None:
            if attn_mask.mask is not None:
                scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scores, dim=-1))
        V = torch.einsum('bhls,bshd->blhd', A, values)

        if self.output_attention:
            return V.contiguous(), A
        return V.contiguous(), None


# ============================================================
# Attention Layer (wraps ProbAttention or FullAttention)
# ============================================================

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_proj = nn.Linear(d_model, d_keys * n_heads)
        self.key_proj = nn.Linear(d_model, d_keys * n_heads)
        self.value_proj = nn.Linear(d_model, d_values * n_heads)
        self.out_proj = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.d_keys = d_keys
        self.d_values = d_values

    def forward(self, queries, keys, values, attn_mask=None):
        B, L_Q, _ = queries.shape
        _, L_K, _ = keys.shape
        H = self.n_heads

        Q = self.query_proj(queries).view(B, L_Q, H, self.d_keys)
        K = self.key_proj(keys).view(B, L_K, H, self.d_keys)
        V = self.value_proj(values).view(B, L_K, H, self.d_values)

        out, attn = self.inner_attention(Q, K, V, attn_mask)
        out = out.view(B, L_Q, H * self.d_values)
        return self.out_proj(out), attn


# ============================================================
# Conv Layer (between encoder layers for sequence distillation)
# ============================================================

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_in, kernel_size=3, padding=1, padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = self.conv(x.permute(0, 2, 1))  # (batch, d_model, seq_len)
        x = self.norm(x)
        x = self.activation(x)
        x = self.pool(x)
        return x.transpose(1, 2)  # (batch, seq_len//2, d_model)


# ============================================================
# Encoder Layer (with KAN replacing FFN)
# ============================================================

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, n_step, kan_class,
                 kan_grid=5, kan_k=3, dropout=0.05):
        super().__init__()
        self.attention = attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.n_step = n_step
        self.d_model = d_model

        # KAN replaces FFN; operates on flattened (batch, n_step*d_model)
        enc_width = d_model * n_step
        self.kan = kan_class(
            width=[enc_width, 20, enc_width],
            grid=kan_grid,
            k=kan_k,
        )

    def forward(self, x, attn_mask=None):
        # Self-attention
        new_x, attn = self.attention(x, x, x, attn_mask)
        x = self.norm1(x + self.dropout(new_x))

        # KAN block
        residual = x
        B, S, D = x.shape
        x_flat = x.reshape(B, S * D)
        x_flat = self.kan(x_flat)
        x = x_flat.reshape(B, S, D)
        x = self.norm2(residual + self.dropout(x))
        return x, attn


# ============================================================
# Encoder
# ============================================================

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers[:-1], self.conv_layers):
                x, attn = attn_layer(x, attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)
        return x, attns


# ============================================================
# Decoder Layer (with KAN replacing FFN)
# ============================================================

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, dec_seq_len,
                 kan_class, kan_grid=5, kan_k=3, dropout=0.05):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.dec_seq_len = dec_seq_len

        dec_width = d_model * dec_seq_len
        self.kan = kan_class(
            width=[dec_width, 20, dec_width],
            grid=kan_grid,
            k=kan_k,
        )

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # Masked self-attention
        new_x, _ = self.self_attention(x, x, x, x_mask)
        x = self.norm1(x + self.dropout(new_x))

        # Cross-attention with encoder output
        new_x, _ = self.cross_attention(x, cross, cross, cross_mask)
        x = self.norm2(x + self.dropout(new_x))

        # KAN block
        residual = x
        B, S, D = x.shape
        x_flat = x.reshape(B, S * D)
        x_flat = self.kan(x_flat)
        x = x_flat.reshape(B, S, D)
        x = self.norm3(residual + self.dropout(x))
        return x


# ============================================================
# Decoder
# ============================================================

class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask, cross_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x


# ============================================================
# Full KANInformer Model
# ============================================================

class KANInformer(nn.Module):
    def __init__(self, n_heng, n_step, n_out, d_model, n_heads, e_layers, d_layers,
                 factor, dropout, kan_grid, kan_k, kan_class):
        super().__init__()
        self.n_step = n_step
        self.n_out = n_out
        self.d_model = d_model

        # Decoder sequence length = (n_step - 1) + n_out
        self.dec_seq_len = (n_step - 1) + n_out

        # Embeddings
        self.enc_embedding = DataEmbedding(n_heng, d_model, dropout)
        self.dec_embedding = DataEmbedding(n_heng, d_model, dropout)

        # Encoder
        enc_layers = []
        for _ in range(e_layers):
            enc_layers.append(EncoderLayer(
                attention=AttentionLayer(
                    ProbAttention(mask_flag=False, factor=factor,
                                  attention_dropout=dropout),
                    d_model, n_heads
                ),
                d_model=d_model,
                n_step=n_step,
                kan_class=kan_class,
                kan_grid=kan_grid,
                kan_k=kan_k,
                dropout=dropout,
            ))

        conv_layers = [ConvLayer(d_model) for _ in range(e_layers - 1)]

        self.encoder = Encoder(
            attn_layers=enc_layers,
            conv_layers=conv_layers,
            norm_layer=nn.LayerNorm(d_model),
        )

        # Decoder
        dec_layers = []
        for _ in range(d_layers):
            dec_layers.append(DecoderLayer(
                self_attention=AttentionLayer(
                    ProbAttention(mask_flag=True, factor=factor,
                                  attention_dropout=dropout),
                    d_model, n_heads
                ),
                cross_attention=AttentionLayer(
                    FullAttention(mask_flag=False,
                                  attention_dropout=dropout),
                    d_model, n_heads
                ),
                d_model=d_model,
                dec_seq_len=self.dec_seq_len,
                kan_class=kan_class,
                kan_grid=kan_grid,
                kan_k=kan_k,
                dropout=dropout,
            ))

        self.decoder = Decoder(
            layers=dec_layers,
            norm_layer=nn.LayerNorm(d_model),
        )

        self.projection = nn.Linear(d_model, 1, bias=True)

    def forward(self, x_enc):
        """
        x_enc: (batch, n_step, n_heng)
        returns: (batch, n_out)  — predicted WS for t+1, t+2, t+3
        """
        B, N_S, N_H = x_enc.shape

        # Encoder
        enc_emb = self.enc_embedding(x_enc)
        enc_out, _ = self.encoder(enc_emb)

        # Decoder input: last (n_step-1) encoder inputs + n_out zeros
        token = x_enc[:, 1:, :]                             # (B, n_step-1, n_heng)
        zeros_pad = torch.zeros(B, self.n_out, N_H, device=x_enc.device)
        x_dec = torch.cat([token, zeros_pad], dim=1)        # (B, dec_seq_len, n_heng)

        dec_emb = self.dec_embedding(x_dec)

        # Decoder mask for self-attention
        dec_mask = TriangularCausalMask(B, self.dec_seq_len, device=x_enc.device)

        dec_out = self.decoder(dec_emb, enc_out, x_mask=dec_mask)

        # Take last n_out positions, project to scalar
        out = self.projection(dec_out[:, -self.n_out:, :])  # (B, n_out, 1)
        return out.squeeze(-1)                               # (B, n_out)


# ============================================================
# Training and Evaluation
# ============================================================

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_season(season, device, kan_class, args):
    print(f'\n{"="*60}')
    print(f'  TRAINING: {season.upper()}')
    print(f'{"="*60}')

    # Check if model already trained
    model_path = os.path.join(config.MODEL_DIR, f'{season}_best_model.pt')
    pred_path = os.path.join(config.OUTPUTS_DIR, f'{season}_predictions.csv')
    if not args.force and os.path.exists(pred_path):
        print(f'  Predictions already exist for {season}. Skipping training.')
        return None

    set_seeds(config.RANDOM_SEED)

    # Load windowed data
    X_train = np.load(os.path.join(config.PROCESSED_DIR, f'{season}_X_train.npy'))
    Y_train = np.load(os.path.join(config.PROCESSED_DIR, f'{season}_Y_train.npy'))
    X_val = np.load(os.path.join(config.PROCESSED_DIR, f'{season}_X_val.npy'))
    Y_val = np.load(os.path.join(config.PROCESSED_DIR, f'{season}_Y_val.npy'))
    X_test = np.load(os.path.join(config.PROCESSED_DIR, f'{season}_X_test.npy'))
    Y_test = np.load(os.path.join(config.PROCESSED_DIR, f'{season}_Y_test.npy'))

    n_heng = X_train.shape[2]
    print(f'  n_heng={n_heng}, X_train={X_train.shape}, Y_train={Y_train.shape}')
    print(f'  X_val={X_val.shape}, X_test={X_test.shape}')

    # Tensors and DataLoaders
    train_ds = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(Y_train),
    )
    val_ds = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(Y_val),
    )
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    # Model
    model = KANInformer(
        n_heng=n_heng,
        n_step=config.N_STEP,
        n_out=config.N_OUT,
        d_model=config.D_MODEL,
        n_heads=config.N_HEADS,
        e_layers=config.E_LAYERS,
        d_layers=config.D_LAYERS,
        factor=config.FACTOR,
        dropout=config.DROPOUT,
        kan_grid=config.KAN_GRID,
        kan_k=config.KAN_K,
        kan_class=kan_class,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Model parameters: {n_params:,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    for epoch in range(1, config.MAX_EPOCHS + 1):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            pred = model(X_batch)
            loss = criterion(pred, Y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_vb, Y_vb in val_loader:
                X_vb = X_vb.to(device)
                Y_vb = Y_vb.to(device)
                val_pred = model(X_vb)
                val_loss += criterion(val_pred, Y_vb).item()
        val_loss /= len(val_loader)

        print(f'  Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f'  Early stopping triggered at epoch {epoch}')
                break

    # Load best model
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f'  Loaded best model from {model_path}')

    # Evaluate on test set
    metrics = evaluate_season(season, model, X_test, Y_test, device)
    return metrics


def evaluate_season(season, model, X_test, Y_test, device):
    """Run inference, inverse transform, compute metrics, save predictions."""

    # Load WS scaler
    scaler_path = os.path.join(config.SEASONS_DIR, season, 'scaler.pkl')
    with open(scaler_path, 'rb') as f:
        scalers = pickle.load(f)
    ws_scaler = scalers[-1]

    model.eval()
    X_tensor = torch.FloatTensor(X_test).to(device)

    with torch.no_grad():
        pred_norm = model(X_tensor).cpu().numpy()  # (N_test, 3)

    metrics_per_step = {}
    pred_cols = {}
    actual_cols = {}

    for s in range(config.N_OUT):
        pred_ms = ws_scaler.inverse_transform(
            pred_norm[:, s].reshape(-1, 1)
        ).flatten()
        actual_ms = ws_scaler.inverse_transform(
            Y_test[:, s].reshape(-1, 1)
        ).flatten()

        step = s + 1
        m = compute_all_metrics(actual_ms, pred_ms)
        metrics_per_step[step] = m

        pred_cols[f'actual_{step}step'] = actual_ms
        pred_cols[f'pred_{step}step'] = pred_ms

        print(f'  Step {step}: RMSE={m["rmse"]:.4f}, MAE={m["mae"]:.4f}, '
              f'MAPE={m["mape"]:.2f}%')

        exp = config.EXPECTED_RESULTS.get(season, {}).get(step, {})
        if exp:
            print(f'    Paper: RMSE={exp["rmse"]}, MAE={exp["mae"]}, MAPE={exp["mape"]}%')

    # Save predictions
    import pandas as pd
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    pred_df = pd.DataFrame(pred_cols)
    pred_path = os.path.join(config.OUTPUTS_DIR, f'{season}_predictions.csv')
    pred_df.to_csv(pred_path, index=False)
    print(f'  Saved predictions to {pred_path}')

    return metrics_per_step


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--force', action='store_true', help='Retrain even if predictions exist')
    p.add_argument('--season', type=str, default=None, help='Single season to train')
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()

    kan_class = get_kan_class()
    print(f'KAN class: {kan_class}')

    seasons = [args.season] if args.season else config.SEASON_ORDER
    all_metrics = {}

    for season in seasons:
        if season not in config.SEASON_DATES:
            print(f'Unknown season: {season}')
            continue
        metrics = train_season(season, device, kan_class, args)
        if metrics:
            all_metrics[season] = metrics

    # Print summary table
    if all_metrics:
        print('\n' + '='*80)
        print('RESULTS SUMMARY vs PAPER EXPECTED')
        print('='*80)
        header = f'{"Season":8s} | {"Step":4s} | {"RMSE(ours)":10s} | {"RMSE(paper)":11s} | {"MAE(ours)":9s} | {"MAE(paper)":10s} | {"MAPE(ours)":10s} | {"MAPE(paper)":11s}'
        print(header)
        print('-' * len(header))

        import pandas as pd
        rows = []
        for season, steps in all_metrics.items():
            for step, m in steps.items():
                exp = config.EXPECTED_RESULTS.get(season, {}).get(step, {})
                row_str = (f'{season:8s} | {step:4d} | {m["rmse"]:10.4f} | '
                           f'{exp.get("rmse", "N/A"):11} | {m["mae"]:9.4f} | '
                           f'{exp.get("mae", "N/A"):10} | {m["mape"]:10.2f} | '
                           f'{exp.get("mape", "N/A"):11}')
                print(row_str)
                rows.append({
                    'Season': season, 'Step': step,
                    'RMSE_ours': round(m['rmse'], 4),
                    'RMSE_paper': exp.get('rmse', ''),
                    'MAE_ours': round(m['mae'], 4),
                    'MAE_paper': exp.get('mae', ''),
                    'MAPE_ours': round(m['mape'], 2),
                    'MAPE_paper': exp.get('mape', ''),
                })

        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        result_df = pd.DataFrame(rows)
        result_path = os.path.join(config.RESULTS_DIR, 'table9_results.csv')
        result_df.to_csv(result_path, index=False)
        print(f'\nSaved results table to {result_path}')


if __name__ == '__main__':
    main()
