#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CEEMDAN + EWT preprocessing paired with advanced PyTorch regressors (TCN,
Informer-lite, BiLSTM + Attention). The script mirrors the data-handling
ideas used in myfunctions.py/myfunctions_france.py but runs entirely on PyTorch
with GPU support whenever available.

Example:
    python ceemdan_ewt_models.py \
        --data_path dataset/final_la_haute_R0711.csv \
        --target_col "P_avg" \
        --month_col "Month" \
        --months 1 2 3 \
        --model tcn \
        --look_back 48 \
        --train_ratio 0.8 \
        --cap 2100
"""

import argparse
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PyEMD import CEEMDAN
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import ewtpy


# -----------------------------
# Reproducibility helpers
# -----------------------------
def set_seed(seed: int = 1234) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# Data utilities
# -----------------------------
def create_dataset(series: np.ndarray, look_back: int) -> Tuple[np.ndarray, np.ndarray]:
    data_x, data_y = [], []
    for idx in range(len(series) - look_back):
        window = series[idx : idx + look_back]
        data_x.append(window)
        data_y.append(series[idx + look_back])
    return np.asarray(data_x, dtype=np.float32), np.asarray(data_y, dtype=np.float32)


def filter_by_month(
    df: pd.DataFrame,
    months: Optional[List[int]],
    month_col: Optional[str],
    timestamp_col: Optional[str],
) -> pd.DataFrame:
    if not months:
        return df

    months = [int(m) for m in months]

    if month_col and month_col in df.columns:
        return df[df[month_col].isin(months)]

    if timestamp_col and timestamp_col in df.columns:
        if not np.issubdtype(df[timestamp_col].dtype, np.datetime64):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df["_tmp_month"] = df[timestamp_col].dt.month
        filtered = df[df["_tmp_month"].isin(months)].copy()
        filtered.drop(columns=["_tmp_month"], inplace=True)
        return filtered

    raise ValueError(
        "Không tìm thấy cột tháng. Cung cấp month_col hoặc timestamp_col hợp lệ."
    )


def load_series(
    data_path: str,
    target_col: str,
    timestamp_col: Optional[str],
    month_col: Optional[str],
    months: Optional[List[int]],
) -> pd.Series:
    df = pd.read_csv(data_path)
    if timestamp_col and timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    df = filter_by_month(df, months, month_col, timestamp_col)
    if target_col not in df.columns:
        raise ValueError(f"Cột {target_col} không tồn tại trong dữ liệu.")

    series = df[target_col].astype(np.float32).dropna().reset_index(drop=True)
    if len(series) < 10:
        raise ValueError("Số lượng quan sát quá ít sau khi lọc.")
    return series


# -----------------------------
# CEEMDAN + EWT decomposition
# -----------------------------
def ceemdan_ewt_decompose(
    series: pd.Series,
    num_ewt_components: int = 3,
    epsilon: float = 0.05,
    seed: int = 1234,
) -> pd.DataFrame:
    ceemdan = CEEMDAN(epsilon=epsilon)
    ceemdan.noise_seed(seed)
    imfs = ceemdan(series.values)
    imf_df = pd.DataFrame(imfs).T

    first_imf = imf_df.iloc[:, 0].values
    ewt, _, _ = ewtpy.EWT1D(first_imf, N=num_ewt_components)
    ewt_df = pd.DataFrame(ewt).T

    if ewt_df.shape[1] >= 3:
        ewt_df = ewt_df.drop(columns=ewt_df.columns[-1], errors="ignore")
    denoised_imf = ewt_df.sum(axis=1)

    components = pd.concat([denoised_imf, imf_df.iloc[:, 1:]], axis=1)
    components.columns = [f"component_{idx}" for idx in range(components.shape[1])]

    components = components.replace([np.inf, -np.inf], np.nan)
    components = components.apply(
        lambda col: col.interpolate(limit_direction="both").fillna(0.0)
    )
    return components


# -----------------------------
# Model definitions
# -----------------------------
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)
        residual = x if self.downsample is None else self.downsample(x)
        out = out[:, :, -(residual.shape[2]) :]
        return self.relu(out + residual)


class TemporalConvNet(nn.Module):
    def __init__(self, num_layers: int, hidden_dim: int, kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        in_channels = 1
        for layer_idx in range(num_layers):
            out_channels = hidden_dim
            dilation = 2 ** layer_idx
            layers.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_channels = out_channels
        self.network = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features = self.network(x)
        last_step = features[:, :, -1]
        return self.head(last_step)


class AttentionPool(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.context = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features -> [batch, seq_len, hidden]
        scores = torch.tanh(self.proj(features))
        weights = torch.softmax(self.context(scores), dim=1)  # [batch, seq_len, 1]
        return (features * weights).sum(dim=1)


class BiLSTMAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.attn = AttentionPool(hidden_dim * 2)
        self.head = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        # x -> [batch, seq, 1]
        outputs, _ = self.lstm(x)
        context = self.attn(outputs)
        return self.head(context)


class ProbSparseSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, factor: int = 5, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.factor = factor
        self.embed_dim = embed_dim

    def forward(self, x):
        batch, seq_len, _ = x.shape
        select_count = max(1, int(self.factor * math.log(seq_len + 1)))
        select_count = min(select_count, seq_len)

        energy = x.norm(dim=-1)  # [batch, seq_len]
        topk_idx = energy.topk(select_count, dim=1).indices

        attn_output = torch.zeros_like(x)
        attn_output_default = x.mean(dim=1, keepdim=True)

        for b in range(batch):
            idx = topk_idx[b]
            queries = x[b : b + 1, idx]
            keys = x[b : b + 1]
            values = x[b : b + 1]
            out, _ = self.attn(queries, keys, values)
            attn_output[b, idx] = out.squeeze(0)
            mask = torch.ones(seq_len, dtype=torch.bool, device=x.device)
            mask[idx] = False
            attn_output[b, mask] = attn_output_default[b]

        return attn_output


class InformerLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, factor: int = 5, dropout: float = 0.1):
        super().__init__()
        self.attn = ProbSparseSelfAttention(embed_dim, num_heads, factor=factor, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out = self.attn(x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        return self.norm2(x + ff_out)


class InformerEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, depth: int, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, embed_dim)
        self.layers = nn.ModuleList(
            [InformerLayer(embed_dim, num_heads, dropout=dropout) for _ in range(depth)]
        )
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # x -> [batch, seq_len, 1]
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h)
        return self.head(h[:, -1, :])


# -----------------------------
# Training helpers
# -----------------------------
@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    lr: float
    device: torch.device
    model_name: str
    look_back: int
    hidden_dim: int
    verbose: bool = False
    log_interval: int = 10
    val_ratio: float = 0.1
    patience: int = 10


def make_dataloader(inputs: np.ndarray, targets: np.ndarray, mode: str, batch_size: int) -> DataLoader:
    tensor_x = torch.from_numpy(inputs)
    if mode == "channels_first":
        tensor_x = tensor_x.squeeze(-1) if tensor_x.ndim == 3 else tensor_x
        tensor_x = tensor_x.unsqueeze(1)
    elif mode == "seq_last_dim":
        tensor_x = tensor_x.unsqueeze(-1) if tensor_x.ndim == 2 else tensor_x
    else:
        raise ValueError("Chế độ đầu vào không hợp lệ.")

    tensor_y = torch.from_numpy(targets).unsqueeze(-1)
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=len(dataset) > batch_size)


def format_eval_tensor(inputs: np.ndarray, mode: str) -> torch.Tensor:
    tensor = torch.from_numpy(inputs)
    if mode == "channels_first":
        tensor = tensor.squeeze(-1) if tensor.ndim == 3 else tensor
        tensor = tensor.unsqueeze(1)
    elif mode == "seq_last_dim":
        tensor = tensor.unsqueeze(-1) if tensor.ndim == 2 else tensor
    else:
        raise ValueError("Chế độ đầu vào không hợp lệ.")
    return tensor


def get_model(model_name: str, hidden_dim: int) -> Tuple[nn.Module, str]:
    if model_name == "tcn":
        return TemporalConvNet(num_layers=3, hidden_dim=hidden_dim), "channels_first"
    if model_name == "bilstm_attn":
        return BiLSTMAttention(hidden_dim=hidden_dim), "seq_last_dim"
    if model_name == "informer":
        return InformerEncoder(embed_dim=hidden_dim, num_heads=4, depth=3), "seq_last_dim"
    raise ValueError(f"Model {model_name} chưa được hỗ trợ.")


def train_component_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    config: TrainConfig,
    component_idx: int,
) -> np.ndarray:
    model, mode = get_model(config.model_name, config.hidden_dim)
    model.to(config.device)

    val_size = int(len(train_x) * config.val_ratio)
    if config.val_ratio > 0 and val_size < 1 and len(train_x) > 1:
        val_size = 1
    if val_size >= len(train_x):
        val_size = max(1, len(train_x) // 5)

    if val_size > 0:
        val_x = train_x[-val_size:]
        val_y = train_y[-val_size:]
        train_x = train_x[:-val_size]
        train_y = train_y[:-val_size]
        val_tensor_x = format_eval_tensor(val_x, mode=mode).to(config.device)
        val_tensor_y = torch.from_numpy(val_y).unsqueeze(-1).to(config.device)
    else:
        val_tensor_x = None
        val_tensor_y = None

    loader = make_dataloader(train_x, train_y, mode=mode, batch_size=config.batch_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    model.train()
    for epoch in range(config.epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(config.device)
            batch_y = batch_y.to(config.device)
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_loss = None
        if val_tensor_x is not None:
            model.eval()
            with torch.no_grad():
                val_preds = model(val_tensor_x)
                val_loss = criterion(val_preds, val_tensor_y).item()
            model.train()
            if val_loss + 1e-6 < best_val:
                best_val = val_loss
                best_state = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
        if config.verbose and (epoch + 1) % config.log_interval == 0:
            msg = (
                f"[Comp {component_idx}] Epoch {epoch+1}/{config.epochs} "
                f"TrainLoss: {loss.item():.6f}"
            )
            if val_loss is not None:
                msg += f" | ValLoss: {val_loss:.6f}"
            print(msg)
        if val_tensor_x is not None and epochs_no_improve >= config.patience:
            if config.verbose:
                print(
                    f"[Comp {component_idx}] Early stop tại epoch {epoch+1}; "
                    f"best val loss {best_val:.6f}"
                )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    eval_tensor = format_eval_tensor(test_x, mode=mode).to(config.device)
    with torch.no_grad():
        preds = model(eval_tensor).cpu().numpy().reshape(-1)
    return preds


# -----------------------------
# Main pipeline
# -----------------------------
def run_pipeline(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    series = load_series(
        data_path=args.data_path,
        target_col=args.target_col,
        timestamp_col=args.timestamp_col,
        month_col=args.month_col,
        months=args.months,
    )

    components = ceemdan_ewt_decompose(
        series=series,
        num_ewt_components=args.num_ewt_components,
        epsilon=args.epsilon,
        seed=args.seed,
    )

    series_array = series.values.astype(np.float32).reshape(-1, 1)
    full_train_size = int(len(series_array) * args.train_ratio)
    if full_train_size <= args.look_back:
        raise ValueError("look_back quá lớn so với dữ liệu (chuỗi gốc).")
    full_test = series_array[full_train_size:]
    _, series_test_y = create_dataset(full_test, args.look_back)
    if len(series_test_y) == 0:
        raise ValueError("Tập test của chuỗi gốc rỗng. Điều chỉnh train_ratio hoặc look_back.")
    series_target = series_test_y.reshape(-1)

    predictions = []

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        model_name=args.model,
        look_back=args.look_back,
        hidden_dim=args.hidden_dim,
        verbose=not args.quiet,
        log_interval=args.log_interval,
        val_ratio=args.val_ratio,
        patience=args.patience,
    )

    for idx, column in enumerate(components.columns):
        arr = components[column].values.reshape(-1, 1)
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()

        train_size = int(len(arr) * args.train_ratio)
        if train_size <= args.look_back:
            raise ValueError("look_back quá lớn so với dữ liệu huấn luyện.")

        train, test = arr[:train_size], arr[train_size:]
        train_x, train_y = create_dataset(train, args.look_back)
        test_x, test_y = create_dataset(test, args.look_back)

        if len(test_y) == 0:
            raise ValueError("Kiểm tra lại train_ratio/look_back vì tập test rỗng.")

        train_x = scaler_x.fit_transform(train_x.reshape(train_x.shape[0], -1))
        train_y = scaler_y.fit_transform(train_y.reshape(-1, 1)).reshape(-1)
        test_x = scaler_x.transform(test_x.reshape(test_x.shape[0], -1))

        preds_scaled = train_component_model(train_x, train_y, test_x, config, idx + 1)
        preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).reshape(-1)

        min_len = min(len(preds), len(test_y))
        predictions.append(preds[:min_len])

    common_len = min(len(arr) for arr in predictions)
    predictions = [arr[:common_len] for arr in predictions]

    aggregated_pred = np.sum(np.vstack(predictions), axis=0)

    final_len = min(len(series_target), len(aggregated_pred))
    if final_len == 0:
        raise ValueError("Không có mẫu trùng khớp giữa dự báo và chuỗi gốc.")
    aggregated_pred = aggregated_pred[:final_len]
    target = series_target[:final_len]

    cap_value = args.cap if args.cap else np.max(np.abs(target))
    epsilon = 1e-6
    mape = np.mean(np.abs((target - aggregated_pred) / (cap_value + epsilon))) * 100
    rmse = math.sqrt(mean_squared_error(target, aggregated_pred))
    mae = mean_absolute_error(target, aggregated_pred)

    result_df = pd.DataFrame(
        {
            "y_true": target,
            "y_pred": aggregated_pred,
        }
    )
    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        result_df.to_csv(args.output_path, index=False)

    print(f"Thiết bị: {device}")
    print(f"Kết quả ({args.model}):")
    print(f"  MAPE: {mape:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    if args.output_path:
        print(f"Đã lưu dự báo tại: {args.output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CEEMDAN + EWT + (TCN / Informer-lite / BiLSTM Attention) bằng PyTorch."
    )
    parser.add_argument("--data_path", required=True, help="Đường dẫn tới file CSV.")
    parser.add_argument("--target_col", required=True, help="Tên cột mục tiêu.")
    parser.add_argument("--timestamp_col", default=None, help="Cột timestamp (nếu có).")
    parser.add_argument("--month_col", default=None, help="Cột tháng (vd: Month).")
    parser.add_argument(
        "--months",
        nargs="*",
        default=None,
        help="Danh sách tháng cần huấn luyện (vd: --months 1 2 3).",
    )
    parser.add_argument(
        "--model",
        default="tcn",
        choices=["tcn", "informer", "bilstm_attn"],
        help="Chọn mô hình huấn luyện.",
    )
    parser.add_argument("--look_back", type=int, default=48, help="Độ dài chuỗi đầu vào.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Tỉ lệ train/test.")
    parser.add_argument("--epochs", type=int, default=50, help="Số epoch huấn luyện.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Kích thước ẩn.")
    parser.add_argument("--num_ewt_components", type=int, default=3, help="Số thành phần EWT.")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Thông số CEEMDAN epsilon.")
    parser.add_argument("--cap", type=float, default=None, help="Giá trị công suất định mức để tính MAPE.")
    parser.add_argument("--output_path", default=None, help="File lưu kết quả dự báo.")
    parser.add_argument("--seed", type=int, default=1234, help="Seed tái lập.")
    parser.add_argument("--quiet", action="store_true", help="Tắt in log huấn luyện.")
    parser.add_argument("--log_interval", type=int, default=10, help="Chu kỳ in loss.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Tỉ lệ dữ liệu train dùng cho validation.")
    parser.add_argument("--patience", type=int, default=10, help="Số epoch không cải thiện trước khi dừng sớm.")
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())

