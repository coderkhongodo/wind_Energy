#!/usr/bin/env python
# coding: utf-8
"""
CEEMDAN + EWT ensemble learning script.

Stage 1:  PyTorch LSTM predicts the wind power signal reconstructed from CEEMDAN + EWT components.
Stage 2:  LightGBM models the LSTM residuals using richer handcrafted features (IMF summaries,
          optional weather variables, trending metrics, lagged residuals).

Final prediction = LSTM prediction + residual prediction.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PyEMD import CEEMDAN
import ewtpy
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int = 1234) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ceemdan_ewt_decomposition(
    series: np.ndarray,
    epsilon: float = 0.05,
    noise_seed: int = 12345,
    ewt_components: int = 3,
) -> np.ndarray:
    """
    Decompose a 1-D series using CEEMDAN, then denoise the highest-frequency IMF with EWT.
    Returns an array of shape (len(series), n_imf).
    """
    ceemdan = CEEMDAN(epsilon=epsilon)
    ceemdan.noise_seed(noise_seed)
    imfs = ceemdan(series)
    if imfs.size == 0:
        raise RuntimeError("CEEMDAN produced zero IMFs. Check the input series.")
    imf_matrix = np.array(imfs).T

    first_imf = imf_matrix[:, 0]
    ewt, _, _ = ewtpy.EWT1D(first_imf, N=ewt_components)
    ewt_df = pd.DataFrame(ewt)
    if ewt_df.shape[1] > 2:
        # Drop the noisiest component to keep denoising consistent.
        ewt_df = ewt_df.iloc[:, :2]
    denoised = ewt_df.sum(axis=1).values
    imf_matrix[:, 0] = denoised
    return imf_matrix


def build_lstm_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    look_back: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create rolling windows for LSTM training.
    Returns X (samples, look_back, n_features), y (samples,) and the raw indices used.
    """
    seqs, ys, indices = [], [], []
    for idx in range(look_back, len(targets)):
        seq = features[idx - look_back : idx]
        seqs.append(seq)
        ys.append(targets[idx])
        indices.append(idx)
    return np.array(seqs), np.array(ys), np.array(indices)


def build_base_feature_frame(
    imf_matrix: np.ndarray,
    targets: np.ndarray,
    weather: Optional[pd.DataFrame],
    indices: np.ndarray,
    look_back: int,
) -> pd.DataFrame:
    """
    Build IMF/weather/trend features for each prediction index.
    """
    rows = []
    n_imf = imf_matrix.shape[1]
    for idx in indices:
        window_imf = imf_matrix[idx - look_back : idx]
        window_target = targets[idx - look_back : idx]
        entry = {}
        for k in range(n_imf):
            entry[f"imf_last_{k}"] = float(window_imf[-1, k])
            entry[f"imf_mean_{k}"] = float(window_imf[:, k].mean())
        entry["trend_mean"] = float(window_target.mean())
        entry["trend_std"] = float(window_target.std(ddof=0))
        entry["trend_delta"] = float(window_target[-1] - window_target[0])
        if weather is not None and not weather.empty:
            weather_row = weather.iloc[idx - 1]
            for col in weather.columns:
                entry[f"weather_{col}"] = float(weather_row[col])
        rows.append(entry)
    return pd.DataFrame(rows)


def add_residual_lags(
    frame: pd.DataFrame, residuals: Sequence[float], lag_steps: int
) -> pd.DataFrame:
    res_series = pd.Series(residuals, index=frame.index)
    for lag in range(1, lag_steps + 1):
        frame[f"residual_lag_{lag}"] = res_series.shift(lag)
    return frame.fillna(0.0)


class RecurrentRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        cell_type: str = "lstm",
        bidirectional: bool = False,
    ):
        super().__init__()
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}.get(cell_type.lower(), nn.LSTM)
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(x)
        last = output[:, -1, :]
        return self.fc(last)


def train_recurrent_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    epochs: int,
    device: torch.device,
    lr: float = 1e-3,
    patience: int = 10,
) -> None:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float("inf")
    best_state = None
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x).squeeze()
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
        avg_loss = running_loss / len(train_loader.dataset)
        val_loss = None
        if val_loader is not None:
            model.eval()
            val_running = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    preds = model(batch_x).squeeze()
                    val_running += criterion(preds, batch_y).item() * batch_x.size(0)
            val_loss = val_running / len(val_loader.dataset)
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = model.state_dict()
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    print(f"Early stopping at epoch {epoch} (best val_loss={best_val:.6f}).")
                    if best_state is not None:
                        model.load_state_dict(best_state)
                    break
        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            if val_loss is not None:
                print(f"[Epoch {epoch:03d}] train_loss={avg_loss:.6f} val_loss={val_loss:.6f}")
            else:
                print(f"[Epoch {epoch:03d}] train_loss={avg_loss:.6f}")


def predict_recurrent_model(
    model: nn.Module,
    data: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(torch.from_numpy(data).float(), batch_size=batch_size)
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch).squeeze().detach().cpu().numpy()
            preds.append(out)
    return np.concatenate(preds)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, cap: float) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / cap)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CEEMDAN + EWT Ensemble Learning (LSTM + LightGBM).")
    parser.add_argument("--data-path", type=str, required=True, help="Path to CSV dataset.")
    parser.add_argument("--target-column", type=str, default="P_avg", help="Name of the power column.")
    parser.add_argument("--datetime-column", type=str, default=None, help="Optional datetime column for sorting.")
    parser.add_argument("--datetime-format", type=str, default=None, help="Optional strptime format for datetime parsing.")
    parser.add_argument("--dayfirst", action="store_true", help="Interpret day as first when parsing datetime.")
    parser.add_argument("--month-column", type=str, default="Month", help="Column storing numeric month (auto-created from datetime if missing).")
    parser.add_argument("--months", nargs="+", type=int, help="Filter the dataset to these months for a single run.")
    parser.add_argument("--month-folds", nargs="+", help="Multiple month groups for CV-style runs. Example: \"1\" \"2\" \"3\" or \"1,2\".")
    parser.add_argument("--weather-columns", nargs="*", default=[], help="Optional weather/exogenous columns.")
    parser.add_argument("--look-back", type=int, default=24, help="Look-back window for sequences.")
    parser.add_argument("--sequence-mode", choices=["shared", "per-imf"], default="per-imf", help="Use a single multivariate model or dedicated IMF models.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train/test split ratio.")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs for LSTM.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for LSTM.")
    parser.add_argument("--hidden-size", type=int, default=128, help="Hidden size of LSTM.")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of LSTM layers.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout for stacked LSTM layers.")
    parser.add_argument("--patience", type=int, default=12, help="Early stopping patience (epochs).")
    parser.add_argument("--shared-cell", choices=["lstm", "gru"], default="lstm", help="Cell type for the shared multivariate model.")
    parser.add_argument("--shared-bidirectional", action="store_true", help="Use bidirectional layers for the shared model.")
    parser.add_argument("--imf-cell", choices=["lstm", "gru"], default="gru", help="Cell type for individual IMF models.")
    parser.add_argument("--imf-bidirectional", action="store_true", help="Use bidirectional layers for IMF models.")
    parser.add_argument("--imf-hidden-size", type=int, default=128, help="Hidden size for IMF models.")
    parser.add_argument("--imf-num-layers", type=int, default=1, help="Number of layers for IMF models.")
    parser.add_argument("--imf-dropout", type=float, default=0.1, help="Dropout for IMF models.")
    parser.add_argument("--imf-epochs", type=int, default=80, help="Epochs for IMF models.")
    parser.add_argument("--imf-batch-size", type=int, default=128, help="Batch size for IMF models.")
    parser.add_argument("--imf-lr", type=float, default=1e-3, help="Learning rate for IMF models.")
    parser.add_argument("--imf-patience", type=int, default=12, help="Early stopping patience for IMF models.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for LSTM.")
    parser.add_argument("--ewt-components", type=int, default=3, help="Number of EWT components.")
    parser.add_argument("--epsilon", type=float, default=0.05, help="CEEMDAN epsilon parameter.")
    parser.add_argument("--capacity", type=float, default=None, help="Wind farm capacity for MAPE scaling.")
    parser.add_argument("--residual-lags", type=int, default=2, help="Number of residual lags for LightGBM features.")
    parser.add_argument("--lgbm-estimators", type=int, default=500, help="Number of LightGBM trees.")
    parser.add_argument("--lgbm-learning-rate", type=float, default=0.05, help="LightGBM learning rate.")
    parser.add_argument("--lgbm-max-depth", type=int, default=-1, help="LightGBM max depth.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    return parser.parse_args()


def prepare_dataframe(args: argparse.Namespace) -> Tuple[pd.DataFrame, Optional[str]]:
    df = pd.read_csv(args.data_path)
    if args.datetime_column and args.datetime_column in df.columns:
        df[args.datetime_column] = pd.to_datetime(
            df[args.datetime_column],
            format=args.datetime_format,
            dayfirst=args.dayfirst,
            errors="raise",
        )
        df = df.sort_values(args.datetime_column).reset_index(drop=True)

    if args.target_column not in df.columns:
        raise ValueError(f"Target column '{args.target_column}' not found in dataset.")

    month_col = args.month_column
    if month_col and month_col in df.columns:
        return df, month_col
    if args.datetime_column is None or args.datetime_column not in df.columns:
        return df, None
    month_col = month_col or "Month"
    df[month_col] = df[args.datetime_column].dt.month
    return df, month_col


def parse_month_groups(raw_groups: Optional[Sequence[str]]) -> List[List[int]]:
    groups: List[List[int]] = []
    if not raw_groups:
        return groups
    for group in raw_groups:
        tokens = group.replace(",", " ").split()
        months = [int(tok) for tok in tokens]
        if months:
            groups.append(months)
    return groups


def filter_by_months(df: pd.DataFrame, month_col: Optional[str], months: Sequence[int]) -> pd.DataFrame:
    if month_col is None:
        raise ValueError("Month filtering requested but no month column is available.")
    subset = df[df[month_col].isin(months)].copy()
    return subset


def train_shared_sequence_model(
    seq_features: np.ndarray,
    seq_targets: np.ndarray,
    split_idx: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    look_back = seq_features.shape[1]
    num_features = seq_features.shape[2]

    X_train = seq_features[:split_idx]
    X_test = seq_features[split_idx:]
    y_train = seq_targets[:split_idx]
    y_test = seq_targets[split_idx:]

    sc_X = StandardScaler()
    sc_y = StandardScaler()

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    X_train_scaled = sc_X.fit_transform(X_train_flat)
    X_test_scaled = sc_X.transform(X_test_flat)

    y_train_scaled = sc_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = sc_y.transform(y_test.reshape(-1, 1)).ravel()

    train_inputs = X_train_scaled.reshape(X_train.shape[0], look_back, num_features)
    test_inputs = X_test_scaled.reshape(X_test.shape[0], look_back, num_features)

    train_dataset = TensorDataset(
        torch.from_numpy(train_inputs).float(),
        torch.from_numpy(y_train_scaled).float(),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(test_inputs).float(),
        torch.from_numpy(y_test_scaled).float(),
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    model = RecurrentRegressor(
        input_size=num_features,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        cell_type=args.shared_cell,
        bidirectional=args.shared_bidirectional,
    ).to(device)

    train_recurrent_model(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        device=device,
        lr=args.lr,
        patience=args.patience,
    )

    train_preds = predict_recurrent_model(model, train_inputs, device)
    test_preds = predict_recurrent_model(model, test_inputs, device)

    train_preds = sc_y.inverse_transform(train_preds.reshape(-1, 1)).ravel()
    test_preds = sc_y.inverse_transform(test_preds.reshape(-1, 1)).ravel()

    y_train_actual = sc_y.inverse_transform(y_train_scaled.reshape(-1, 1)).ravel()
    y_test_actual = sc_y.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()

    return train_preds, test_preds, y_train_actual, y_test_actual


def train_per_imf_models(
    imf_matrix: np.ndarray,
    target_series: np.ndarray,
    look_back: int,
    raw_split_idx: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Train separate models for each IMF, splitting raw data FIRST (like notebook).
    raw_split_idx: index to split raw IMF data (before creating sequences).
    """
    agg_train = None
    agg_test = None

    for col in range(imf_matrix.shape[1]):
        series = imf_matrix[:, col]
        # Split raw data FIRST (like notebook)
        train_raw = series[:raw_split_idx]
        test_raw = series[raw_split_idx:]
        
        # Create sequences separately for train and test
        seq_features_train = train_raw.reshape(-1, 1)
        seq_features_test = test_raw.reshape(-1, 1)
        X_train, y_train, _ = build_lstm_sequences(seq_features_train, train_raw, look_back)
        X_test, y_test, _ = build_lstm_sequences(seq_features_test, test_raw, look_back)
        
        if len(X_train) == 0 or len(X_test) == 0:
            continue

        sc_X = StandardScaler()
        sc_y = StandardScaler()

        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)

        X_train_scaled = sc_X.fit_transform(X_train_flat)
        X_test_scaled = sc_X.transform(X_test_flat)

        y_train_scaled = sc_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test_scaled = sc_y.transform(y_test.reshape(-1, 1)).ravel()

        train_inputs = X_train_scaled.reshape(X_train.shape[0], look_back, 1)
        test_inputs = X_test_scaled.reshape(X_test.shape[0], look_back, 1)

        train_dataset = TensorDataset(
            torch.from_numpy(train_inputs).float(),
            torch.from_numpy(y_train_scaled).float(),
        )
        val_dataset = TensorDataset(
            torch.from_numpy(test_inputs).float(),
            torch.from_numpy(y_test_scaled).float(),
        )

        train_loader = DataLoader(train_dataset, batch_size=args.imf_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.imf_batch_size)

        model = RecurrentRegressor(
            input_size=1,
            hidden_size=args.imf_hidden_size,
            num_layers=args.imf_num_layers,
            dropout=args.imf_dropout,
            cell_type=args.imf_cell,
            bidirectional=args.imf_bidirectional,
        ).to(device)

        train_recurrent_model(
            model,
            train_loader,
            val_loader,
            epochs=args.imf_epochs,
            device=device,
            lr=args.imf_lr,
            patience=args.imf_patience,
        )

        train_pred = predict_recurrent_model(model, train_inputs, device)
        test_pred = predict_recurrent_model(model, test_inputs, device)

        train_pred = sc_y.inverse_transform(train_pred.reshape(-1, 1)).ravel()
        test_pred = sc_y.inverse_transform(test_pred.reshape(-1, 1)).ravel()

        if agg_train is None:
            agg_train = np.zeros_like(train_pred)
            agg_test = np.zeros_like(test_pred)
            # Get actual targets from original target_series (not IMF)
            train_target_raw = target_series[:raw_split_idx]
            test_target_raw = target_series[raw_split_idx:]
            _, y_train_actual, _ = build_lstm_sequences(train_target_raw.reshape(-1, 1), train_target_raw, look_back)
            _, y_test_actual, _ = build_lstm_sequences(test_target_raw.reshape(-1, 1), test_target_raw, look_back)

        agg_train += train_pred
        agg_test += test_pred

    if agg_train is None or agg_test is None:
        raise RuntimeError("No IMF predictions were generated.")

    return agg_train, agg_test, y_train_actual, y_test_actual


def run_experiment(
    df: pd.DataFrame,
    args: argparse.Namespace,
    run_label: str,
) -> None:
    if len(df) <= args.look_back + 1:
        print(f"Skipping run '{run_label}' because the subset is too short (len={len(df)}).")
        return

    print(f"\n=== Run: {run_label} | samples={len(df)} ===")
    target_series = df[args.target_column].astype(float).values
    cap = args.capacity or float(np.max(target_series))

    weather_df = df[args.weather_columns].astype(float) if args.weather_columns else None

    print("Running CEEMDAN + EWT decomposition ...")
    imf_matrix = ceemdan_ewt_decomposition(
        target_series,
        epsilon=args.epsilon,
        noise_seed=args.seed,
        ewt_components=args.ewt_components,
    )

    # Split raw data FIRST (like notebook), then create sequences separately
    raw_split_idx = int(len(target_series) * args.train_ratio)
    if raw_split_idx == 0 or raw_split_idx >= len(target_series):
        print("Train/test split invalid.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training base sequence models on device: {device}")

    if args.sequence_mode == "shared":
        scaler = StandardScaler()
        scaled_imfs = scaler.fit_transform(imf_matrix)
        seq_features, seq_targets, indices = build_lstm_sequences(scaled_imfs, target_series, args.look_back)
        split_idx = int(len(seq_targets) * args.train_ratio)
        base_train_pred, base_test_pred, y_train_actual, y_test_actual = train_shared_sequence_model(
            seq_features, seq_targets, split_idx, args, device
        )
    else:
        base_train_pred, base_test_pred, y_train_actual, y_test_actual = train_per_imf_models(
            imf_matrix, target_series, args.look_back, raw_split_idx, args, device
        )

    base_metrics = compute_metrics(y_test_actual, base_test_pred, cap)
    print(
        f"Sequence model metrics ({args.sequence_mode}): "
        f"MAPE={base_metrics['MAPE']:.3f}% RMSE={base_metrics['RMSE']:.5f} MAE={base_metrics['MAE']:.5f}"
    )

    residual_train = y_train_actual - base_train_pred
    residual_test_true = y_test_actual - base_test_pred

    print("Preparing LightGBM residual features ...")
    # Build features using absolute indices in full data
    train_indices = np.arange(args.look_back, raw_split_idx)
    test_indices = np.arange(raw_split_idx + args.look_back, len(target_series))
    base_train = build_base_feature_frame(imf_matrix, target_series, weather_df, train_indices, args.look_back)
    base_test = build_base_feature_frame(imf_matrix, target_series, weather_df, test_indices, args.look_back)

    lag_steps = max(1, args.residual_lags)
    train_features = add_residual_lags(base_train, residual_train, lag_steps)
    feature_columns = train_features.columns.tolist()

    lgbm = LGBMRegressor(
        n_estimators=args.lgbm_estimators,
        learning_rate=args.lgbm_learning_rate,
        max_depth=args.lgbm_max_depth,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="regression",
        random_state=args.seed,
    )

    lgbm.fit(train_features.values, residual_train)

    residual_history = list(residual_train.astype(float))
    residual_preds = []
    for _, base_row in base_test.iterrows():
        row_values = base_row.tolist()
        for lag in range(1, lag_steps + 1):
            value = residual_history[-lag] if len(residual_history) >= lag else 0.0
            row_values.append(float(value))
        row_frame = pd.DataFrame([row_values], columns=feature_columns)
        pred_residual = float(lgbm.predict(row_frame)[0])
        residual_history.append(pred_residual)
        residual_preds.append(pred_residual)

    residual_preds = np.array(residual_preds)
    final_predictions = base_test_pred + residual_preds

    final_metrics = compute_metrics(y_test_actual, final_predictions, cap)
    print(
        f"Ensemble metrics: MAPE={final_metrics['MAPE']:.3f}% "
        f"RMSE={final_metrics['RMSE']:.5f} MAE={final_metrics['MAE']:.5f}"
    )
    if residual_test_true.size:
        res_mae = mean_absolute_error(residual_test_true, residual_preds)
        res_rmse = np.sqrt(mean_squared_error(residual_test_true, residual_preds))
        print(f"Residual model diagnostics -> RMSE: {res_rmse:.5f}, MAE: {res_mae:.5f}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    df, month_col = prepare_dataframe(args)

    month_groups = parse_month_groups(args.month_folds)
    if month_groups:
        for idx, months in enumerate(month_groups, 1):
            subset = filter_by_months(df, month_col, months)
            if subset.empty:
                print(f"Skipping month fold {idx} ({months}) because subset is empty.")
                continue
            label = f"fold{idx}_months_{'_'.join(map(str, months))}"
            run_experiment(subset, args, label)
    else:
        if args.months:
            subset = filter_by_months(df, month_col, args.months)
            if subset.empty:
                print(f"No samples after filtering months {args.months}.")
                return
            label = f"months_{'_'.join(map(str, args.months))}"
            run_experiment(subset, args, label)
        else:
            run_experiment(df, args, "all")


if __name__ == "__main__":
    main()

