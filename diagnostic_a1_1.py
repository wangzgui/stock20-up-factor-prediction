"""
V3.2-A1.1 offline strategy diagnostics (no submission).

The online-best sorted-row V3 reconstruction remains the probability base:
the unchanged 59-feature up model plus the unchanged absolute-q80 tail overlay.
The A1.1 candidate predicts whether a stock's future 20-day return is in the
top 20% of its own anchor-day market cross-section.  It uses the original 59
features plus the 12 leakage-safe A1 return-relative features and is added only
as a small, fixed, centered-rank overlay after passing coherent-day gates.

Kaggle inputs:
  /kaggle/input/datasets/zhuowamg/test-data/train.csv
  /kaggle/input/datasets/zhuowamg/true-test-data/test.csv

Reports:
  prediction-decile monotonicity, Top-K performance, Top5 replacements,
  and performance by future market regime.
"""

from __future__ import annotations

import gc
import time
import warnings
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TRAIN_PATH = Path("/kaggle/input/datasets/zhuowamg/test-data/train.csv")
TEST_PATH = Path("/kaggle/input/datasets/zhuowamg/true-test-data/test.csv")
OUTPUT_PATH = Path("/kaggle/working/submission_v3_2_a1_1.csv")
REFERENCE_BASELINE_PATH = Path("/kaggle/working/submission_v3_2_a1.csv")
DIAGNOSTIC_ONLY = True
DIAGNOSTIC_TOP_K = (5, 10, 20, 50, 100)

WINDOW = 20
HORIZON = 20
TRAIN_STRIDE = 10
RANDOM_STATE = 20260822
PROBABILITY_CLIP = (0.02, 0.98)

PRICE_COLUMNS = ["open", "high", "low", "close"]
RAW_COLUMNS = ["code", "date", *PRICE_COLUMNS, "volume"]
FLOAT_DTYPES = {c: "float32" for c in [*PRICE_COLUMNS, "volume"]}

HOLDOUT_CAL_BUCKET_START = 60
HOLDOUT_STRATEGY_BUCKET_START = 78
MODEL_TRAIN_END_DAY = 2200
CALIBRATION_START_DAY = 2241
CALIBRATION_END_DAY = 2400
STRATEGY_START_DAY = 2441

TAIL_QUANTILE = 0.80
NEUTRAL_PROBABILITY_CENTER = 0.50
PSEUDO_BATCH_REPEATS = 30
V32_GAMMA = 0.005

# A1.1 is deliberately a third overlay; it never replaces the absolute q80
# tail model which produced the current 0.03403 online baseline.
RELATIVE_TAIL_QUANTILE = 0.80
A11_GAMMA = 0.0025
A11_MIN_SAME_DAY_AUC = 0.530
A11_MIN_MEAN_SCORE_GAIN = 0.001
A11_MIN_PAIRED_WIN_RATE = 0.55
A11_MAX_Q20_SCORE_DROP = 0.005
A11_MAX_BRIER_CHANGE = 0.0005
A11_MAX_INDEPENDENT_MEAN_DROP = 0.003
A11_TARGET_COLUMN = "a11_relative_tail_y"
A1_HORIZONS = (1, 5, 10, 19)
A1_FEATURE_COLUMNS = tuple(
    name
    for h in A1_HORIZONS
    for name in (
        f"a1_ret{h}_rank_pct",
        f"a1_ret{h}_excess_median",
        f"a1_ret{h}_zscore",
    )
)


def timer(message: str):
    class _Timer:
        def __enter__(self):
            self.start = time.time()
            print(f"\n[{message}]", flush=True)
            return self

        def __exit__(self, exc_type, exc, tb):
            print(f"Finished in {time.time() - self.start:.1f}s", flush=True)
    return _Timer()


def read_market_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(
        path,
        usecols=RAW_COLUMNS,
        dtype={"code": "category", "date": "string", **FLOAT_DTYPES},
    )
    missing = sorted(set(RAW_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path.name}: {missing}")
    df = df[RAW_COLUMNS]
    day_text = df["date"].str.extract(r"DAY_(\d+)", expand=False)
    if day_text.isna().any():
        raise ValueError(f"Invalid date values in {path.name}")
    df["day"] = day_text.astype("int16")
    df.drop(columns="date", inplace=True)
    numeric = [*PRICE_COLUMNS, "volume"]
    if df[numeric].isna().any().any():
        raise ValueError(f"NaN found in {path.name}")
    if (df[PRICE_COLUMNS] <= 0).any().any() or (df["volume"] < 0).any():
        raise ValueError(f"Invalid non-positive price or negative volume in {path.name}")
    return df


def add_a1_cross_sectional_returns(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Compute A1 features on the full raw panel before window subsampling.

    A shift is valid only when the earlier row is exactly h calendar DAYs ago;
    this prevents a suspension/listing gap from masquerading as an h-day return.
    Cross-sectional statistics use only contemporaneously observable past data.
    """
    df = df.sort_values(["code", "day"], kind="stable").reset_index(drop=True)
    by_code = df.groupby("code", observed=True, sort=False)
    close = df["close"].to_numpy(np.float64, copy=False)
    day = df["day"].to_numpy(np.int32, copy=False)

    print(f"  {label}: calculating {len(A1_FEATURE_COLUMNS)} A1 panel features", flush=True)
    for h in A1_HORIZONS:
        previous_close = by_code["close"].shift(h).to_numpy(np.float64)
        previous_day = by_code["day"].shift(h).to_numpy(np.float64)
        valid = np.isfinite(previous_close) & ((day - previous_day) == h)

        ret = np.full(len(df), np.nan, dtype=np.float64)
        ret[valid] = close[valid] / previous_close[valid] - 1.0
        ret_s = pd.Series(ret, index=df.index)
        by_day = ret_s.groupby(df["day"], sort=False)

        rank = by_day.rank(method="average", pct=True)
        median = by_day.transform("median")
        mean = by_day.transform("mean")
        std = by_day.transform("std")

        df[f"a1_ret{h}_rank_pct"] = rank.fillna(0.5).astype("float32")
        df[f"a1_ret{h}_excess_median"] = (
            ret_s - median
        ).fillna(0.0).clip(-2.0, 2.0).astype("float32")
        df[f"a1_ret{h}_zscore"] = (
            (ret_s - mean) / std.where(std > 1e-8)
        ).fillna(0.0).clip(-8.0, 8.0).astype("float32")

        del previous_close, previous_day, ret, ret_s, by_day, rank, median, mean, std
        gc.collect()
        print(f"    horizon={h:2d}: valid rows={valid.sum():,}", flush=True)

    a1 = df[list(A1_FEATURE_COLUMNS)].to_numpy(np.float32, copy=False)
    if not np.isfinite(a1).all():
        raise ValueError(f"Non-finite A1 features remain in {label}")
    return df


def add_a11_relative_future_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create a training-only top-20%-within-anchor-day future label.

    The label at anchor t uses close[t+20] / close[t] - 1 and ranks that
    future return only against other stocks with a valid label at the same t.
    It is never called for test data.  Value 255 marks rows without a valid
    continuous 20-day future and must never survive window construction.
    """
    by_code = df.groupby("code", observed=True, sort=False)
    close = df["close"].to_numpy(np.float64, copy=False)
    day = df["day"].to_numpy(np.int32, copy=False)
    future_close = by_code["close"].shift(-HORIZON).to_numpy(np.float64)
    future_day = by_code["day"].shift(-HORIZON).to_numpy(np.float64)
    valid = np.isfinite(future_close) & ((future_day - day) == HORIZON)

    future_return = np.full(len(df), np.nan, dtype=np.float64)
    future_return[valid] = future_close[valid] / close[valid] - 1.0
    future_s = pd.Series(future_return, index=df.index)
    future_rank = future_s.groupby(df["day"], sort=False).rank(
        method="average", pct=True
    )
    relative_y = np.full(len(df), 255, dtype=np.uint8)
    relative_y[valid] = (
        future_rank.to_numpy(np.float64, na_value=np.nan)[valid]
        > RELATIVE_TAIL_QUANTILE
    ).astype(np.uint8)
    df[A11_TARGET_COLUMN] = relative_y

    valid_rate = float(relative_y[valid].mean())
    print(
        f"  relative target: valid rows={valid.sum():,}, "
        f"positive rate={valid_rate:.4f}",
        flush=True,
    )
    if not (0.18 <= valid_rate <= 0.22):
        raise ValueError(f"Unexpected relative-tail positive rate: {valid_rate:.4f}")
    return df


def _safe_std(x: np.ndarray, axis: int = 1) -> np.ndarray:
    return np.std(x, axis=axis, dtype=np.float64).astype(np.float32)


def make_window_features(values: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """The unchanged 59 scale-free V3 feature representation."""
    offsets = np.arange(-(WINDOW - 1), 1, dtype=np.int32)
    idx = anchors[:, None] + offsets[None, :]
    w = values[idx].astype(np.float64, copy=False)
    op, hi, lo, cl, vol = (w[:, :, i] for i in range(5))
    eps = 1e-12

    log_close = np.log(np.maximum(cl, eps))
    ret = np.diff(log_close, axis=1)
    log_vol = np.log(np.maximum(vol, eps))
    vol_change = np.diff(log_vol, axis=1)
    candle_range = (hi - lo) / np.maximum(cl, eps)
    candle_body = (cl - op) / np.maximum(op, eps)
    close_location = (cl - lo) / np.maximum(hi - lo, eps)
    gap = op[:, 1:] / np.maximum(cl[:, :-1], eps) - 1.0
    true_range = np.maximum.reduce([
        hi[:, 1:] - lo[:, 1:],
        np.abs(hi[:, 1:] - cl[:, :-1]),
        np.abs(lo[:, 1:] - cl[:, :-1]),
    ]) / np.maximum(cl[:, :-1], eps)

    blocks = [ret.astype(np.float32)]
    for h in (1, 2, 3, 5, 10, 19):
        blocks.append((cl[:, -1] / np.maximum(cl[:, -1 - h], eps) - 1.0)[:, None])
    for h in (5, 10, 19):
        rr = ret[:, -h:]
        blocks.extend([
            rr.mean(axis=1)[:, None],
            _safe_std(rr)[:, None],
            np.sqrt(np.mean(np.minimum(rr, 0.0) ** 2, axis=1))[:, None],
        ])

    centered = ret - ret.mean(axis=1, keepdims=True)
    ret_std = np.maximum(_safe_std(ret), eps)
    blocks.append((np.mean(centered ** 3, axis=1) / (ret_std ** 3))[:, None])
    x_axis = np.arange(WINDOW, dtype=np.float64)
    x_axis -= x_axis.mean()
    slope = (log_close @ x_axis) / np.sum(x_axis * x_axis)
    fitted = log_close.mean(axis=1, keepdims=True) + slope[:, None] * x_axis[None, :]
    blocks.extend([slope[:, None], _safe_std(log_close - fitted)[:, None]])
    running_max = np.maximum.accumulate(cl, axis=1)
    blocks.append(np.min(cl / np.maximum(running_max, eps) - 1.0, axis=1)[:, None])

    for z in (candle_range, candle_body, close_location, gap, true_range):
        blocks.extend([z.mean(axis=1)[:, None], _safe_std(z)[:, None], z[:, -1][:, None]])

    volume_median = np.maximum(np.median(vol, axis=1), eps)
    relative_volume = vol / volume_median[:, None]
    blocks.extend([
        np.mean(vol_change, axis=1)[:, None],
        _safe_std(vol_change)[:, None],
        relative_volume[:, -1][:, None],
        relative_volume[:, -5:].mean(axis=1)[:, None],
        (vol[:, -5:].mean(axis=1) / np.maximum(vol.mean(axis=1), eps))[:, None],
    ])
    r0 = ret - ret.mean(axis=1, keepdims=True)
    v0 = vol_change - vol_change.mean(axis=1, keepdims=True)
    corr = np.mean(r0 * v0, axis=1) / np.maximum(
        _safe_std(ret) * _safe_std(vol_change), eps
    )
    blocks.append(corr[:, None])
    result = np.concatenate([np.asarray(b, dtype=np.float32) for b in blocks], axis=1)
    return np.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0)


def contiguous_segments(days: np.ndarray):
    cuts = np.flatnonzero(np.diff(days) != 1) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, len(days)]
    for start, end in zip(starts, ends):
        yield int(start), int(end)


def build_training_matrices(df: pd.DataFrame, stride: int):
    up_xs, a1_xs = [], []
    ys, relative_ys = [], []
    anchor_days, paths20, group_ids, group_codes = [], [], [], []
    grouped = df.groupby("code", observed=True, sort=False)

    for group_no, (code, g) in enumerate(grouped, start=0):
        g = g.sort_values("day")
        days = g["day"].to_numpy(np.int32)
        values = g[[*PRICE_COLUMNS, "volume"]].to_numpy(np.float32)
        panel_a1 = g[list(A1_FEATURE_COLUMNS)].to_numpy(np.float32)
        panel_relative_y = g[A11_TARGET_COLUMN].to_numpy(np.uint8)
        group_codes.append(str(code))

        for start, end in contiguous_segments(days):
            n = end - start
            if n < WINDOW + HORIZON:
                continue
            local = np.arange(WINDOW - 1, n - HORIZON, stride, dtype=np.int32)
            absolute = start + local
            current = values[absolute, 3]
            future_idx = absolute[:, None] + np.arange(1, HORIZON + 1)[None, :]
            future_path = values[future_idx, 3] / current[:, None] - 1.0

            up_xs.append(make_window_features(values, absolute))
            a1_xs.append(panel_a1[absolute])
            ys.append((future_path[:, -1] > 0).astype(np.uint8))
            selected_relative_y = panel_relative_y[absolute]
            if np.any(selected_relative_y == 255):
                raise ValueError("Invalid A1.1 target survived a continuous window")
            relative_ys.append(selected_relative_y)
            anchor_days.append(days[absolute].astype(np.int16))
            paths20.append(future_path.astype(np.float32))
            group_ids.append(np.full(len(absolute), group_no, dtype=np.int16))

        if (group_no + 1) % 500 == 0:
            print(f"  processed {group_no + 1:,}/{grouped.ngroups:,} stocks", flush=True)

    x_up = np.vstack(up_xs)
    x_a1 = np.concatenate([x_up, np.vstack(a1_xs)], axis=1).astype(np.float32)
    return (
        x_up,
        x_a1,
        np.concatenate(ys),
        np.concatenate(relative_ys),
        np.concatenate(anchor_days),
        np.vstack(paths20),
        np.concatenate(group_ids),
        np.asarray(group_codes, dtype=object),
    )


def build_test_matrices(df: pd.DataFrame):
    up_xs, a1_xs, codes = [], [], []
    grouped = df.groupby("code", observed=True, sort=False)
    for code, g in grouped:
        g = g.sort_values("day")
        days = g["day"].to_numpy(np.int32)
        if len(g) < WINDOW or not np.all(np.diff(days[-WINDOW:]) == 1):
            raise ValueError(f"{code} lacks a continuous final 20-day test window")
        values = g[[*PRICE_COLUMNS, "volume"]].to_numpy(np.float32)[-WINDOW:]
        up_xs.append(make_window_features(values, np.array([WINDOW - 1], dtype=np.int32)))
        a1_xs.append(g[list(A1_FEATURE_COLUMNS)].to_numpy(np.float32)[-1])
        codes.append(str(code))
    x_up = np.vstack(up_xs)
    x_a1 = np.concatenate([x_up, np.vstack(a1_xs)], axis=1).astype(np.float32)
    return x_up, x_a1, np.asarray(codes, dtype=object)


def stable_code_bucket(code: str) -> int:
    return zlib.crc32(str(code).encode("utf-8")) % 100


def make_binary_model(seed: int, rounds: int | None = None):
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            objective="binary",
            n_estimators=rounds or 700,
            learning_rate=0.035,
            num_leaves=31,
            min_child_samples=220,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            reg_alpha=0.25,
            reg_lambda=2.5,
            random_state=seed,
            bagging_seed=seed,
            feature_fraction_seed=seed + 1,
            data_random_seed=seed + 2,
            n_jobs=-1,
            verbosity=-1,
        ), "lightgbm"
    except ImportError:
        return HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=rounds or 300,
            max_leaf_nodes=31,
            min_samples_leaf=220,
            l2_regularization=2.5,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
            random_state=seed,
        ), "sklearn_hist_gbdt"


def fit_binary(model, backend, x_train, y_train, x_valid=None, y_valid=None,
               sample_weight=None):
    if backend == "lightgbm" and x_valid is not None:
        from lightgbm import early_stopping, log_evaluation
        model.fit(
            x_train, y_train, sample_weight=sample_weight,
            eval_set=[(x_valid, y_valid)],
            eval_metric="binary_logloss",
            callbacks=[early_stopping(60, verbose=False), log_evaluation(100)],
        )
    else:
        model.fit(x_train, y_train, sample_weight=sample_weight)
    return model


def best_round_count(model, backend: str) -> int:
    if backend == "lightgbm":
        return int(getattr(model, "best_iteration_", 0) or model.n_estimators)
    return int(getattr(model, "n_iter_", 300))


def brier_score(y_true, probability) -> float:
    return float(np.mean((np.asarray(probability) - np.asarray(y_true)) ** 2))


def percentile_rank(values: np.ndarray) -> np.ndarray:
    if len(values) <= 1:
        return np.full(len(values), 0.5, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks / (len(values) - 1)


def choose_neutral_shrinkage(y_true, raw_probability):
    raw_center = float(np.mean(raw_probability))
    best_alpha, best_brier = 0.0, np.inf
    for alpha in np.linspace(0.0, 1.20, 61):
        p = np.clip(
            NEUTRAL_PROBABILITY_CENTER + alpha * (raw_probability - raw_center),
            *PROBABILITY_CLIP,
        )
        score = brier_score(y_true, p)
        if score < best_brier:
            best_alpha, best_brier = float(alpha), score
    return best_alpha, best_brier, raw_center


def build_independent_window_batches(group_ids, days, repeats, seed):
    rng = np.random.default_rng(seed)
    by_group = {}
    for idx in range(len(group_ids)):
        by_group.setdefault(int(group_ids[idx]), []).append(idx)
    by_group = {g: np.asarray(v, dtype=np.int32) for g, v in by_group.items() if len(v)}
    if len(by_group) < 200:
        raise RuntimeError("Too few cold-stock groups for independent-window validation")
    groups = np.asarray(sorted(by_group), dtype=np.int32)
    return [
        np.fromiter(
            (rng.choice(by_group[int(g)]) for g in groups),
            dtype=np.int32,
            count=len(groups),
        )
        for _ in range(repeats)
    ]


def build_same_day_batches(days: np.ndarray, min_assets: int = 200):
    """Coherent market cross-sections; diagnostic specific to A1."""
    result = []
    for day in np.unique(days):
        idx = np.flatnonzero(days == day).astype(np.int32)
        if len(idx) >= min_assets:
            result.append(idx)
    if not result:
        raise RuntimeError("No coherent same-day A1 validation batches were created")
    return result


def score_one_batch(y, paths, probability):
    top = np.argsort(-probability, kind="stable")[:5]
    selected_p = probability[top]
    weights = selected_p / selected_p.sum() if selected_p.sum() > 0 else np.full(5, 0.2)
    relative_value = 1.0 + paths[top]
    nav = 0.999 * (weights @ relative_value)
    nav[-1] *= 0.999
    full_nav = np.r_[1.0, nav]
    peak = np.maximum.accumulate(full_nav)
    mdd = float(np.max(1.0 - full_nav / peak))
    total_return = float(nav[-1] - 1.0)
    brier = brier_score(y, probability)
    score = 0.6 * total_return - 0.2 * mdd - 0.2 * brier
    return score, total_return, mdd, brier


def evaluate_fixed_tail_model(batches, y, paths, p_up, p_tail):
    rows = []
    for idx in batches:
        tail_rank = percentile_rank(p_tail[idx])
        probability = np.clip(
            p_up[idx] + V32_GAMMA * (tail_rank - 0.5), *PROBABILITY_CLIP
        )
        rows.append(score_one_batch(y[idx], paths[idx], probability))
    a = np.asarray(rows)
    return {
        "score_mean": float(a[:, 0].mean()),
        "score_median": float(np.median(a[:, 0])),
        "score_q20": float(np.quantile(a[:, 0], 0.20)),
        "return_mean": float(a[:, 1].mean()),
        "mdd_mean": float(a[:, 2].mean()),
        "brier_mean": float(a[:, 3].mean()),
        "batches": len(rows),
        "scores": a[:, 0].copy(),
    }


def evaluate_a11_overlay(batches, y, paths, p_up, p_absolute_tail,
                         p_relative_tail, relative_gamma):
    rows = []
    for idx in batches:
        absolute_rank = percentile_rank(p_absolute_tail[idx])
        relative_rank = percentile_rank(p_relative_tail[idx])
        probability = np.clip(
            p_up[idx]
            + V32_GAMMA * (absolute_rank - 0.5)
            + relative_gamma * (relative_rank - 0.5),
            *PROBABILITY_CLIP,
        )
        rows.append(score_one_batch(y[idx], paths[idx], probability))
    a = np.asarray(rows)
    return {
        "score_mean": float(a[:, 0].mean()),
        "score_median": float(np.median(a[:, 0])),
        "score_q20": float(np.quantile(a[:, 0], 0.20)),
        "return_mean": float(a[:, 1].mean()),
        "mdd_mean": float(a[:, 2].mean()),
        "brier_mean": float(a[:, 3].mean()),
        "batches": len(rows),
        "scores": a[:, 0].copy(),
    }


def mean_same_day_auc(batches, y_true, prediction):
    aucs = []
    for idx in batches:
        yy = y_true[idx]
        if np.unique(yy).size == 2:
            aucs.append(roc_auc_score(yy, prediction[idx]))
    if not aucs:
        raise RuntimeError("No same-day batch contains both A1.1 target classes")
    return float(np.mean(aucs)), float(np.median(aucs)), len(aucs)


def batch_probabilities(idx, p_up, p_absolute_tail, p_relative_tail):
    absolute_rank = percentile_rank(p_absolute_tail[idx])
    relative_rank = percentile_rank(p_relative_tail[idx])
    base = np.clip(
        p_up[idx] + V32_GAMMA * (absolute_rank - 0.5),
        *PROBABILITY_CLIP,
    )
    candidate = np.clip(
        base + A11_GAMMA * (relative_rank - 0.5),
        *PROBABILITY_CLIP,
    )
    return base, candidate, absolute_rank, relative_rank


def score_portfolio_k(y, paths, probability, k):
    k = min(int(k), len(probability))
    top = np.argsort(-probability, kind="stable")[:k]
    selected_p = probability[top]
    if selected_p.sum() > 0:
        weights = selected_p / selected_p.sum()
    else:
        weights = np.full(k, 1.0 / k)
    nav = 0.999 * (weights @ (1.0 + paths[top]))
    nav[-1] *= 0.999
    full_nav = np.r_[1.0, nav]
    peak = np.maximum.accumulate(full_nav)
    mdd = float(np.max(1.0 - full_nav / peak))
    total_return = float(nav[-1] - 1.0)
    brier = brier_score(y, probability)
    score = 0.6 * total_return - 0.2 * mdd - 0.2 * brier
    return {
        "score": score,
        "return": total_return,
        "mdd": mdd,
        "brier": brier,
        "top": top,
    }


def individual_path_mdd(paths):
    full_nav = np.concatenate(
        [np.ones((len(paths), 1), dtype=np.float32), 1.0 + paths], axis=1
    )
    peak = np.maximum.accumulate(full_nav, axis=1)
    return np.max(1.0 - full_nav / np.maximum(peak, 1e-12), axis=1)


def print_prediction_decile_diagnostic(
    batches, days, y, relative_y, paths, absolute_tail_y, relative_prediction
):
    rows = []
    ret20 = paths[:, -1]
    asset_mdd = individual_path_mdd(paths)
    for idx in batches:
        pred_rank = percentile_rank(relative_prediction[idx])
        future_rank = percentile_rank(ret20[idx])
        decile = np.minimum((pred_rank * 10.0).astype(np.int16), 9)
        rows.append(pd.DataFrame({
            "decile": decile,
            "pred_rank": pred_rank,
            "future_rank": future_rank,
            "future_ret": ret20[idx],
            "up": y[idx],
            "absolute_q80": absolute_tail_y[idx],
            "relative_top20": relative_y[idx],
            "asset_mdd": asset_mdd[idx],
        }))
    frame = pd.concat(rows, ignore_index=True)
    table = frame.groupby("decile", sort=True).agg(
        n=("future_ret", "size"),
        pred_rank_mean=("pred_rank", "mean"),
        future_rank_mean=("future_rank", "mean"),
        future_ret_mean=("future_ret", "mean"),
        future_ret_median=("future_ret", "median"),
        up_rate=("up", "mean"),
        absolute_q80_rate=("absolute_q80", "mean"),
        relative_top20_rate=("relative_top20", "mean"),
        asset_mdd_mean=("asset_mdd", "mean"),
    ).reset_index()
    means = table["future_ret_mean"].to_numpy(np.float64)
    rank_means = table["future_rank_mean"].to_numpy(np.float64)
    adjacent_up = int(np.sum(np.diff(means) > 0))
    return_corr = float(np.corrcoef(table["decile"], means)[0, 1])
    rank_corr = float(np.corrcoef(table["decile"], rank_means)[0, 1])

    print("\nDIAGNOSTIC 1/4: predicted relative-score decile monotonicity")
    print("  decile 0=lowest predicted relative strength; 9=highest")
    print(table.to_string(index=False, formatters={
        "pred_rank_mean": "{:.3f}".format,
        "future_rank_mean": "{:.3f}".format,
        "future_ret_mean": "{:.3%}".format,
        "future_ret_median": "{:.3%}".format,
        "up_rate": "{:.3f}".format,
        "absolute_q80_rate": "{:.3f}".format,
        "relative_top20_rate": "{:.3f}".format,
        "asset_mdd_mean": "{:.3%}".format,
    }))
    print(
        f"  adjacent mean-return increases={adjacent_up}/9; "
        f"decile/mean-return corr={return_corr:.4f}; "
        f"decile/future-rank corr={rank_corr:.4f}"
    )
    print(
        f"  top-minus-bottom mean-return spread="
        f"{means[-1] - means[0]:+.3%}; "
        f"future-rank spread={rank_means[-1] - rank_means[0]:+.3f}"
    )
    return table


def print_topk_diagnostic(
    batches, y, paths, p_up, absolute_prediction, relative_prediction
):
    rows = []
    for batch_no, idx in enumerate(batches):
        base_p, candidate_p, _, _ = batch_probabilities(
            idx, p_up, absolute_prediction, relative_prediction
        )
        for k in DIAGNOSTIC_TOP_K:
            base = score_portfolio_k(y[idx], paths[idx], base_p, k)
            candidate = score_portfolio_k(y[idx], paths[idx], candidate_p, k)
            overlap = len(set(base["top"]) & set(candidate["top"])) / min(k, len(idx))
            rows.append({
                "batch": batch_no,
                "k": k,
                "base_score": base["score"],
                "candidate_score": candidate["score"],
                "base_return": base["return"],
                "candidate_return": candidate["return"],
                "base_mdd": base["mdd"],
                "candidate_mdd": candidate["mdd"],
                "overlap": overlap,
            })
    frame = pd.DataFrame(rows)
    summaries = []
    for k, g in frame.groupby("k", sort=True):
        score_delta = g["candidate_score"] - g["base_score"]
        return_delta = g["candidate_return"] - g["base_return"]
        summaries.append({
            "K": int(k),
            "base_ret": g["base_return"].mean(),
            "A11_ret": g["candidate_return"].mean(),
            "ret_delta": return_delta.mean(),
            "base_mdd": g["base_mdd"].mean(),
            "A11_mdd": g["candidate_mdd"].mean(),
            "score_delta": score_delta.mean(),
            "win_rate": np.mean(score_delta > 0),
            "topK_overlap": g["overlap"].mean(),
        })
    table = pd.DataFrame(summaries)
    print("\nDIAGNOSTIC 2/4: portfolio performance at different K")
    print(table.to_string(index=False, formatters={
        "base_ret": "{:.3%}".format,
        "A11_ret": "{:.3%}".format,
        "ret_delta": "{:+.3%}".format,
        "base_mdd": "{:.3%}".format,
        "A11_mdd": "{:.3%}".format,
        "score_delta": "{:+.6f}".format,
        "win_rate": "{:.3f}".format,
        "topK_overlap": "{:.3f}".format,
    }))
    return frame, table


def _selection_summary(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return np.nan, np.nan, 0
    return float(values.mean()), float(np.mean(values > 0)), len(values)


def print_top5_replacement_diagnostic(
    batches, days, y, relative_y, paths, absolute_tail_y,
    p_up, absolute_prediction, relative_prediction
):
    ret20 = paths[:, -1]
    day_rows, removed_ret, added_ret = [], [], []
    removed_abs, added_abs, removed_rel, added_rel = [], [], [], []
    removed_pred_rank, added_pred_rank = [], []

    for idx in batches:
        base_p, candidate_p, _, relative_rank = batch_probabilities(
            idx, p_up, absolute_prediction, relative_prediction
        )
        base = score_portfolio_k(y[idx], paths[idx], base_p, 5)
        candidate = score_portfolio_k(y[idx], paths[idx], candidate_p, 5)
        base_set, candidate_set = set(base["top"]), set(candidate["top"])
        removed = np.asarray(sorted(base_set - candidate_set), dtype=np.int32)
        added = np.asarray(sorted(candidate_set - base_set), dtype=np.int32)
        kept = len(base_set & candidate_set)

        if len(removed):
            removed_ret.extend(ret20[idx][removed])
            removed_abs.extend(absolute_tail_y[idx][removed])
            removed_rel.extend(relative_y[idx][removed])
            removed_pred_rank.extend(relative_rank[removed])
        if len(added):
            added_ret.extend(ret20[idx][added])
            added_abs.extend(absolute_tail_y[idx][added])
            added_rel.extend(relative_y[idx][added])
            added_pred_rank.extend(relative_rank[added])

        day_rows.append({
            "day": int(days[idx[0]]),
            "replacements": len(added),
            "kept": kept,
            "base_return": base["return"],
            "candidate_return": candidate["return"],
            "return_delta": candidate["return"] - base["return"],
            "score_delta": candidate["score"] - base["score"],
        })

    day_frame = pd.DataFrame(day_rows)
    removed_ret_mean, removed_up, removed_n = _selection_summary(removed_ret)
    added_ret_mean, added_up, added_n = _selection_summary(added_ret)
    detail = pd.DataFrame([
        {
            "selection": "removed_by_A11",
            "n": removed_n,
            "future_ret_mean": removed_ret_mean,
            "up_rate": removed_up,
            "absolute_q80_rate": np.mean(removed_abs) if removed_abs else np.nan,
            "relative_top20_rate": np.mean(removed_rel) if removed_rel else np.nan,
            "relative_pred_rank": np.mean(removed_pred_rank) if removed_pred_rank else np.nan,
        },
        {
            "selection": "added_by_A11",
            "n": added_n,
            "future_ret_mean": added_ret_mean,
            "up_rate": added_up,
            "absolute_q80_rate": np.mean(added_abs) if added_abs else np.nan,
            "relative_top20_rate": np.mean(added_rel) if added_rel else np.nan,
            "relative_pred_rank": np.mean(added_pred_rank) if added_pred_rank else np.nan,
        },
    ])
    by_count = day_frame.groupby("replacements", sort=True).agg(
        days=("day", "size"),
        mean_return_delta=("return_delta", "mean"),
        median_return_delta=("return_delta", "median"),
        mean_score_delta=("score_delta", "mean"),
        win_rate=("score_delta", lambda x: np.mean(x > 0)),
    ).reset_index()

    print("\nDIAGNOSTIC 3/4: exact Top5 replacements caused by A1.1")
    print(
        f"  days={len(day_frame)}, mean kept={day_frame['kept'].mean():.3f}/5, "
        f"unchanged days={np.mean(day_frame['replacements'] == 0):.3f}, "
        f"mean replacements={day_frame['replacements'].mean():.3f}"
    )
    print(detail.to_string(index=False, formatters={
        "future_ret_mean": "{:.3%}".format,
        "up_rate": "{:.3f}".format,
        "absolute_q80_rate": "{:.3f}".format,
        "relative_top20_rate": "{:.3f}".format,
        "relative_pred_rank": "{:.3f}".format,
    }))
    print("  Performance grouped by number of Top5 replacements:")
    print(by_count.to_string(index=False, formatters={
        "mean_return_delta": "{:+.3%}".format,
        "median_return_delta": "{:+.3%}".format,
        "mean_score_delta": "{:+.6f}".format,
        "win_rate": "{:.3f}".format,
    }))
    return day_frame, detail, by_count


def print_future_market_regime_diagnostic(
    batches, days, y, paths, p_up, absolute_prediction, relative_prediction
):
    ret20 = paths[:, -1]
    rows = []
    for idx in batches:
        base_p, candidate_p, _, _ = batch_probabilities(
            idx, p_up, absolute_prediction, relative_prediction
        )
        base = score_portfolio_k(y[idx], paths[idx], base_p, 5)
        candidate = score_portfolio_k(y[idx], paths[idx], candidate_p, 5)
        rows.append({
            "day": int(days[idx[0]]),
            "market_future_median": float(np.median(ret20[idx])),
            "market_future_mean": float(np.mean(ret20[idx])),
            "base_return": base["return"],
            "candidate_return": candidate["return"],
            "base_mdd": base["mdd"],
            "candidate_mdd": candidate["mdd"],
            "base_score": base["score"],
            "candidate_score": candidate["score"],
        })
    frame = pd.DataFrame(rows)
    q33, q67 = np.quantile(frame["market_future_median"], [1 / 3, 2 / 3])
    frame["regime"] = pd.cut(
        frame["market_future_median"],
        bins=[-np.inf, q33, q67, np.inf],
        labels=["bear", "neutral", "bull"],
        include_lowest=True,
    )
    summaries = []
    for regime, g in frame.groupby("regime", observed=True, sort=True):
        score_delta = g["candidate_score"] - g["base_score"]
        summaries.append({
            "regime": str(regime),
            "days": len(g),
            "market_median": g["market_future_median"].mean(),
            "base_ret": g["base_return"].mean(),
            "A11_ret": g["candidate_return"].mean(),
            "ret_delta": (g["candidate_return"] - g["base_return"]).mean(),
            "base_mdd": g["base_mdd"].mean(),
            "A11_mdd": g["candidate_mdd"].mean(),
            "score_delta": score_delta.mean(),
            "win_rate": np.mean(score_delta > 0),
        })
    table = pd.DataFrame(summaries)
    print("\nDIAGNOSTIC 4/4: Top5 performance by future market regime")
    print(
        f"  regime boundaries use validation-only future market median: "
        f"q33={q33:.3%}, q67={q67:.3%}"
    )
    print("  This is explanatory only; future regime is unavailable at inference.")
    print(table.to_string(index=False, formatters={
        "market_median": "{:.3%}".format,
        "base_ret": "{:.3%}".format,
        "A11_ret": "{:.3%}".format,
        "ret_delta": "{:+.3%}".format,
        "base_mdd": "{:.3%}".format,
        "A11_mdd": "{:.3%}".format,
        "score_delta": "{:+.6f}".format,
        "win_rate": "{:.3f}".format,
    }))
    return frame, table


def run_full_a11_diagnostics(
    batches, days, y, relative_y, paths, absolute_tail_y,
    p_up, absolute_prediction, relative_prediction
):
    print("\n" + "=" * 78)
    print("A1.1 FIXED-MODEL DIAGNOSTIC REPORT (strategy holdout only)")
    print("=" * 78)
    print_prediction_decile_diagnostic(
        batches, days, y, relative_y, paths,
        absolute_tail_y, relative_prediction,
    )
    print_topk_diagnostic(
        batches, y, paths, p_up, absolute_prediction, relative_prediction
    )
    print_top5_replacement_diagnostic(
        batches, days, y, relative_y, paths, absolute_tail_y,
        p_up, absolute_prediction, relative_prediction,
    )
    print_future_market_regime_diagnostic(
        batches, days, y, paths,
        p_up, absolute_prediction, relative_prediction,
    )
    print("=" * 78)


def print_metrics(name, metrics):
    print(
        f"  {name:10s}: mean={metrics['score_mean']:.6f} "
        f"median={metrics['score_median']:.6f} q20={metrics['score_q20']:.6f} "
        f"ret={metrics['return_mean']:.3%} mdd={metrics['mdd_mean']:.3%} "
        f"brier={metrics['brier_mean']:.6f} batches={metrics['batches']}"
    )


def validate_submission(submission: pd.DataFrame, expected_codes: np.ndarray):
    if list(submission.columns) != ["code", "up_factor"]:
        raise AssertionError("Submission columns must be code,up_factor")
    if len(submission) != len(expected_codes):
        raise AssertionError("Wrong number of submission rows")
    if submission["code"].duplicated().any():
        raise AssertionError("Duplicate code in submission")
    if set(submission["code"]) != set(expected_codes):
        raise AssertionError("Submission code set differs from test code set")
    p = submission["up_factor"].to_numpy()
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise AssertionError("Invalid up_factor probability")


def main():
    print("V3.2-A1.1 offline diagnostic run")
    print(f"  train:  {TRAIN_PATH}")
    print(f"  test:   {TEST_PATH}")
    print("  mode:   diagnostic-only; no submission will be written")
    print(
        f"  base absolute-tail gamma={V32_GAMMA}; "
        f"candidate relative-tail gamma={A11_GAMMA}"
    )
    print(
        f"  relative label=top {1.0 - RELATIVE_TAIL_QUANTILE:.0%} "
        f"within anchor day; candidate features=59+{len(A1_FEATURE_COLUMNS)}"
    )

    with timer("Read and validate CSV files"):
        train = read_market_csv(TRAIN_PATH)
        test = read_market_csv(TEST_PATH)
        print(f"  train shape={train.shape}, codes={train['code'].nunique():,}")
        print(f"  test  shape={test.shape}, codes={test['code'].nunique():,}")

    with timer("Build leakage-safe A1 features and group-relative train target"):
        train = add_a1_cross_sectional_returns(train, "train")
        test = add_a1_cross_sectional_returns(test, "test")
        train = add_a11_relative_future_target(train)

    with timer("Build sorted-row V3 base and A1.1 candidate matrices"):
        (
            x_up, x_a1, y, relative_y, days, paths, group_ids, group_codes
        ) = build_training_matrices(train, TRAIN_STRIDE)
        x_up_test, x_a1_test, test_codes = build_test_matrices(test)
        ret20 = paths[:, -1]
        print(f"  base={x_up.shape}, A1.1={x_a1.shape}, test={x_up_test.shape}")
        print(f"  sampled relative-tail rate={relative_y.mean():.4f}")
        del train, test
        gc.collect()

    buckets_by_group = np.asarray(
        [stable_code_bucket(c) for c in group_codes], dtype=np.int16
    )
    sample_bucket = buckets_by_group[group_ids]
    model_train = (sample_bucket < HOLDOUT_CAL_BUCKET_START) & (days <= MODEL_TRAIN_END_DAY)
    calibration = (
        (sample_bucket >= HOLDOUT_CAL_BUCKET_START)
        & (sample_bucket < HOLDOUT_STRATEGY_BUCKET_START)
        & (days >= CALIBRATION_START_DAY)
        & (days <= CALIBRATION_END_DAY)
    )
    strategy = (sample_bucket >= HOLDOUT_STRATEGY_BUCKET_START) & (days >= STRATEGY_START_DAY)
    print("\nFixed V3.2 code- and time-disjoint split")
    print(
        f"  train={model_train.sum():,}, calibration={calibration.sum():,}, "
        f"strategy={strategy.sum():,}"
    )

    tail_threshold = float(np.quantile(ret20[model_train], TAIL_QUANTILE))
    tail_y = (ret20 > tail_threshold).astype(np.uint8)
    print(f"  absolute tail q{TAIL_QUANTILE:.0%} threshold={tail_threshold:.4%}")
    print(
        f"  relative-tail rates train/cal/strategy="
        f"{relative_y[model_train].mean():.4f}/"
        f"{relative_y[calibration].mean():.4f}/"
        f"{relative_y[strategy].mean():.4f}"
    )

    with timer("Fit unchanged V3 up model"):
        up_model, backend = make_binary_model(RANDOM_STATE)
        up_model = fit_binary(
            up_model, backend,
            x_up[model_train], y[model_train],
            x_up[calibration], y[calibration],
        )
        up_rounds = best_round_count(up_model, backend)
        raw_up_cal = up_model.predict_proba(x_up[calibration])[:, 1]
        raw_up_strategy = up_model.predict_proba(x_up[strategy])[:, 1]

    alpha, cal_brier, raw_cal_center = choose_neutral_shrinkage(y[calibration], raw_up_cal)
    p_up_strategy = np.clip(
        NEUTRAL_PROBABILITY_CENTER + alpha * (raw_up_strategy - raw_cal_center),
        *PROBABILITY_CLIP,
    )
    print(
        f"  up AUC={roc_auc_score(y[strategy], raw_up_strategy):.6f}, "
        f"alpha={alpha:.2f}, cal Brier={cal_brier:.6f}"
    )
    del up_model
    gc.collect()

    with timer("Fit unchanged absolute-tail base and A1.1 relative-tail model"):
        absolute_tail, absolute_backend = make_binary_model(RANDOM_STATE + 101)
        relative_tail, relative_backend = make_binary_model(RANDOM_STATE + 211)
        if absolute_backend != backend or relative_backend != backend:
            raise RuntimeError("Model backends disagree")
        absolute_tail = fit_binary(
            absolute_tail, backend,
            x_up[model_train], tail_y[model_train],
            x_up[calibration], tail_y[calibration],
        )
        relative_tail = fit_binary(
            relative_tail, backend,
            x_a1[model_train], relative_y[model_train],
            x_a1[calibration], relative_y[calibration],
        )
        absolute_rounds = best_round_count(absolute_tail, backend)
        relative_rounds = best_round_count(relative_tail, backend)
        absolute_pred = absolute_tail.predict_proba(x_up[strategy])[:, 1]
        relative_pred = relative_tail.predict_proba(x_a1[strategy])[:, 1]

    absolute_auc = roc_auc_score(tail_y[strategy], absolute_pred)
    relative_auc = roc_auc_score(relative_y[strategy], relative_pred)
    print("\nA1.1 model diagnostics")
    print(f"  absolute-tail AUC={absolute_auc:.6f}, rounds={absolute_rounds}")
    print(f"  relative-tail pooled AUC={relative_auc:.6f}, rounds={relative_rounds}")

    strategy_groups = group_ids[strategy]
    strategy_days = days[strategy]
    independent_batches = build_independent_window_batches(
        strategy_groups, strategy_days, PSEUDO_BATCH_REPEATS, RANDOM_STATE
    )
    same_day_batches = build_same_day_batches(strategy_days)

    same_auc_mean, same_auc_median, same_auc_days = mean_same_day_auc(
        same_day_batches, relative_y[strategy], relative_pred
    )
    print(
        f"  relative-tail same-day AUC mean/median="
        f"{same_auc_mean:.6f}/{same_auc_median:.6f} ({same_auc_days} days)"
    )

    independent_base = evaluate_fixed_tail_model(
        independent_batches, y[strategy], paths[strategy], p_up_strategy, absolute_pred
    )
    independent_a11 = evaluate_a11_overlay(
        independent_batches, y[strategy], paths[strategy],
        p_up_strategy, absolute_pred, relative_pred, A11_GAMMA,
    )
    same_day_base = evaluate_fixed_tail_model(
        same_day_batches, y[strategy], paths[strategy], p_up_strategy, absolute_pred
    )
    same_day_a11 = evaluate_a11_overlay(
        same_day_batches, y[strategy], paths[strategy],
        p_up_strategy, absolute_pred, relative_pred, A11_GAMMA,
    )

    print("\nExisting V3.2 independent-window diagnostic")
    print_metrics("baseline", independent_base)
    print_metrics("A1.1", independent_a11)
    print("\nPrimary coherent same-day cross-section diagnostic")
    print_metrics("baseline", same_day_base)
    print_metrics("A1.1", same_day_a11)

    run_full_a11_diagnostics(
        same_day_batches,
        strategy_days,
        y[strategy],
        relative_y[strategy],
        paths[strategy],
        tail_y[strategy],
        p_up_strategy,
        absolute_pred,
        relative_pred,
    )

    paired_delta = same_day_a11["scores"] - same_day_base["scores"]
    mean_gain = same_day_a11["score_mean"] - same_day_base["score_mean"]
    median_gain = same_day_a11["score_median"] - same_day_base["score_median"]
    paired_win_rate = float(np.mean(paired_delta > 0.0))
    independent_mean_drop = independent_base["score_mean"] - independent_a11["score_mean"]

    auc_pass = same_auc_mean >= A11_MIN_SAME_DAY_AUC
    mean_pass = mean_gain >= A11_MIN_MEAN_SCORE_GAIN
    median_pass = median_gain > 0.0
    win_pass = paired_win_rate >= A11_MIN_PAIRED_WIN_RATE
    q20_pass = (
        same_day_a11["score_q20"]
        >= same_day_base["score_q20"] - A11_MAX_Q20_SCORE_DROP
    )
    brier_pass = abs(
        same_day_a11["brier_mean"] - same_day_base["brier_mean"]
    ) <= A11_MAX_BRIER_CHANGE
    independent_safety_pass = independent_mean_drop <= A11_MAX_INDEPENDENT_MEAN_DROP
    use_a11 = all([
        auc_pass, mean_pass, median_pass, win_pass, q20_pass,
        brier_pass, independent_safety_pass,
    ])

    print("\nPre-registered A1.1 acceptance gate")
    print(f"  same-day AUC >= {A11_MIN_SAME_DAY_AUC:.3f}          : {auc_pass}")
    print(f"  same-day mean gain >= {A11_MIN_MEAN_SCORE_GAIN:.3f} : {mean_pass} ({mean_gain:+.6f})")
    print(f"  same-day median improves        : {median_pass} ({median_gain:+.6f})")
    print(f"  paired win rate >= {A11_MIN_PAIRED_WIN_RATE:.2f}     : {win_pass} ({paired_win_rate:.3f})")
    print(f"  same-day q20 protected          : {q20_pass}")
    print(f"  Brier change <= {A11_MAX_BRIER_CHANGE:.4f}       : {brier_pass}")
    print(
        f"  independent mean drop <= {A11_MAX_INDEPENDENT_MEAN_DROP:.3f}: "
        f"{independent_safety_pass} (drop={independent_mean_drop:+.6f})"
    )
    print(f"  selected overlay               : {'A1.1' if use_a11 else 'none; exact base fallback'}")

    if DIAGNOSTIC_ONLY:
        print("\nDiagnostic-only run complete; no submission file was written.")
        return

    del absolute_tail, relative_tail, absolute_pred, relative_pred
    gc.collect()

    max_day = max(int(days.max()), 1)
    sample_weight = (0.60 + 0.40 * (days.astype(np.float32) / max_day) ** 2).astype(np.float32)

    with timer("Refit final unchanged up model"):
        final_up, final_backend = make_binary_model(RANDOM_STATE, up_rounds)
        if final_backend == "sklearn_hist_gbdt":
            final_up.set_params(early_stopping=False)
        final_up = fit_binary(final_up, final_backend, x_up, y, sample_weight=sample_weight)
        raw_up_test = final_up.predict_proba(x_up_test)[:, 1]

    with timer("Refit final unchanged absolute-tail model"):
        final_tail, final_tail_backend = make_binary_model(
            RANDOM_STATE + 101, absolute_rounds
        )
        if final_tail_backend == "sklearn_hist_gbdt":
            final_tail.set_params(early_stopping=False)
        final_tail = fit_binary(
            final_tail, final_tail_backend, x_up, tail_y, sample_weight=sample_weight
        )
        raw_tail_test = final_tail.predict_proba(x_up_test)[:, 1]

    raw_relative_test = None
    if use_a11:
        with timer("Refit accepted A1.1 relative-tail model"):
            final_relative, final_relative_backend = make_binary_model(
                RANDOM_STATE + 211, relative_rounds
            )
            if final_relative_backend == "sklearn_hist_gbdt":
                final_relative.set_params(early_stopping=False)
            final_relative = fit_binary(
                final_relative, final_relative_backend,
                x_a1, relative_y, sample_weight=sample_weight,
            )
            raw_relative_test = final_relative.predict_proba(x_a1_test)[:, 1]

    p_up_test = np.clip(
        NEUTRAL_PROBABILITY_CENTER
        + alpha * (raw_up_test - float(raw_up_test.mean())),
        *PROBABILITY_CLIP,
    )
    tail_rank_test = percentile_rank(raw_tail_test)
    base_probability = np.clip(
        p_up_test + V32_GAMMA * (tail_rank_test - 0.5), *PROBABILITY_CLIP
    )
    test_probability = base_probability.copy()
    if use_a11:
        relative_rank_test = percentile_rank(raw_relative_test)
        test_probability = np.clip(
            base_probability + A11_GAMMA * (relative_rank_test - 0.5),
            *PROBABILITY_CLIP,
        )

    if REFERENCE_BASELINE_PATH.exists():
        reference = pd.read_csv(REFERENCE_BASELINE_PATH)
        reference = reference.set_index("code").reindex(test_codes)
        if reference["up_factor"].isna().any():
            print("  reference baseline exists but code coverage differs")
        else:
            max_base_diff = float(np.max(np.abs(
                reference["up_factor"].to_numpy(np.float64) - base_probability
            )))
            print(f"  max difference vs saved 0.03403 base={max_base_diff:.10f}")
    submission = pd.DataFrame({
        "code": test_codes,
        "up_factor": test_probability.astype(np.float64),
    })
    validate_submission(submission, test_codes)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUTPUT_PATH, index=False, encoding="utf-8", float_format="%.8f")
    check = pd.read_csv(OUTPUT_PATH)
    validate_submission(check, test_codes)

    print("\nV3.2-A1.1 submission created")
    print(f"  path={OUTPUT_PATH}")
    print(f"  selected overlay={'A1.1' if use_a11 else 'none; exact base fallback'}")
    print(f"  rows={len(check):,}")
    print(f"  probability range=[{check.up_factor.min():.6f}, {check.up_factor.max():.6f}]")
    print(
        check.sort_values(["up_factor", "code"], ascending=[False, True])
        .head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
