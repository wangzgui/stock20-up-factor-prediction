"""V5-A robust-return-regression ablation on the exact cold-stock framework.

The unchanged V3 q80 tail classifier is compared with a winsorized Huber
regressor.  Both use the same 59 features, up-probability model, gamma, exact
1,500-stock synchronous queries, code-disjoint roles, and non-overlapping
20-day time folds.  This diagnostic reads train.csv only and writes no
submission file.
"""

from __future__ import annotations

import gc
import time
import warnings
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TRAIN_PATH = Path("/kaggle/input/datasets/zhuowamg/test-data/train.csv")

WINDOW = 20
HORIZON = 20
TRAIN_STRIDE = 10
RANDOM_STATE = 20260822
PROBABILITY_CLIP = (0.02, 0.98)
REGRESSION_WINSOR_LOWER = 0.01
REGRESSION_WINSOR_UPPER = 0.99

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
GLOBAL_ANCHOR_PHASE = 1
MODEL_CODE_COUNT = 1775
CALIBRATION_CODE_COUNT = 400
COLD_CODE_COUNT = 2200
EXACT_COLD_QUERY_SIZE = 1500
FOLDS = (
    {"name": "fold_1", "train_end": 1600,
     "cal_start": 1641, "cal_end": 1800,
     "strategy_start": 1841, "strategy_end": 2074},
    {"name": "fold_2", "train_end": 1950,
     "cal_start": 1991, "cal_end": 2150,
     "strategy_start": 2191, "strategy_end": 2424},
    {"name": "fold_3", "train_end": 2300,
     "cal_start": 2341, "cal_end": 2500,
     "strategy_start": 2541, "strategy_end": 2774},
)
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
        panel_a1 = (
            g[list(A1_FEATURE_COLUMNS)].to_numpy(np.float32)
            if all(c in g.columns for c in A1_FEATURE_COLUMNS) else None
        )
        panel_relative_y = (
            g[A11_TARGET_COLUMN].to_numpy(np.uint8)
            if A11_TARGET_COLUMN in g.columns else None
        )
        group_codes.append(str(code))

        for start, end in contiguous_segments(days):
            n = end - start
            if n < WINDOW + HORIZON:
                continue
            # All stocks must share the same global DAY grid.  A per-stock
            # ``arange(..., stride)`` silently gives different anchor dates to
            # late-listed stocks and therefore breaks a same-day query.
            all_local = np.arange(WINDOW - 1, n - HORIZON, dtype=np.int32)
            all_absolute = start + all_local
            keep = ((days[all_absolute] - GLOBAL_ANCHOR_PHASE) % stride) == 0
            absolute = all_absolute[keep]
            if len(absolute) == 0:
                continue
            current = values[absolute, 3]
            future_idx = absolute[:, None] + np.arange(1, HORIZON + 1)[None, :]
            future_path = values[future_idx, 3] / current[:, None] - 1.0

            up_xs.append(make_window_features(values, absolute))
            if panel_a1 is not None:
                a1_xs.append(panel_a1[absolute])
            ys.append((future_path[:, -1] > 0).astype(np.uint8))
            if panel_relative_y is not None:
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
    x_a1 = (
        np.concatenate([x_up, np.vstack(a1_xs)], axis=1).astype(np.float32)
        if a1_xs else x_up
    )
    return (
        x_up,
        x_a1,
        np.concatenate(ys),
        (np.concatenate(relative_ys) if relative_ys
         else np.zeros(len(x_up), dtype=np.uint8)),
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


def make_robust_regression_model(seed: int):
    try:
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            objective="huber",
            alpha=0.90,
            n_estimators=700,
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
        return HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.06,
            max_iter=300,
            max_leaf_nodes=31,
            min_samples_leaf=220,
            l2_regularization=2.5,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
            random_state=seed,
        ), "sklearn_hist_gbdt"


def fit_regression(model, backend, x_train, y_train, x_valid=None, y_valid=None):
    if backend == "lightgbm" and x_valid is not None:
        from lightgbm import early_stopping, log_evaluation
        model.fit(
            x_train, y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric="l1",
            callbacks=[early_stopping(60, verbose=False), log_evaluation(100)],
        )
    else:
        model.fit(x_train, y_train)
    return model


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


def official_order(probability: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Descending probability, then ascending code as the official tie-break."""
    probability = np.asarray(probability, dtype=np.float64)
    code_text = np.asarray(codes, dtype=str)
    return np.lexsort((code_text, -probability)).astype(np.int32)


def choose_pool50_top5(base_probability, relative_prediction, codes):
    if len(base_probability) < POOL50_SIZE:
        raise ValueError(f"Test universe has fewer than {POOL50_SIZE} assets")
    relative_order = official_order(relative_prediction, codes)
    pool = relative_order[:POOL50_SIZE]
    pool_order = pool[
        np.lexsort((np.asarray(codes, dtype=str)[pool], -base_probability[pool]))
    ]
    desired_top5 = pool_order[:5].astype(np.int32)
    return pool.astype(np.int32), desired_top5


def encode_top5_by_local_probability_permutation(
    base_probability, desired_top5, codes
):
    """Make desired codes Top5 while preserving the full probability multiset."""
    base_probability = np.asarray(base_probability, dtype=np.float64)
    desired_top5 = np.asarray(desired_top5, dtype=np.int32)
    if len(np.unique(desired_top5)) != 5:
        raise ValueError("Desired Top5 must contain five unique rows")

    original_top5 = official_order(base_probability, codes)[:5]
    union = np.union1d(original_top5, desired_top5).astype(np.int32)
    probability_values = np.sort(base_probability[union])[::-1]

    desired_order = desired_top5[
        np.lexsort((
            np.asarray(codes, dtype=str)[desired_top5],
            -base_probability[desired_top5],
        ))
    ]
    remaining = union[~np.isin(union, desired_top5)]
    if len(remaining):
        remaining = remaining[
            np.lexsort((
                np.asarray(codes, dtype=str)[remaining],
                -base_probability[remaining],
            ))
        ]
    assignment_order = np.r_[desired_order, remaining]
    if len(assignment_order) != len(probability_values):
        raise AssertionError("Local probability permutation size mismatch")

    encoded = base_probability.copy()
    encoded[assignment_order] = probability_values
    if not np.array_equal(np.sort(encoded), np.sort(base_probability)):
        raise AssertionError("Probability multiset changed during encoding")
    encoded_top5 = official_order(encoded, codes)[:5]
    if set(encoded_top5) != set(desired_top5):
        raise AssertionError("Encoded probabilities do not select desired Top5")

    changed = np.flatnonzero(encoded != base_probability)
    # Label-free worst-case upper bound on the increase in mean Brier score.
    old = base_probability[changed]
    new = encoded[changed]
    if len(changed):
        delta_y0 = new ** 2 - old ** 2
        delta_y1 = (new - 1.0) ** 2 - (old - 1.0) ** 2
        worst_brier_increase = float(
            np.maximum.reduce([delta_y0, delta_y1, np.zeros(len(changed))]).sum()
            / len(base_probability)
        )
    else:
        worst_brier_increase = 0.0
    return encoded, original_top5, changed, worst_brier_increase


def legacy_pool50_main_disabled():
    print("V3.2-Pool50 relative-candidate/base-final submission")
    print(f"  train:  {TRAIN_PATH}")
    print(f"  test:   {TEST_PATH}")
    print(f"  output: {OUTPUT_PATH}")
    print(f"  base absolute-tail gamma={V32_GAMMA}")
    print(f"  relative candidate pool=Top{POOL50_SIZE}; final selection=base Top5 in pool")
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
    print(
        "  Pool50 is a separate fixed intersection rule and will be used; "
        "the failed linear-overlay gate does not disable it."
    )

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

    with timer("Refit final A1.1 relative candidate-pool model"):
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
    relative_pool, desired_top5 = choose_pool50_top5(
        base_probability, raw_relative_test, test_codes
    )
    (
        test_probability,
        original_top5,
        changed_rows,
        worst_brier_increase,
    ) = encode_top5_by_local_probability_permutation(
        base_probability, desired_top5, test_codes
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
    checked_probability = (
        check.set_index("code").reindex(test_codes)["up_factor"].to_numpy(np.float64)
    )
    checked_top5 = official_order(checked_probability, test_codes)[:5]
    if set(checked_top5) != set(desired_top5):
        raise AssertionError("CSV rounding changed the desired official Top5")

    print("\nPool50 selection audit")
    print(f"  relative pool size={len(relative_pool)}")
    print(f"  original/desired Top5 overlap={len(set(original_top5) & set(desired_top5))}/5")
    print(f"  probability rows changed={len(changed_rows)}")
    print(f"  probability multiset preserved exactly=True")
    print(f"  label-free worst Brier increase bound={worst_brier_increase:.8f}")
    print(f"  worst composite-score Brier penalty bound={0.2 * worst_brier_increase:.8f}")
    audit = pd.DataFrame({
        "code": test_codes,
        "base_probability": base_probability,
        "final_probability": test_probability,
        "relative_rank": percentile_rank(raw_relative_test),
        "in_relative_top50": np.isin(np.arange(len(test_codes)), relative_pool),
        "original_top5": np.isin(np.arange(len(test_codes)), original_top5),
        "desired_top5": np.isin(np.arange(len(test_codes)), desired_top5),
    })
    print(
        audit[audit["original_top5"] | audit["desired_top5"]]
        .sort_values(["desired_top5", "final_probability"], ascending=[False, False])
        .to_string(index=False)
    )

    print("\nV3.2-Pool50 submission created")
    print(f"  path={OUTPUT_PATH}")
    print("  selection=relative Top50 candidate pool -> base final Top5")
    print(f"  rows={len(check):,}")
    print(f"  probability range=[{check.up_factor.min():.6f}, {check.up_factor.max():.6f}]")
    print(
        check.sort_values(["up_factor", "code"], ascending=[False, True])
        .head(10).to_string(index=False)
    )


def stable_code_hash(code: str) -> int:
    return zlib.crc32(str(code).encode("utf-8")) & 0xFFFFFFFF


def make_code_roles(group_codes: np.ndarray) -> np.ndarray:
    """Fixed stock-disjoint roles: model, calibration, cold strategy."""
    expected = MODEL_CODE_COUNT + CALIBRATION_CODE_COUNT + COLD_CODE_COUNT
    if len(group_codes) != expected:
        raise ValueError(
            f"Expected exactly {expected:,} train codes, found {len(group_codes):,}. "
            "Update the fixed partition counts only after inspecting the new data."
        )
    order = np.asarray(
        sorted(
            range(len(group_codes)),
            key=lambda i: (stable_code_hash(group_codes[i]), str(group_codes[i])),
        ),
        dtype=np.int32,
    )
    roles = np.full(len(group_codes), -1, dtype=np.int8)
    roles[order[:MODEL_CODE_COUNT]] = 0
    roles[order[MODEL_CODE_COUNT:MODEL_CODE_COUNT + CALIBRATION_CODE_COUNT]] = 1
    roles[order[MODEL_CODE_COUNT + CALIBRATION_CODE_COUNT:]] = 2
    if not np.array_equal(np.bincount(roles, minlength=3),
                          [MODEL_CODE_COUNT, CALIBRATION_CODE_COUNT, COLD_CODE_COUNT]):
        raise AssertionError("Code-role partition size mismatch")
    return roles


def build_exact_cold_queries(days, group_ids, group_codes, roles, fold):
    """Return exact 1,500-stock, synchronous, deterministic queries."""
    priority_order = np.asarray(
        sorted(
            range(len(group_codes)),
            key=lambda i: (stable_code_hash(group_codes[i]), str(group_codes[i])),
        ),
        dtype=np.int32,
    )
    priority = np.empty(len(group_codes), dtype=np.int32)
    priority[priority_order] = np.arange(len(group_codes), dtype=np.int32)
    target_days = np.arange(
        fold["strategy_start"], fold["strategy_end"] + 1, HORIZON,
        dtype=np.int32,
    )
    queries = []
    availability = []
    sample_roles = roles[group_ids]
    for day in target_days:
        eligible = np.flatnonzero((sample_roles == 2) & (days == day)).astype(np.int32)
        availability.append(len(eligible))
        if len(eligible) < EXACT_COLD_QUERY_SIZE:
            raise RuntimeError(
                f"{fold['name']} DAY_{day:04d}: only {len(eligible):,} valid cold "
                f"stocks; need {EXACT_COLD_QUERY_SIZE:,}"
            )
        eligible = eligible[np.argsort(priority[group_ids[eligible]], kind="stable")]
        idx = eligible[:EXACT_COLD_QUERY_SIZE]
        if len(np.unique(group_ids[idx])) != EXACT_COLD_QUERY_SIZE:
            raise AssertionError("A cold query contains a duplicate stock")
        if not np.all(days[idx] == day):
            raise AssertionError("A cold query is not synchronous")
        queries.append(idx)
    if len(target_days) > 1 and not np.all(np.diff(target_days) >= HORIZON):
        raise AssertionError("Strategy target periods overlap")
    return target_days, queries, np.asarray(availability, dtype=np.int32)


def safe_auc(y_true, prediction) -> float:
    return float(roc_auc_score(y_true, prediction)) if np.unique(y_true).size == 2 else np.nan


def make_within_universe_relative_labels(ret20, days, mask) -> np.ndarray:
    """Top-20% labels using only the explicitly supplied code/time universe."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        raise ValueError("Cannot build relative labels for an empty universe")
    ranked = pd.Series(ret20[idx]).groupby(days[idx], sort=False).rank(
        method="average", pct=True
    )
    labels = np.full(len(ret20), 255, dtype=np.uint8)
    labels[idx] = (ranked.to_numpy() > RELATIVE_TAIL_QUANTILE).astype(np.uint8)
    return labels


def make_exact_query_relative_labels(ret20, queries) -> np.ndarray:
    labels = np.full(len(ret20), 255, dtype=np.uint8)
    for idx in queries:
        ranks = pd.Series(ret20[idx]).rank(method="average", pct=True).to_numpy()
        labels[idx] = (ranks > RELATIVE_TAIL_QUANTILE).astype(np.uint8)
    return labels


def score_official_query(y, paths, probability, codes):
    top = official_order(probability, codes)[:5]
    selected_p = probability[top]
    weights = (
        selected_p / selected_p.sum()
        if selected_p.sum() > 0 else np.full(5, 0.2, dtype=np.float64)
    )
    nav = 0.999 * (weights @ (1.0 + paths[top]))
    nav[-1] *= 0.999
    full_nav = np.r_[1.0, nav]
    peak = np.maximum.accumulate(full_nav)
    mdd = float(np.max(1.0 - full_nav / peak))
    total_return = float(nav[-1] - 1.0)
    brier = brier_score(y, probability)
    score = 0.6 * total_return - 0.2 * mdd - 0.2 * brier
    return score, total_return, mdd, brier, top


def summarize_rule(frame: pd.DataFrame, prefix: str) -> str:
    return (
        f"score={frame[prefix + '_score'].mean():+.6f}, "
        f"ret={frame[prefix + '_ret'].mean():+.3%}, "
        f"mdd={frame[prefix + '_mdd'].mean():.3%}, "
        f"brier={frame[prefix + '_brier'].mean():.6f}"
    )


def evaluate_rank_product_queries(
    queries, days, y, paths, group_ids, group_codes,
    raw_up, raw_absolute, raw_relative, alpha, raw_cal_center,
):
    rows = []
    for idx in queries:
        codes = group_codes[group_ids[idx]]
        p_up = np.clip(
            NEUTRAL_PROBABILITY_CENTER + alpha * (raw_up[idx] - raw_cal_center),
            *PROBABILITY_CLIP,
        )
        base_probability = np.clip(
            p_up + V32_GAMMA * (percentile_rank(raw_absolute[idx]) - 0.5),
            *PROBABILITY_CLIP,
        )
        product_score = (
            percentile_rank(base_probability) * percentile_rank(raw_relative[idx])
        )
        desired_top5 = official_order(product_score, codes)[:5]
        product_probability, original_top5, changed, brier_bound = (
            encode_top5_by_local_probability_permutation(
                base_probability, desired_top5, codes
            )
        )
        base = score_official_query(y[idx], paths[idx], base_probability, codes)
        product = score_official_query(y[idx], paths[idx], product_probability, codes)
        rows.append({
            "day": int(days[idx[0]]),
            "base_score": base[0], "base_ret": base[1],
            "base_mdd": base[2], "base_brier": base[3],
            "product_score": product[0], "product_ret": product[1],
            "product_mdd": product[2], "product_brier": product[3],
            "score_delta": product[0] - base[0],
            "ret_delta": product[1] - base[1],
            "top5_overlap": len(set(original_top5) & set(desired_top5)) / 5.0,
            "replacements": 5 - len(set(original_top5) & set(desired_top5)),
            "changed_probability_rows": len(changed),
            "worst_brier_increase_bound": brier_bound,
        })
    return pd.DataFrame(rows)


def main_rank_product_disabled():
    print("V5 exact-cold multi-time-fold rank-product validation")
    print(f"  train: {TRAIN_PATH}")
    print("  test data: NOT READ")
    print("  submission: NOT WRITTEN")
    print(
        f"  query={EXACT_COLD_QUERY_SIZE:,} cold stocks on one DAY; "
        f"strategy anchors separated by {HORIZON} days"
    )

    with timer("Read train and build leakage-safe full-panel targets/features"):
        train = read_market_csv(TRAIN_PATH)
        print(f"  train shape={train.shape}, codes={train['code'].nunique():,}")
        train = add_a1_cross_sectional_returns(train, "train")

    with timer("Build globally aligned 20-day-window matrix"):
        (
            x_up, x_a1, y, _unused_relative_y, days, paths, group_ids, group_codes
        ) = build_training_matrices(train, TRAIN_STRIDE)
        del train
        gc.collect()
        ret20 = paths[:, -1]
        del _unused_relative_y
        print(f"  base={x_up.shape}, A1.1={x_a1.shape}")
        print(
            f"  anchor DAY modulo {TRAIN_STRIDE}: "
            f"{np.unique((days - GLOBAL_ANCHOR_PHASE) % TRAIN_STRIDE).tolist()}"
        )

    roles = make_code_roles(group_codes)
    sample_roles = roles[group_ids]
    print("\nFixed code-disjoint universe")
    print(
        f"  model={np.sum(roles == 0):,}, calibration={np.sum(roles == 1):,}, "
        f"cold strategy pool={np.sum(roles == 2):,}"
    )

    all_results = []
    for fold in FOLDS:
        print("\n" + "=" * 78)
        print(
            f"{fold['name']}: train<=DAY_{fold['train_end']:04d}; "
            f"cal=DAY_{fold['cal_start']:04d}..DAY_{fold['cal_end']:04d}; "
            f"strategy=DAY_{fold['strategy_start']:04d}..DAY_{fold['strategy_end']:04d}"
        )
        model_train = (sample_roles == 0) & (days <= fold["train_end"])
        calibration = (
            (sample_roles == 1)
            & (days >= fold["cal_start"])
            & (days <= fold["cal_end"])
        )
        strategy_days, queries, availability = build_exact_cold_queries(
            days, group_ids, group_codes, roles, fold
        )
        strategy_idx = np.concatenate(queries)
        print(
            f"  samples: train={model_train.sum():,}, cal={calibration.sum():,}; "
            f"queries={len(queries)}, each={len(queries[0]):,}"
        )
        print(
            f"  cold availability/query before fixed sampling="
            f"{availability.min():,}..{availability.max():,}"
        )
        print(f"  strategy anchor days={strategy_days.tolist()}")

        tail_threshold = float(np.quantile(ret20[model_train], TAIL_QUANTILE))
        tail_y = (ret20 > tail_threshold).astype(np.uint8)
        relative_train_y = make_within_universe_relative_labels(
            ret20, days, model_train
        )
        relative_cal_y = make_within_universe_relative_labels(
            ret20, days, calibration
        )
        relative_strategy_y = make_exact_query_relative_labels(ret20, queries)
        print(
            f"  relative-label rates train/cal/strategy="
            f"{relative_train_y[model_train].mean():.4f}/"
            f"{relative_cal_y[calibration].mean():.4f}/"
            f"{relative_strategy_y[strategy_idx].mean():.4f}"
        )

        with timer(f"{fold['name']} fit up/absolute/relative models"):
            # Keep model seeds fixed across folds so fold differences reflect
            # time/universe changes rather than an avoidable seed confounder.
            up_model, backend = make_binary_model(RANDOM_STATE)
            absolute_model, absolute_backend = make_binary_model(
                RANDOM_STATE + 101
            )
            relative_model, relative_backend = make_binary_model(
                RANDOM_STATE + 211
            )
            if len({backend, absolute_backend, relative_backend}) != 1:
                raise RuntimeError("Model backends disagree")
            up_model = fit_binary(
                up_model, backend, x_up[model_train], y[model_train],
                x_up[calibration], y[calibration],
            )
            absolute_model = fit_binary(
                absolute_model, backend, x_up[model_train], tail_y[model_train],
                x_up[calibration], tail_y[calibration],
            )
            relative_model = fit_binary(
                relative_model, backend,
                x_a1[model_train], relative_train_y[model_train],
                x_a1[calibration], relative_cal_y[calibration],
            )

            raw_up_cal = up_model.predict_proba(x_up[calibration])[:, 1]
            alpha, cal_brier, raw_cal_center = choose_neutral_shrinkage(
                y[calibration], raw_up_cal
            )
            raw_up = np.full(len(y), np.nan, dtype=np.float32)
            raw_absolute = np.full(len(y), np.nan, dtype=np.float32)
            raw_relative = np.full(len(y), np.nan, dtype=np.float32)
            raw_up[strategy_idx] = up_model.predict_proba(x_up[strategy_idx])[:, 1]
            raw_absolute[strategy_idx] = absolute_model.predict_proba(
                x_up[strategy_idx]
            )[:, 1]
            raw_relative[strategy_idx] = relative_model.predict_proba(
                x_a1[strategy_idx]
            )[:, 1]

        same_auc_mean, same_auc_median, auc_days = mean_same_day_auc(
            queries, relative_strategy_y, raw_relative
        )
        print("  model diagnostics")
        print(
            f"    up AUC={safe_auc(y[strategy_idx], raw_up[strategy_idx]):.6f}, "
            f"absolute AUC={safe_auc(tail_y[strategy_idx], raw_absolute[strategy_idx]):.6f}"
        )
        print(
            f"    relative pooled AUC="
            f"{safe_auc(relative_strategy_y[strategy_idx], raw_relative[strategy_idx]):.6f}; "
            f"same-day mean/median={same_auc_mean:.6f}/{same_auc_median:.6f} "
            f"({auc_days} days)"
        )
        print(f"    probability alpha={alpha:.2f}, calibration Brier={cal_brier:.6f}")

        result = evaluate_rank_product_queries(
            queries, days, y, paths, group_ids, group_codes,
            raw_up, raw_absolute, raw_relative, alpha, raw_cal_center,
        )
        result.insert(0, "fold", fold["name"])
        all_results.append(result)
        print("  exact-query strategy results")
        print(f"    baseline     {summarize_rule(result, 'base')}")
        print(f"    rank-product {summarize_rule(result, 'product')}")
        print(
            f"    delta mean/median/q20="
            f"{result.score_delta.mean():+.6f}/"
            f"{result.score_delta.median():+.6f}/"
            f"{result.score_delta.quantile(.20):+.6f}; "
            f"win rate={(result.score_delta > 0).mean():.3f}"
        )
        print(
            f"    mean Top5 overlap={result.top5_overlap.mean():.3f}; "
            f"mean replacements={result.replacements.mean():.3f}; "
            f"max label-free Brier bound="
            f"{result.worst_brier_increase_bound.max():.8f}"
        )
        del up_model, absolute_model, relative_model
        del raw_up, raw_absolute, raw_relative, tail_y
        del relative_train_y, relative_cal_y, relative_strategy_y
        gc.collect()

    report = pd.concat(all_results, ignore_index=True)
    fold_report = report.groupby("fold", sort=False).agg(
        queries=("score_delta", "size"),
        mean_delta=("score_delta", "mean"),
        median_delta=("score_delta", "median"),
        q20_delta=("score_delta", lambda s: s.quantile(.20)),
        win_rate=("score_delta", lambda s: (s > 0).mean()),
        ret_delta=("ret_delta", "mean"),
        mean_overlap=("top5_overlap", "mean"),
    )
    fold_means = fold_report["mean_delta"]
    gate_all_fold_positive = bool((fold_means > 0).all())
    gate_no_bad_fold = bool((fold_means >= -0.005).all())
    gate_median = bool(report.score_delta.median() > 0)
    gate_win = bool((report.score_delta > 0).mean() >= 0.55)
    gate_q20 = bool(report.score_delta.quantile(.20) >= -0.020)

    print("\n" + "=" * 78)
    print("MULTI-FOLD EXACT-1500 REPORT")
    print(fold_report.to_string(float_format=lambda v: f"{v:+.6f}"))
    print("\nAggregate over all non-overlapping queries")
    print(f"  baseline     {summarize_rule(report, 'base')}")
    print(f"  rank-product {summarize_rule(report, 'product')}")
    print(
        f"  paired delta mean/median/q20/worst="
        f"{report.score_delta.mean():+.6f}/"
        f"{report.score_delta.median():+.6f}/"
        f"{report.score_delta.quantile(.20):+.6f}/"
        f"{report.score_delta.min():+.6f}"
    )
    print(f"  paired win rate={(report.score_delta > 0).mean():.3f}")
    print("\nPre-registered rank-product evidence gate (diagnostic only)")
    print(f"  positive mean in all 3 folds : {gate_all_fold_positive}")
    print(f"  no fold mean below -0.005    : {gate_no_bad_fold}")
    print(f"  aggregate median positive    : {gate_median}")
    print(f"  aggregate win rate >= 0.55   : {gate_win}")
    print(f"  paired q20 >= -0.020         : {gate_q20}")
    print(
        f"  overall decision             : "
        f"{'PASS FOR A SEPARATE SUBMISSION SCRIPT' if all([gate_all_fold_positive, gate_no_bad_fold, gate_median, gate_win, gate_q20]) else 'REJECT / INVESTIGATE; DO NOT SUBMIT'}"
    )
    print("\nDiagnostic complete. No test.csv was read and no submission was written.")


def same_day_spearman(queries, truth, prediction):
    values = []
    for idx in queries:
        true_rank = pd.Series(truth[idx]).rank(method="average", pct=True).to_numpy()
        pred_rank = pd.Series(prediction[idx]).rank(method="average", pct=True).to_numpy()
        if np.std(true_rank) > 0 and np.std(pred_rank) > 0:
            values.append(float(np.corrcoef(true_rank, pred_rank)[0, 1]))
    if not values:
        raise RuntimeError("No valid same-day Spearman correlations")
    a = np.asarray(values)
    return float(a.mean()), float(np.median(a)), float(np.quantile(a, 0.20))


def evaluate_tail_classifier_vs_regression(
    queries, days, y, paths, ret20, group_ids, group_codes,
    raw_up, raw_tail, raw_regression, alpha, raw_cal_center,
):
    rows, topk_rows = [], []
    for idx in queries:
        codes = group_codes[group_ids[idx]]
        p_up = np.clip(
            NEUTRAL_PROBABILITY_CENTER + alpha * (raw_up[idx] - raw_cal_center),
            *PROBABILITY_CLIP,
        )
        classifier_probability = np.clip(
            p_up + V32_GAMMA * (percentile_rank(raw_tail[idx]) - 0.5),
            *PROBABILITY_CLIP,
        )
        regression_probability = np.clip(
            p_up + V32_GAMMA * (percentile_rank(raw_regression[idx]) - 0.5),
            *PROBABILITY_CLIP,
        )
        classifier = score_official_query(
            y[idx], paths[idx], classifier_probability, codes
        )
        regression = score_official_query(
            y[idx], paths[idx], regression_probability, codes
        )
        overlap = len(set(classifier[4]) & set(regression[4]))
        rows.append({
            "day": int(days[idx[0]]),
            "classifier_score": classifier[0],
            "classifier_ret": classifier[1],
            "classifier_mdd": classifier[2],
            "classifier_brier": classifier[3],
            "regression_score": regression[0],
            "regression_ret": regression[1],
            "regression_mdd": regression[2],
            "regression_brier": regression[3],
            "score_delta": regression[0] - classifier[0],
            "ret_delta": regression[1] - classifier[1],
            "top5_overlap": overlap / 5.0,
            "replacements": 5 - overlap,
        })
        classifier_order = official_order(raw_tail[idx], codes)
        regression_order = official_order(raw_regression[idx], codes)
        for k in (5, 10, 20, 50):
            topk_rows.append({
                "day": int(days[idx[0]]),
                "K": k,
                "classifier_future_ret": float(ret20[idx[classifier_order[:k]]].mean()),
                "regression_future_ret": float(ret20[idx[regression_order[:k]]].mean()),
            })
    return pd.DataFrame(rows), pd.DataFrame(topk_rows)


def print_regression_deciles(queries, ret20, prediction):
    rank_blocks, return_blocks = [], []
    for idx in queries:
        rank_blocks.append(percentile_rank(prediction[idx]))
        return_blocks.append(ret20[idx])
    rank = np.concatenate(rank_blocks)
    future_return = np.concatenate(return_blocks)
    decile = np.minimum((rank * 10).astype(np.int8), 9)
    table = pd.DataFrame({"decile": decile, "future_return": future_return}).groupby(
        "decile", sort=True
    ).agg(n=("future_return", "size"),
          mean_return=("future_return", "mean"),
          median_return=("future_return", "median"))
    print(table.to_string(formatters={
        "mean_return": lambda v: f"{v:+.3%}",
        "median_return": lambda v: f"{v:+.3%}",
    }))
    means = table["mean_return"].to_numpy()
    print(
        f"    adjacent mean-return increases="
        f"{int(np.sum(np.diff(means) > 0))}/{max(len(means) - 1, 1)}; "
        f"top-bottom spread={means[-1] - means[0]:+.3%}"
    )


def main():
    print("V5-A exact-cold robust-return-regression ablation")
    print(f"  train: {TRAIN_PATH}")
    print("  test data: NOT READ")
    print("  submission: NOT WRITTEN")
    print(
        f"  baseline=q{TAIL_QUANTILE:.0%} absolute-tail classifier; "
        f"candidate=ret20 winsorized at training-only "
        f"q{REGRESSION_WINSOR_LOWER:.0%}/q{REGRESSION_WINSOR_UPPER:.0%} + Huber"
    )
    print(
        f"  both branches: same 59 V3 features, up model, gamma={V32_GAMMA}, "
        f"and exact {EXACT_COLD_QUERY_SIZE:,}-stock queries"
    )

    with timer("Read train CSV"):
        train = read_market_csv(TRAIN_PATH)
        print(f"  train shape={train.shape}, codes={train['code'].nunique():,}")

    with timer("Build globally aligned V3 20-day-window matrix"):
        (
            x_up, _unused_x, y, _unused_relative_y,
            days, paths, group_ids, group_codes,
        ) = build_training_matrices(train, TRAIN_STRIDE)
        del train, _unused_x, _unused_relative_y
        gc.collect()
        ret20 = paths[:, -1].astype(np.float32)
        print(f"  features={x_up.shape}, future paths={paths.shape}")
        print(
            f"  aligned anchor check (DAY-phase modulo {TRAIN_STRIDE})="
            f"{np.unique((days - GLOBAL_ANCHOR_PHASE) % TRAIN_STRIDE).tolist()}"
        )

    roles = make_code_roles(group_codes)
    sample_roles = roles[group_ids]
    print("\nFixed code-disjoint universe")
    print(
        f"  model={np.sum(roles == 0):,}, calibration={np.sum(roles == 1):,}, "
        f"cold strategy pool={np.sum(roles == 2):,}"
    )

    all_results, all_topk = [], []
    fold_ic = []
    for fold in FOLDS:
        print("\n" + "=" * 78)
        print(
            f"{fold['name']}: train<=DAY_{fold['train_end']:04d}; "
            f"cal=DAY_{fold['cal_start']:04d}..DAY_{fold['cal_end']:04d}; "
            f"strategy=DAY_{fold['strategy_start']:04d}..DAY_{fold['strategy_end']:04d}"
        )
        model_train = (sample_roles == 0) & (days <= fold["train_end"])
        calibration = (
            (sample_roles == 1)
            & (days >= fold["cal_start"])
            & (days <= fold["cal_end"])
        )
        strategy_days, queries, availability = build_exact_cold_queries(
            days, group_ids, group_codes, roles, fold
        )
        strategy_idx = np.concatenate(queries)
        print(
            f"  samples: train={model_train.sum():,}, cal={calibration.sum():,}; "
            f"queries={len(queries)}, each={len(queries[0]):,}"
        )
        print(
            f"  cold availability before sampling="
            f"{availability.min():,}..{availability.max():,}; "
            f"anchors={strategy_days.tolist()}"
        )

        tail_threshold = float(np.quantile(ret20[model_train], TAIL_QUANTILE))
        tail_y = (ret20 > tail_threshold).astype(np.uint8)
        winsor_low, winsor_high = np.quantile(
            ret20[model_train],
            [REGRESSION_WINSOR_LOWER, REGRESSION_WINSOR_UPPER],
        )
        reg_y = np.clip(ret20, winsor_low, winsor_high).astype(np.float32)
        print(
            f"  q80 tail threshold={tail_threshold:+.3%}; "
            f"train-only regression clip=[{winsor_low:+.3%}, {winsor_high:+.3%}]"
        )

        with timer(f"{fold['name']} fit unchanged up/tail and robust regressor"):
            up_model, backend = make_binary_model(RANDOM_STATE)
            tail_model, tail_backend = make_binary_model(RANDOM_STATE + 101)
            regression_model, regression_backend = make_robust_regression_model(
                RANDOM_STATE + 307
            )
            if len({backend, tail_backend, regression_backend}) != 1:
                raise RuntimeError("Model backends disagree")
            up_model = fit_binary(
                up_model, backend,
                x_up[model_train], y[model_train],
                x_up[calibration], y[calibration],
            )
            tail_model = fit_binary(
                tail_model, backend,
                x_up[model_train], tail_y[model_train],
                x_up[calibration], tail_y[calibration],
            )
            regression_model = fit_regression(
                regression_model, backend,
                x_up[model_train], reg_y[model_train],
                x_up[calibration], reg_y[calibration],
            )
            raw_up_cal = up_model.predict_proba(x_up[calibration])[:, 1]
            alpha, cal_brier, raw_cal_center = choose_neutral_shrinkage(
                y[calibration], raw_up_cal
            )
            raw_up = np.full(len(y), np.nan, dtype=np.float32)
            raw_tail = np.full(len(y), np.nan, dtype=np.float32)
            raw_regression = np.full(len(y), np.nan, dtype=np.float32)
            raw_up[strategy_idx] = up_model.predict_proba(x_up[strategy_idx])[:, 1]
            raw_tail[strategy_idx] = tail_model.predict_proba(
                x_up[strategy_idx]
            )[:, 1]
            raw_regression[strategy_idx] = regression_model.predict(
                x_up[strategy_idx]
            )

        prediction = raw_regression[strategy_idx].astype(np.float64)
        truth = ret20[strategy_idx].astype(np.float64)
        clipped_truth = reg_y[strategy_idx].astype(np.float64)
        raw_rmse = float(np.sqrt(np.mean((prediction - truth) ** 2)))
        raw_mae = float(np.mean(np.abs(prediction - truth)))
        clipped_mae = float(np.mean(np.abs(prediction - clipped_truth)))
        ic_mean, ic_median, ic_q20 = same_day_spearman(
            queries, ret20, raw_regression
        )
        fold_ic.append(ic_mean)
        print("  model diagnostics")
        print(
            f"    up AUC={safe_auc(y[strategy_idx], raw_up[strategy_idx]):.6f}; "
            f"tail AUC={safe_auc(tail_y[strategy_idx], raw_tail[strategy_idx]):.6f}"
        )
        print(
            f"    regression raw RMSE/MAE={raw_rmse:.3%}/{raw_mae:.3%}; "
            f"winsorized MAE={clipped_mae:.3%}"
        )
        print(
            f"    same-day regression Spearman mean/median/q20="
            f"{ic_mean:+.5f}/{ic_median:+.5f}/{ic_q20:+.5f}"
        )
        print(f"    probability alpha={alpha:.2f}, calibration Brier={cal_brier:.6f}")

        result, topk = evaluate_tail_classifier_vs_regression(
            queries, days, y, paths, ret20, group_ids, group_codes,
            raw_up, raw_tail, raw_regression, alpha, raw_cal_center,
        )
        result.insert(0, "fold", fold["name"])
        topk.insert(0, "fold", fold["name"])
        all_results.append(result)
        all_topk.append(topk)
        print("  exact-query strategy results")
        print(f"    classifier {summarize_rule(result, 'classifier')}")
        print(f"    regression {summarize_rule(result, 'regression')}")
        print(
            f"    delta mean/median/q20="
            f"{result.score_delta.mean():+.6f}/"
            f"{result.score_delta.median():+.6f}/"
            f"{result.score_delta.quantile(.20):+.6f}; "
            f"win rate={(result.score_delta > 0).mean():.3f}"
        )
        print(
            f"    mean Top5 overlap={result.top5_overlap.mean():.3f}; "
            f"mean replacements={result.replacements.mean():.3f}"
        )
        topk_fold = topk.groupby("K", sort=True).agg(
            classifier_ret=("classifier_future_ret", "mean"),
            regression_ret=("regression_future_ret", "mean"),
        )
        topk_fold["delta"] = topk_fold.regression_ret - topk_fold.classifier_ret
        print("  unweighted future return by raw model TopK")
        print(topk_fold.to_string(float_format=lambda v: f"{v:+.3%}"))
        print("  regression prediction-decile monotonicity")
        print_regression_deciles(queries, ret20, raw_regression)

        del up_model, tail_model, regression_model
        del raw_up, raw_tail, raw_regression, tail_y, reg_y
        gc.collect()

    report = pd.concat(all_results, ignore_index=True)
    topk_report = pd.concat(all_topk, ignore_index=True)
    fold_report = report.groupby("fold", sort=False).agg(
        queries=("score_delta", "size"),
        mean_delta=("score_delta", "mean"),
        median_delta=("score_delta", "median"),
        q20_delta=("score_delta", lambda s: s.quantile(.20)),
        win_rate=("score_delta", lambda s: (s > 0).mean()),
        ret_delta=("ret_delta", "mean"),
        mean_overlap=("top5_overlap", "mean"),
    )
    aggregate_topk = topk_report.groupby("K", sort=True).agg(
        classifier_ret=("classifier_future_ret", "mean"),
        regression_ret=("regression_future_ret", "mean"),
    )
    aggregate_topk["delta"] = (
        aggregate_topk.regression_ret - aggregate_topk.classifier_ret
    )

    fold_means = fold_report.mean_delta
    gate_all_fold_positive = bool((fold_means > 0).all())
    gate_no_bad_fold = bool((fold_means >= -0.005).all())
    gate_median = bool(report.score_delta.median() > 0)
    gate_win = bool((report.score_delta > 0).mean() >= 0.55)
    gate_q20 = bool(report.score_delta.quantile(.20) >= -0.020)
    gate_ic = bool(np.all(np.asarray(fold_ic) > 0))

    print("\n" + "=" * 78)
    print("V5-A ROBUST REGRESSION MULTI-FOLD REPORT")
    print(fold_report.to_string(float_format=lambda v: f"{v:+.6f}"))
    print("\nAggregate over all 36 non-overlapping exact-1500 queries")
    print(f"  classifier {summarize_rule(report, 'classifier')}")
    print(f"  regression {summarize_rule(report, 'regression')}")
    print(
        f"  paired delta mean/median/q20/worst="
        f"{report.score_delta.mean():+.6f}/"
        f"{report.score_delta.median():+.6f}/"
        f"{report.score_delta.quantile(.20):+.6f}/"
        f"{report.score_delta.min():+.6f}"
    )
    print(f"  paired win rate={(report.score_delta > 0).mean():.3f}")
    print("\nAggregate raw-model TopK future returns")
    print(aggregate_topk.to_string(float_format=lambda v: f"{v:+.3%}"))
    print("\nPre-registered robust-regression evidence gate")
    print(f"  positive mean in all 3 folds : {gate_all_fold_positive}")
    print(f"  no fold mean below -0.005    : {gate_no_bad_fold}")
    print(f"  aggregate median positive    : {gate_median}")
    print(f"  aggregate win rate >= 0.55   : {gate_win}")
    print(f"  paired q20 >= -0.020         : {gate_q20}")
    print(f"  same-day IC positive all folds: {gate_ic}")
    passed = all([
        gate_all_fold_positive, gate_no_bad_fold, gate_median,
        gate_win, gate_q20, gate_ic,
    ])
    print(
        f"  overall decision             : "
        f"{'PASS FOR A SEPARATE SUBMISSION SCRIPT' if passed else 'REJECT / INVESTIGATE; DO NOT SUBMIT'}"
    )
    print("\nDiagnostic complete. No test.csv was read and no submission was written.")


if __name__ == "__main__":
    main()
