"""
V3.2-A1: clean tail-only ablation of cross-sectional return features.

The unchanged V3 probability branch uses the original 59 scale-free features.
The candidate tail branch adds only same-anchor-day return percentile ranks,
median excess returns, and return z-scores for 1/5/10/19-day horizons.

Kaggle inputs:
  /kaggle/input/datasets/zhuowamg/test-data/train.csv
  /kaggle/input/datasets/zhuowamg/true-test-data/test.csv

Output:
  /kaggle/working/submission_v3_2_a1.csv
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
OUTPUT_PATH = Path("/kaggle/working/submission_v3_2_a1.csv")

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

# Same conservative gate as V3.2. A1 must earn its place; otherwise the script
# writes the reconstructed V3 baseline automatically.
A1_MIN_TAIL_AUC_GAIN = 0.003
A1_MAX_Q20_SCORE_DROP = 0.005
A1_MAX_BRIER_CHANGE = 0.001
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
    ys, anchor_days, paths20, group_ids, group_codes = [], [], [], [], []
    grouped = df.groupby("code", observed=True, sort=False)

    for group_no, (code, g) in enumerate(grouped, start=0):
        g = g.sort_values("day")
        days = g["day"].to_numpy(np.int32)
        values = g[[*PRICE_COLUMNS, "volume"]].to_numpy(np.float32)
        panel_a1 = g[list(A1_FEATURE_COLUMNS)].to_numpy(np.float32)
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
    }


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
    print("V3.2-A1 cross-sectional return ablation")
    print(f"  train:  {TRAIN_PATH}")
    print(f"  test:   {TEST_PATH}")
    print(f"  output: {OUTPUT_PATH}")
    print(f"  A1 features: {len(A1_FEATURE_COLUMNS)}; tail-only; fixed gamma={V32_GAMMA}")

    with timer("Read and validate CSV files"):
        train = read_market_csv(TRAIN_PATH)
        test = read_market_csv(TEST_PATH)
        print(f"  train shape={train.shape}, codes={train['code'].nunique():,}")
        print(f"  test  shape={test.shape}, codes={test['code'].nunique():,}")

    with timer("Build leakage-safe full-panel A1 return features"):
        train = add_a1_cross_sectional_returns(train, "train")
        test = add_a1_cross_sectional_returns(test, "test")

    with timer("Build unchanged V3 matrices and A1 tail candidate"):
        x_up, x_a1, y, days, paths, group_ids, group_codes = build_training_matrices(
            train, TRAIN_STRIDE
        )
        x_up_test, x_a1_test, test_codes = build_test_matrices(test)
        ret20 = paths[:, -1]
        print(f"  baseline={x_up.shape}, A1 tail={x_a1.shape}, test={x_up_test.shape}")
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
    print(f"  tail q{TAIL_QUANTILE:.0%} threshold={tail_threshold:.4%}")

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

    with timer("Fit baseline and A1 tail models for clean ablation"):
        baseline_tail, baseline_backend = make_binary_model(RANDOM_STATE + 101)
        a1_tail, a1_backend = make_binary_model(RANDOM_STATE + 101)
        if baseline_backend != backend or a1_backend != backend:
            raise RuntimeError("Model backends disagree")
        baseline_tail = fit_binary(
            baseline_tail, backend,
            x_up[model_train], tail_y[model_train],
            x_up[calibration], tail_y[calibration],
        )
        a1_tail = fit_binary(
            a1_tail, backend,
            x_a1[model_train], tail_y[model_train],
            x_a1[calibration], tail_y[calibration],
        )
        baseline_rounds = best_round_count(baseline_tail, backend)
        a1_rounds = best_round_count(a1_tail, backend)
        baseline_pred = baseline_tail.predict_proba(x_up[strategy])[:, 1]
        a1_pred = a1_tail.predict_proba(x_a1[strategy])[:, 1]

    baseline_auc = roc_auc_score(tail_y[strategy], baseline_pred)
    a1_auc = roc_auc_score(tail_y[strategy], a1_pred)
    print("\nA1 tail feature ablation")
    print(f"  baseline tail AUC={baseline_auc:.6f}, rounds={baseline_rounds}")
    print(f"  A1 tail AUC      ={a1_auc:.6f}, rounds={a1_rounds}")
    print(f"  AUC gain         ={a1_auc - baseline_auc:+.6f}")

    strategy_groups = group_ids[strategy]
    strategy_days = days[strategy]
    independent_batches = build_independent_window_batches(
        strategy_groups, strategy_days, PSEUDO_BATCH_REPEATS, RANDOM_STATE
    )
    same_day_batches = build_same_day_batches(strategy_days)

    independent_base = evaluate_fixed_tail_model(
        independent_batches, y[strategy], paths[strategy], p_up_strategy, baseline_pred
    )
    independent_a1 = evaluate_fixed_tail_model(
        independent_batches, y[strategy], paths[strategy], p_up_strategy, a1_pred
    )
    same_day_base = evaluate_fixed_tail_model(
        same_day_batches, y[strategy], paths[strategy], p_up_strategy, baseline_pred
    )
    same_day_a1 = evaluate_fixed_tail_model(
        same_day_batches, y[strategy], paths[strategy], p_up_strategy, a1_pred
    )

    print("\nExisting V3.2 independent-window diagnostic")
    print_metrics("baseline", independent_base)
    print_metrics("A1", independent_a1)
    print("\nA1-specific coherent same-day cross-section diagnostic")
    print_metrics("baseline", same_day_base)
    print_metrics("A1", same_day_a1)

    auc_pass = a1_auc - baseline_auc >= A1_MIN_TAIL_AUC_GAIN
    independent_mean_pass = independent_a1["score_mean"] > independent_base["score_mean"]
    same_day_mean_pass = same_day_a1["score_mean"] > same_day_base["score_mean"]
    independent_q20_pass = (
        independent_a1["score_q20"] >= independent_base["score_q20"] - A1_MAX_Q20_SCORE_DROP
    )
    same_day_q20_pass = (
        same_day_a1["score_q20"] >= same_day_base["score_q20"] - A1_MAX_Q20_SCORE_DROP
    )
    brier_pass = abs(
        independent_a1["brier_mean"] - independent_base["brier_mean"]
    ) <= A1_MAX_BRIER_CHANGE
    use_a1 = all([
        auc_pass,
        independent_mean_pass,
        same_day_mean_pass,
        independent_q20_pass,
        same_day_q20_pass,
        brier_pass,
    ])

    print("\nPre-registered A1 acceptance gate")
    print(f"  AUC gain >= {A1_MIN_TAIL_AUC_GAIN:.3f}              : {auc_pass}")
    print(f"  independent mean improves        : {independent_mean_pass}")
    print(f"  coherent-day mean improves       : {same_day_mean_pass}")
    print(f"  independent q20 protected        : {independent_q20_pass}")
    print(f"  coherent-day q20 protected       : {same_day_q20_pass}")
    print(f"  Brier change <= {A1_MAX_BRIER_CHANGE:.3f}           : {brier_pass}")
    print(f"  selected tail branch             : {'A1' if use_a1 else 'V3 baseline fallback'}")

    selected_rounds = a1_rounds if use_a1 else baseline_rounds
    selected_x = x_a1 if use_a1 else x_up
    selected_x_test = x_a1_test if use_a1 else x_up_test
    del baseline_tail, a1_tail, baseline_pred, a1_pred
    gc.collect()

    max_day = max(int(days.max()), 1)
    sample_weight = (0.60 + 0.40 * (days.astype(np.float32) / max_day) ** 2).astype(np.float32)

    with timer("Refit final unchanged up model"):
        final_up, final_backend = make_binary_model(RANDOM_STATE, up_rounds)
        if final_backend == "sklearn_hist_gbdt":
            final_up.set_params(early_stopping=False)
        final_up = fit_binary(final_up, final_backend, x_up, y, sample_weight=sample_weight)
        raw_up_test = final_up.predict_proba(x_up_test)[:, 1]

    with timer("Refit final selected tail model"):
        final_tail, final_tail_backend = make_binary_model(RANDOM_STATE + 101, selected_rounds)
        if final_tail_backend == "sklearn_hist_gbdt":
            final_tail.set_params(early_stopping=False)
        final_tail = fit_binary(
            final_tail, final_tail_backend, selected_x, tail_y, sample_weight=sample_weight
        )
        raw_tail_test = final_tail.predict_proba(selected_x_test)[:, 1]

    p_up_test = np.clip(
        NEUTRAL_PROBABILITY_CENTER
        + alpha * (raw_up_test - float(raw_up_test.mean())),
        *PROBABILITY_CLIP,
    )
    tail_rank_test = percentile_rank(raw_tail_test)
    test_probability = np.clip(
        p_up_test + V32_GAMMA * (tail_rank_test - 0.5), *PROBABILITY_CLIP
    )
    submission = pd.DataFrame({
        "code": test_codes,
        "up_factor": test_probability.astype(np.float64),
    })
    validate_submission(submission, test_codes)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUTPUT_PATH, index=False, encoding="utf-8", float_format="%.8f")
    check = pd.read_csv(OUTPUT_PATH)
    validate_submission(check, test_codes)

    print("\nV3.2-A1 submission created")
    print(f"  path={OUTPUT_PATH}")
    print(f"  selected branch={'A1' if use_a1 else 'V3 baseline fallback'}")
    print(f"  rows={len(check):,}")
    print(f"  probability range=[{check.up_factor.min():.6f}, {check.up_factor.max():.6f}]")
    print(
        check.sort_values(["up_factor", "code"], ascending=[False, True])
        .head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
