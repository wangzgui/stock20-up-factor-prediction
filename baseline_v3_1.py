"""
V3.1 robust tail ensemble with rolling code-and-time validation.

Kaggle inputs expected by default:
  /kaggle/input/datasets/zhuowamg/test-data/train.csv
  /kaggle/input/datasets/zhuowamg/true-test-data/test.csv

Output:
  /kaggle/working/submission_v3_1.csv

The latest test data contains independently anonymized 20-day windows.  This
script therefore never joins train and test by code and uses only scale-free
features.  The number of submission rows is inferred from test.csv instead of
being hard-coded to an older competition version.
"""

from __future__ import annotations

import gc
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Configuration: edit only this section for a first run.
# ---------------------------------------------------------------------------
TRAIN_PATH = Path("/kaggle/input/datasets/zhuowamg/test-data/train.csv")
TEST_PATH = Path("/kaggle/input/datasets/zhuowamg/true-test-data/test.csv")
OUTPUT_PATH = Path("/kaggle/working/submission_v3_1.csv")

WINDOW = 20
HORIZON = 20
TRAIN_STRIDE = 10       # 10 ~= 0.79M samples; use 5 for a slower stronger run.
VALID_START_DAY = 2401  # newest regime used for validation/calibration.
PURGE_DAYS = 40         # prevents overlapping train/validation information.
RANDOM_STATE = 20260822
PROBABILITY_CLIP = (0.02, 0.98)

PRICE_COLUMNS = ["open", "high", "low", "close"]
RAW_COLUMNS = ["code", "date", *PRICE_COLUMNS, "volume"]
FLOAT_DTYPES = {c: "float32" for c in [*PRICE_COLUMNS, "volume"]}


def timer(message: str):
    """Small context manager for readable Kaggle logs."""
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
    if list(df.columns) != RAW_COLUMNS:
        # read_csv/usecols may preserve source order; this also catches bad schemas.
        missing = sorted(set(RAW_COLUMNS) - set(df.columns))
        if missing:
            raise ValueError(f"Missing required columns in {path.name}: {missing}")
        df = df[RAW_COLUMNS]

    day_text = df["date"].str.extract(r"DAY_(\d+)", expand=False)
    if day_text.isna().any():
        examples = df.loc[day_text.isna(), "date"].head().tolist()
        raise ValueError(f"Invalid date format; examples: {examples}")
    df["day"] = day_text.astype("int16")
    df.drop(columns="date", inplace=True)

    numeric = [*PRICE_COLUMNS, "volume"]
    if df[numeric].isna().any().any():
        raise ValueError(f"NaN found in numeric columns of {path.name}")
    if (df[PRICE_COLUMNS] <= 0).any().any():
        raise ValueError(f"Non-positive OHLC price found in {path.name}")
    if (df["volume"] < 0).any():
        raise ValueError(f"Negative volume found in {path.name}")
    return df


def _safe_std(x: np.ndarray, axis: int = 1) -> np.ndarray:
    return np.std(x, axis=axis, dtype=np.float64).astype(np.float32)


def make_window_features(values: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Vectorized scale-free features for 20 observations ending at anchors."""
    offsets = np.arange(-(WINDOW - 1), 1, dtype=np.int32)
    idx = anchors[:, None] + offsets[None, :]
    w = values[idx].astype(np.float64, copy=False)

    op, hi, lo, cl, vol = (w[:, :, i] for i in range(5))
    eps = 1e-12

    log_close = np.log(np.maximum(cl, eps))
    ret = np.diff(log_close, axis=1)                       # 19 daily returns
    log_vol = np.log(np.maximum(vol, eps))
    vol_change = np.diff(log_vol, axis=1)

    candle_range = (hi - lo) / np.maximum(cl, eps)
    candle_body = (cl - op) / np.maximum(op, eps)
    close_location = (cl - lo) / np.maximum(hi - lo, eps)
    gap = op[:, 1:] / np.maximum(cl[:, :-1], eps) - 1.0

    # True-range-like quantity, expressed as a ratio to previous close.
    true_range = np.maximum.reduce([
        hi[:, 1:] - lo[:, 1:],
        np.abs(hi[:, 1:] - cl[:, :-1]),
        np.abs(lo[:, 1:] - cl[:, :-1]),
    ]) / np.maximum(cl[:, :-1], eps)

    blocks = [ret.astype(np.float32)]  # retain the complete recent return shape

    # Multi-horizon returns and reversal/trend information.
    for h in (1, 2, 3, 5, 10, 19):
        blocks.append((cl[:, -1] / np.maximum(cl[:, -1 - h], eps) - 1.0)[:, None])

    # Distribution summaries at several horizons.
    for h in (5, 10, 19):
        rr = ret[:, -h:]
        mean = rr.mean(axis=1)
        std = _safe_std(rr)
        downside = np.sqrt(np.mean(np.minimum(rr, 0.0) ** 2, axis=1))
        blocks.extend([mean[:, None], std[:, None], downside[:, None]])

    centered = ret - ret.mean(axis=1, keepdims=True)
    ret_std = np.maximum(_safe_std(ret), eps)
    skew = np.mean(centered ** 3, axis=1) / (ret_std ** 3)
    blocks.append(skew[:, None])

    # Linear trend of log-price; x is centered so slope is scale-free.
    x = np.arange(WINDOW, dtype=np.float64)
    x -= x.mean()
    slope = (log_close @ x) / np.sum(x * x)
    fitted = log_close.mean(axis=1, keepdims=True) + slope[:, None] * x[None, :]
    resid_std = _safe_std(log_close - fitted)
    blocks.extend([slope[:, None], resid_std[:, None]])

    running_max = np.maximum.accumulate(cl, axis=1)
    max_drawdown = np.min(cl / np.maximum(running_max, eps) - 1.0, axis=1)
    blocks.append(max_drawdown[:, None])

    # Candle and gap summaries.
    for z in (candle_range, candle_body, close_location, gap, true_range):
        blocks.extend([
            z.mean(axis=1)[:, None],
            _safe_std(z)[:, None],
            z[:, -1][:, None],
        ])

    # Volume features remain invariant to an arbitrary per-window scale factor.
    volume_median = np.maximum(np.median(vol, axis=1), eps)
    relative_volume = vol / volume_median[:, None]
    blocks.extend([
        np.mean(vol_change, axis=1)[:, None],
        _safe_std(vol_change)[:, None],
        relative_volume[:, -1][:, None],
        relative_volume[:, -5:].mean(axis=1)[:, None],
        (vol[:, -5:].mean(axis=1) / np.maximum(vol.mean(axis=1), eps))[:, None],
    ])

    # Correlation between return and contemporaneous volume change.
    r0 = ret - ret.mean(axis=1, keepdims=True)
    v0 = vol_change - vol_change.mean(axis=1, keepdims=True)
    corr = np.mean(r0 * v0, axis=1) / np.maximum(_safe_std(ret) * _safe_std(vol_change), eps)
    blocks.append(corr[:, None])

    result = np.concatenate([np.asarray(b, dtype=np.float32) for b in blocks], axis=1)
    return np.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0)


def contiguous_segments(days: np.ndarray):
    """Yield [start, end) slices whose DAY numbers increase by exactly one."""
    cuts = np.flatnonzero(np.diff(days) != 1) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, len(days)]
    for start, end in zip(starts, ends):
        yield int(start), int(end)


def build_training_matrix(df: pd.DataFrame, stride: int):
    xs, ys, anchor_days, future_returns = [], [], [], []
    grouped = df.groupby("code", observed=True, sort=False)

    for group_no, (_, g) in enumerate(grouped, start=1):
        g = g.sort_values("day")
        days = g["day"].to_numpy(np.int32)
        values = g[[*PRICE_COLUMNS, "volume"]].to_numpy(np.float32)

        for start, end in contiguous_segments(days):
            n = end - start
            if n < WINDOW + HORIZON:
                continue
            # Local anchor positions: 19 .. n-21, inclusive.
            local = np.arange(WINDOW - 1, n - HORIZON, stride, dtype=np.int32)
            absolute = start + local
            xs.append(make_window_features(values, absolute))

            current = values[absolute, 3]
            future = values[absolute + HORIZON, 3]
            ret20 = future / current - 1.0
            ys.append((ret20 > 0).astype(np.uint8))
            future_returns.append(ret20.astype(np.float32))
            anchor_days.append(days[absolute].astype(np.int16))

        if group_no % 500 == 0:
            print(f"  processed {group_no:,}/{grouped.ngroups:,} stocks", flush=True)

    if not xs:
        raise RuntimeError("No valid continuous 20+20 training windows were found")

    return (
        np.vstack(xs),
        np.concatenate(ys),
        np.concatenate(anchor_days),
        np.concatenate(future_returns),
    )


def build_test_matrix(df: pd.DataFrame):
    xs, codes = [], []
    grouped = df.groupby("code", observed=True, sort=False)

    for code, g in grouped:
        g = g.sort_values("day")
        days = g["day"].to_numpy(np.int32)
        if len(g) < WINDOW:
            raise ValueError(f"{code} has only {len(g)} test rows; need at least {WINDOW}")
        if not np.all(np.diff(days[-WINDOW:]) == 1):
            raise ValueError(f"{code} does not have a continuous final {WINDOW}-day window")
        values = g[[*PRICE_COLUMNS, "volume"]].to_numpy(np.float32)[-WINDOW:]
        xs.append(make_window_features(values, np.array([WINDOW - 1], dtype=np.int32)))
        codes.append(str(code))

    return np.vstack(xs), np.asarray(codes, dtype=object)


def brier_score(y_true: np.ndarray, prob: np.ndarray) -> float:
    return float(np.mean((prob - y_true) ** 2))


def choose_probability_shrinkage(y_true, raw_prob):
    """Center and shrink probabilities using the newest validation regime."""
    target_center = float(np.mean(y_true))
    raw_center = float(np.mean(raw_prob))
    best_alpha, best_brier = 1.0, np.inf
    for alpha in np.linspace(0.0, 1.20, 61):
        p = np.clip(target_center + alpha * (raw_prob - raw_center), *PROBABILITY_CLIP)
        score = brier_score(y_true, p)
        if score < best_brier:
            best_alpha, best_brier = float(alpha), score
    return best_alpha, best_brier, target_center, raw_center


def top5_validation_summary(days, ret20, prob):
    """Diagnostic only: mean one-period top-5 return across validation anchor days."""
    records = []
    for day in np.unique(days):
        mask = days == day
        if mask.sum() < 5:
            continue
        p = prob[mask]
        r = ret20[mask]
        top = np.argsort(-p, kind="stable")[:5]
        denom = p[top].sum()
        weights = p[top] / denom if denom > 0 else np.full(5, 0.2)
        gross = float(np.dot(weights, r[top]))
        after_cost = (1.0 - 0.001) * (1.0 + gross) * (1.0 - 0.001) - 1.0
        records.append(after_cost)
    if not records:
        return float("nan"), float("nan"), 0
    a = np.asarray(records)
    return float(a.mean()), float(np.median(a)), len(a)


def make_model(n_estimators: int | None = None):
    """Prefer LightGBM on Kaggle, with a no-install sklearn fallback."""
    try:
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            objective="binary",
            n_estimators=n_estimators or 700,
            learning_rate=0.035,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=200,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.2,
            reg_lambda=2.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        )
        return model, "lightgbm"
    except ImportError:
        model = HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=n_estimators or 300,
            max_leaf_nodes=31,
            min_samples_leaf=200,
            l2_regularization=2.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
            random_state=RANDOM_STATE,
        )
        return model, "sklearn_hist_gbdt"


def fit_validation_model(model, backend, x_train, y_train, x_valid, y_valid):
    if backend == "lightgbm":
        from lightgbm import early_stopping, log_evaluation

        model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric="binary_logloss",
            callbacks=[early_stopping(60, verbose=True), log_evaluation(50)],
        )
    else:
        model.fit(x_train, y_train)
    return model


def validate_submission(submission: pd.DataFrame, expected_codes: np.ndarray):
    if list(submission.columns) != ["code", "up_factor"]:
        raise AssertionError("Submission columns/order must be code,up_factor")
    if len(submission) != len(expected_codes):
        raise AssertionError("Submission row count does not equal test unique-code count")
    if submission["code"].duplicated().any():
        raise AssertionError("Duplicate code found in submission")
    if set(submission["code"]) != set(expected_codes):
        raise AssertionError("Submission codes do not exactly match test codes")
    p = submission["up_factor"].to_numpy()
    if not np.isfinite(p).all() or not np.all((p >= 0.0) & (p <= 1.0)):
        raise AssertionError("up_factor contains NaN/inf/out-of-range values")


def main():
    print("Configuration")
    print(f"  train:  {TRAIN_PATH}")
    print(f"  test:   {TEST_PATH}")
    print(f"  output: {OUTPUT_PATH}")

    with timer("Read and validate CSV files"):
        train = read_market_csv(TRAIN_PATH)
        test = read_market_csv(TEST_PATH)
        print(f"  train shape={train.shape}, codes={train['code'].nunique():,}")
        print(f"  test  shape={test.shape}, codes={test['code'].nunique():,}")

    with timer("Build scale-free train/test features"):
        x, y, days, ret20 = build_training_matrix(train, TRAIN_STRIDE)
        x_test, test_codes = build_test_matrix(test)
        print(f"  train matrix={x.shape}, test matrix={x_test.shape}")
        print(f"  label up-rate={y.mean():.4f}")
        del train, test
        gc.collect()

    # Purged temporal split.  No sample whose target overlaps validation enters train.
    train_mask = days < (VALID_START_DAY - PURGE_DAYS)
    valid_mask = days >= VALID_START_DAY
    if train_mask.sum() == 0 or valid_mask.sum() == 0:
        raise RuntimeError("Temporal split is empty; adjust VALID_START_DAY for this dataset")
    print(f"\nTemporal split: train={train_mask.sum():,}, valid={valid_mask.sum():,}")
    print(f"Train up-rate={y[train_mask].mean():.4f}, valid up-rate={y[valid_mask].mean():.4f}")

    with timer("Fit validation model"):
        model, backend = make_model()
        print(f"  backend={backend}")
        model = fit_validation_model(
            model, backend,
            x[train_mask], y[train_mask],
            x[valid_mask], y[valid_mask],
        )
        raw_valid = model.predict_proba(x[valid_mask])[:, 1]

    alpha, calibrated_brier, valid_center, raw_valid_center = choose_probability_shrinkage(
        y[valid_mask], raw_valid
    )
    calibrated_valid = np.clip(
        valid_center + alpha * (raw_valid - raw_valid_center), *PROBABILITY_CLIP
    )
    auc = roc_auc_score(y[valid_mask], raw_valid)
    raw_brier = brier_score(y[valid_mask], raw_valid)
    valid_logloss = log_loss(y[valid_mask], calibrated_valid)
    top_mean, top_median, top_periods = top5_validation_summary(
        days[valid_mask], ret20[valid_mask], calibrated_valid
    )
    print("\nValidation diagnostics")
    print(f"  AUC                 : {auc:.6f}")
    print(f"  raw Brier           : {raw_brier:.6f}")
    print(f"  calibrated Brier    : {calibrated_brier:.6f}")
    print(f"  calibrated logloss  : {valid_logloss:.6f}")
    print(f"  shrink alpha        : {alpha:.2f}")
    print(f"  top5 mean return    : {top_mean:.4%} ({top_periods} anchor days)")
    print(f"  top5 median return  : {top_median:.4%}")

    if backend == "lightgbm":
        best_rounds = int(getattr(model, "best_iteration_", 0) or model.n_estimators)
    else:
        best_rounds = int(getattr(model, "n_iter_", 300))

    # Refit on all valid labeled windows.  Mild recency weighting addresses regime drift.
    max_day = max(int(days.max()), 1)
    sample_weight = (0.60 + 0.40 * (days.astype(np.float32) / max_day) ** 2).astype(np.float32)
    with timer(f"Refit final {backend} model on all samples ({best_rounds} rounds)"):
        final_model, final_backend = make_model(best_rounds)
        if final_backend == "sklearn_hist_gbdt":
            # Disable its internal split so requested rounds are used on all data.
            final_model.set_params(early_stopping=False)
        final_model.fit(x, y, sample_weight=sample_weight)
        raw_test = final_model.predict_proba(x_test)[:, 1]

    # Preserve the recent-regime base rate while retaining validation-selected spread.
    recent_center = float(y[days >= VALID_START_DAY].mean())
    test_prob = np.clip(
        recent_center + alpha * (raw_test - float(raw_test.mean())),
        *PROBABILITY_CLIP,
    )

    submission = pd.DataFrame({
        "code": test_codes,
        "up_factor": test_prob.astype(np.float64),
    })
    validate_submission(submission, test_codes)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUTPUT_PATH, index=False, encoding="utf-8", float_format="%.8f")

    # Read back from disk: validates the actual artifact, not only the in-memory frame.
    check = pd.read_csv(OUTPUT_PATH)
    validate_submission(check, test_codes)
    print("\nSubmission created successfully")
    print(f"  path={OUTPUT_PATH}")
    print(f"  rows={len(check):,}")
    print(f"  probability range=[{check.up_factor.min():.6f}, {check.up_factor.max():.6f}]")
    print(check.sort_values("up_factor", ascending=False).head(10).to_string(index=False))




# ---------------------------------------------------------------------------
# V3 standalone extension.
# ---------------------------------------------------------------------------
import zlib

HOLDOUT_CAL_BUCKET_START = 60
HOLDOUT_STRATEGY_BUCKET_START = 78
TAIL_QUANTILE = 0.80
PSEUDO_BATCH_REPEATS = 30
TAIL_GAMMAS = (0.0, 0.005, 0.010, 0.020, 0.030, 0.040)
ROBUSTNESS_PENALTY = 0.15
NEUTRAL_PROBABILITY_CENTER = 0.50
MODEL_TRAIN_END_DAY = 2200
CALIBRATION_START_DAY = 2241
CALIBRATION_END_DAY = 2400
STRATEGY_START_DAY = 2441


def stable_code_bucket(code: str) -> int:
    return zlib.crc32(str(code).encode("utf-8")) % 100


def build_training_matrix_v3(df: pd.DataFrame, stride: int):
    """V1 windows plus code groups and full future paths for honest simulation."""
    xs, ys, anchor_days, paths20, group_ids, group_codes = [], [], [], [], [], []
    grouped = df.groupby("code", observed=True, sort=False)

    for group_no, (code, g) in enumerate(grouped, start=0):
        g = g.sort_values("day")
        days = g["day"].to_numpy(np.int32)
        values = g[[*PRICE_COLUMNS, "volume"]].to_numpy(np.float32)
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

            xs.append(make_window_features(values, absolute))
            ys.append((future_path[:, -1] > 0).astype(np.uint8))
            anchor_days.append(days[absolute].astype(np.int16))
            paths20.append(future_path.astype(np.float32))
            group_ids.append(np.full(len(absolute), group_no, dtype=np.int16))

        if (group_no + 1) % 500 == 0:
            print(f"  processed {group_no + 1:,}/{grouped.ngroups:,} stocks", flush=True)

    return (
        np.vstack(xs),
        np.concatenate(ys),
        np.concatenate(anchor_days),
        np.vstack(paths20),
        np.concatenate(group_ids),
        np.asarray(group_codes, dtype=object),
    )


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


def percentile_rank(values: np.ndarray) -> np.ndarray:
    if len(values) <= 1:
        return np.full(len(values), 0.5, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks / (len(values) - 1)


def choose_neutral_shrinkage(y_true, raw_probability):
    """Calibrate probability spread while fixing the unstable market base at 0.50."""
    raw_center = float(np.mean(raw_probability))
    best_alpha, best_brier = 0.0, np.inf
    for alpha in np.linspace(0.0, 1.20, 61):
        p = np.clip(
            NEUTRAL_PROBABILITY_CENTER
            + alpha * (raw_probability - raw_center),
            *PROBABILITY_CLIP,
        )
        score = brier_score(y_true, p)
        if score < best_brier:
            best_alpha, best_brier = float(alpha), score
    return best_alpha, best_brier, raw_center


def build_independent_window_batches(group_ids, days, strategy_mask, repeats, seed):
    """One random window per unseen stock; repeat across broad/recent regimes."""
    rng = np.random.default_rng(seed)
    batches = []
    # strategy_mask is already code- and time-disjoint. Repeatedly select one
    # recent window per unseen stock to mimic the independent-window test set.
    regime_starts = (int(days[strategy_mask].min()),)

    for regime_start in regime_starts:
        eligible = np.flatnonzero(strategy_mask & (days >= regime_start))
        by_group = {}
        for idx in eligible:
            by_group.setdefault(int(group_ids[idx]), []).append(int(idx))
        by_group = {g: np.asarray(v, dtype=np.int32) for g, v in by_group.items() if len(v)}

        if len(by_group) < 200:
            continue
        groups = np.asarray(sorted(by_group), dtype=np.int32)
        for _ in range(repeats):
            selected = np.fromiter(
                (rng.choice(by_group[int(g)]) for g in groups),
                dtype=np.int32,
                count=len(groups),
            )
            batches.append(selected)

    if not batches:
        raise RuntimeError("No independent-window validation batches were created")
    return batches


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
    brier = float(np.mean((probability - y) ** 2))
    score = 0.6 * total_return - 0.2 * mdd - 0.2 * brier
    return score, total_return, mdd, brier


def evaluate_gamma(batches, y, paths, p_up, p_tail, gamma):
    rows = []
    for idx in batches:
        tail_rank = percentile_rank(p_tail[idx])
        probability = np.clip(p_up[idx] + gamma * (tail_rank - 0.5),
                              *PROBABILITY_CLIP)
        rows.append(score_one_batch(y[idx], paths[idx], probability))
    a = np.asarray(rows)
    return {
        "score_mean": float(a[:, 0].mean()),
        "score_std": float(a[:, 0].std()),
        "score_median": float(np.median(a[:, 0])),
        "return_mean": float(a[:, 1].mean()),
        "mdd_mean": float(a[:, 2].mean()),
        "brier_mean": float(a[:, 3].mean()),
        "robust_objective": float(a[:, 0].mean() - ROBUSTNESS_PENALTY * a[:, 0].std()),
        "batches": len(rows),
    }


def main():
    print("V3 task-driven configuration")
    print(f"  train:  {TRAIN_PATH}")
    print(f"  test:   {TEST_PATH}")
    print(f"  output: {OUTPUT_PATH}")

    with timer("Read and validate CSV files"):
        train = read_market_csv(TRAIN_PATH)
        test = read_market_csv(TEST_PATH)
        print(f"  train shape={train.shape}, codes={train['code'].nunique():,}")
        print(f"  test  shape={test.shape}, codes={test['code'].nunique():,}")

    with timer("Build V1 windows plus future-path audit data"):
        x, y, days, paths, group_ids, group_codes = build_training_matrix_v3(
            train, TRAIN_STRIDE
        )
        x_test, test_codes = build_test_matrix(test)
        ret20 = paths[:, -1]
        print(f"  feature matrix={x.shape}, test={x_test.shape}")
        del train, test
        gc.collect()

    buckets_by_group = np.asarray(
        [stable_code_bucket(c) for c in group_codes], dtype=np.int16
    )
    sample_bucket = buckets_by_group[group_ids]
    model_train = (
        (sample_bucket < HOLDOUT_CAL_BUCKET_START)
        & (days <= MODEL_TRAIN_END_DAY)
    )
    calibration = (
        (sample_bucket >= HOLDOUT_CAL_BUCKET_START)
        & (sample_bucket < HOLDOUT_STRATEGY_BUCKET_START)
        & (days >= CALIBRATION_START_DAY)
        & (days <= CALIBRATION_END_DAY)
    )
    strategy = (
        (sample_bucket >= HOLDOUT_STRATEGY_BUCKET_START)
        & (days >= STRATEGY_START_DAY)
    )
    print("\nCode- and time-disjoint split")
    print(f"  model train={model_train.sum():,}")
    print(f"  calibration={calibration.sum():,}")
    print(f"  strategy simulation={strategy.sum():,}")

    tail_threshold = float(np.quantile(ret20[model_train], TAIL_QUANTILE))
    tail_y = (ret20 > tail_threshold).astype(np.uint8)
    print(f"  strong-return threshold q{TAIL_QUANTILE:.0%}={tail_threshold:.4%}")
    print(f"  train up-rate={y[model_train].mean():.4f}")
    print(f"  train tail-rate={tail_y[model_train].mean():.4f}")

    with timer("Fit code-disjoint up and strong-return models"):
        up_model, backend = make_binary_model(RANDOM_STATE)
        tail_model, tail_backend = make_binary_model(RANDOM_STATE + 101)
        if tail_backend != backend:
            raise RuntimeError("Model backends disagree")

        up_model = fit_binary(
            up_model, backend,
            x[model_train], y[model_train],
            x[calibration], y[calibration],
        )
        tail_model = fit_binary(
            tail_model, backend,
            x[model_train], tail_y[model_train],
            x[calibration], tail_y[calibration],
        )
        up_rounds = best_round_count(up_model, backend)
        tail_rounds = best_round_count(tail_model, backend)
        raw_up_cal = up_model.predict_proba(x[calibration])[:, 1]
        raw_up_strategy = up_model.predict_proba(x[strategy])[:, 1]
        raw_tail_strategy = tail_model.predict_proba(x[strategy])[:, 1]

    alpha, cal_brier, raw_cal_center = choose_neutral_shrinkage(
        y[calibration], raw_up_cal
    )
    p_up_strategy = np.clip(
        NEUTRAL_PROBABILITY_CENTER
        + alpha * (raw_up_strategy - raw_cal_center),
        *PROBABILITY_CLIP,
    )
    print("\nCalibration")
    print(f"  backend={backend}")
    print(f"  up rounds={up_rounds}, tail rounds={tail_rounds}")
    print(f"  up shrink alpha={alpha:.2f}")
    print(f"  calibration Brier={cal_brier:.6f}")
    print(f"  strategy up AUC={roc_auc_score(y[strategy], raw_up_strategy):.6f}")
    print(f"  strategy tail AUC={roc_auc_score(tail_y[strategy], raw_tail_strategy):.6f}")

    strategy_local_group_ids = group_ids[strategy]
    strategy_days = days[strategy]
    strategy_y = y[strategy]
    strategy_paths = paths[strategy]
    batches = build_independent_window_batches(
        strategy_local_group_ids,
        strategy_days,
        np.ones(strategy.sum(), dtype=bool),
        PSEUDO_BATCH_REPEATS,
        RANDOM_STATE,
    )

    print("\nIndependent-window pseudo-test validation")
    gamma_results = []
    for gamma in TAIL_GAMMAS:
        metrics = evaluate_gamma(
            batches, strategy_y, strategy_paths,
            p_up_strategy, raw_tail_strategy, gamma,
        )
        gamma_results.append((metrics["robust_objective"], gamma, metrics))
        print(
            f"  gamma={gamma:>5.3f} robust={metrics['robust_objective']:.6f} "
            f"mean={metrics['score_mean']:.6f} median={metrics['score_median']:.6f} "
            f"ret={metrics['return_mean']:.3%} mdd={metrics['mdd_mean']:.3%} "
            f"brier={metrics['brier_mean']:.6f}"
        )
    gamma_results.sort(reverse=True, key=lambda z: (z[0], z[2]["score_median"]))
    _, best_gamma, best_metrics = gamma_results[0]
    print(f"\nSelected tail gamma={best_gamma:.3f}")
    print(f"  evaluated on {best_metrics['batches']} independent-window batches")

    max_day = max(int(days.max()), 1)
    sample_weight = (
        0.60 + 0.40 * (days.astype(np.float32) / max_day) ** 2
    ).astype(np.float32)

    with timer("Refit final up and strong-return models on every stock"):
        final_up, final_backend = make_binary_model(RANDOM_STATE, up_rounds)
        final_tail, final_tail_backend = make_binary_model(
            RANDOM_STATE + 101, tail_rounds
        )
        if final_backend != backend or final_tail_backend != backend:
            raise RuntimeError("Final backend changed")
        if backend == "sklearn_hist_gbdt":
            final_up.set_params(early_stopping=False)
            final_tail.set_params(early_stopping=False)
        final_up = fit_binary(
            final_up, backend, x, y, sample_weight=sample_weight
        )
        final_tail = fit_binary(
            final_tail, backend, x, tail_y, sample_weight=sample_weight
        )
        raw_up_test = final_up.predict_proba(x_test)[:, 1]
        raw_tail_test = final_tail.predict_proba(x_test)[:, 1]

    p_up_test = np.clip(
        NEUTRAL_PROBABILITY_CENTER
        + alpha * (raw_up_test - float(raw_up_test.mean())),
        *PROBABILITY_CLIP,
    )
    tail_rank_test = percentile_rank(raw_tail_test)
    test_probability = np.clip(
        p_up_test + best_gamma * (tail_rank_test - 0.5),
        *PROBABILITY_CLIP,
    )

    submission = pd.DataFrame({
        "code": test_codes,
        "up_factor": test_probability.astype(np.float64),
    })
    validate_submission(submission, test_codes)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(
        OUTPUT_PATH, index=False, encoding="utf-8", float_format="%.8f"
    )
    check = pd.read_csv(OUTPUT_PATH)
    validate_submission(check, test_codes)

    print("\nV3 submission created")
    print(f"  path={OUTPUT_PATH}")
    print(f"  rows={len(check):,}")
    print(f"  probability range=[{check.up_factor.min():.6f}, "
          f"{check.up_factor.max():.6f}]")
    print(check.sort_values(
        ["up_factor", "code"], ascending=[False, True]
    ).head(10).to_string(index=False))




# ---------------------------------------------------------------------------
# V3.1 extension: rolling validation + multi-seed tail rank stability.
# ---------------------------------------------------------------------------
TAIL_SEEDS = (20260923, 20261031, 20261207)
TAIL_RANK_STD_PENALTY = 0.15
V31_BATCH_REPEATS = 20
V31_TAIL_GAMMAS = (0.0, 0.0025, 0.0050, 0.0075, 0.0100)
ROLLING_FOLDS = (
    # name, train_end, calibration_start, calibration_end,
    # strategy_start, strategy_end
    ("fold_a", 1600, 1641, 1800, 1841, 2100),
    ("fold_b", 1900, 1941, 2100, 2141, 2400),
    ("fold_c", 2200, 2241, 2400, 2441, 2774),
)


def tail_stability_rank(tail_predictions: np.ndarray) -> np.ndarray:
    """Rank within each seed, reward consensus, penalize seed disagreement."""
    seed_ranks = np.vstack([
        percentile_rank(tail_predictions[i])
        for i in range(tail_predictions.shape[0])
    ])
    stable = (
        seed_ranks.mean(axis=0)
        - TAIL_RANK_STD_PENALTY * seed_ranks.std(axis=0)
    )
    return percentile_rank(stable)


def evaluate_gamma_across_folds(fold_payloads, gamma: float):
    rows = []
    for payload in fold_payloads:
        y_fold = payload["y"]
        paths_fold = payload["paths"]
        p_up_fold = payload["p_up"]
        p_tail_fold = payload["p_tail"]
        for idx in payload["batches"]:
            tail_rank = tail_stability_rank(p_tail_fold[:, idx])
            probability = np.clip(
                p_up_fold[idx] + gamma * (tail_rank - 0.5),
                *PROBABILITY_CLIP,
            )
            rows.append(score_one_batch(y_fold[idx], paths_fold[idx], probability))

    a = np.asarray(rows)
    fold_means = []
    for fold_no in range(len(fold_payloads)):
        fold_rows = [
            row for row, owner in zip(rows, [
                f for f, payload in enumerate(fold_payloads)
                for _ in payload["batches"]
            ]) if owner == fold_no
        ]
        fold_means.append(float(np.mean(np.asarray(fold_rows)[:, 0])))

    return {
        "score_mean": float(a[:, 0].mean()),
        "score_std": float(a[:, 0].std()),
        "score_median": float(np.median(a[:, 0])),
        "return_mean": float(a[:, 1].mean()),
        "mdd_mean": float(a[:, 2].mean()),
        "brier_mean": float(a[:, 3].mean()),
        "fold_score_std": float(np.std(fold_means)),
        "robust_objective": float(
            a[:, 0].mean()
            - ROBUSTNESS_PENALTY * a[:, 0].std()
            - 0.10 * np.std(fold_means)
        ),
        "batches": len(rows),
        "fold_means": fold_means,
    }


def main_v31():
    print("V3.1 robust-tail configuration")
    print(f"  train:  {TRAIN_PATH}")
    print(f"  test:   {TEST_PATH}")
    print(f"  output: {OUTPUT_PATH}")
    print(f"  tail seeds: {TAIL_SEEDS}")

    with timer("Read and validate CSV files"):
        train = read_market_csv(TRAIN_PATH)
        test = read_market_csv(TEST_PATH)
        print(f"  train shape={train.shape}, codes={train['code'].nunique():,}")
        print(f"  test  shape={test.shape}, codes={test['code'].nunique():,}")

    with timer("Build V1 windows and future paths once"):
        x, y, days, paths, group_ids, group_codes = build_training_matrix_v3(
            train, TRAIN_STRIDE
        )
        x_test, test_codes = build_test_matrix(test)
        ret20 = paths[:, -1]
        print(f"  train matrix={x.shape}, test matrix={x_test.shape}")
        del train, test
        gc.collect()

    buckets_by_group = np.asarray(
        [stable_code_bucket(c) for c in group_codes], dtype=np.int16
    )
    sample_bucket = buckets_by_group[group_ids]

    fold_payloads = []
    up_rounds_all = []
    tail_rounds_all = {seed: [] for seed in TAIL_SEEDS}
    alphas = []
    tail_thresholds = []

    for fold_no, (
        fold_name, train_end, cal_start, cal_end, strategy_start, strategy_end
    ) in enumerate(ROLLING_FOLDS):
        print(f"\n{'=' * 18} {fold_name} {'=' * 18}")
        model_train = (
            (sample_bucket < HOLDOUT_CAL_BUCKET_START)
            & (days <= train_end)
        )
        calibration = (
            (sample_bucket >= HOLDOUT_CAL_BUCKET_START)
            & (sample_bucket < HOLDOUT_STRATEGY_BUCKET_START)
            & (days >= cal_start)
            & (days <= cal_end)
        )
        strategy = (
            (sample_bucket >= HOLDOUT_STRATEGY_BUCKET_START)
            & (days >= strategy_start)
            & (days <= strategy_end)
        )
        if min(model_train.sum(), calibration.sum(), strategy.sum()) == 0:
            raise RuntimeError(f"{fold_name} contains an empty split")
        print(
            f"train={model_train.sum():,}, calibration={calibration.sum():,}, "
            f"strategy={strategy.sum():,}"
        )

        tail_threshold = float(np.quantile(ret20[model_train], TAIL_QUANTILE))
        tail_thresholds.append(tail_threshold)
        tail_y = (ret20 > tail_threshold).astype(np.uint8)
        print(f"tail q{TAIL_QUANTILE:.0%} threshold={tail_threshold:.4%}")

        up_model, backend = make_binary_model(RANDOM_STATE + fold_no * 1000)
        up_model = fit_binary(
            up_model, backend,
            x[model_train], y[model_train],
            x[calibration], y[calibration],
        )
        up_rounds_all.append(best_round_count(up_model, backend))
        raw_up_cal = up_model.predict_proba(x[calibration])[:, 1]
        raw_up_strategy = up_model.predict_proba(x[strategy])[:, 1]

        alpha, cal_brier, raw_cal_center = choose_neutral_shrinkage(
            y[calibration], raw_up_cal
        )
        alphas.append(alpha)
        p_up_strategy = np.clip(
            NEUTRAL_PROBABILITY_CENTER
            + alpha * (raw_up_strategy - raw_cal_center),
            *PROBABILITY_CLIP,
        )
        print(
            f"up AUC={roc_auc_score(y[strategy], raw_up_strategy):.6f}, "
            f"alpha={alpha:.2f}, cal Brier={cal_brier:.6f}"
        )
        del up_model
        gc.collect()

        tail_predictions = []
        tail_aucs = []
        for seed in TAIL_SEEDS:
            tail_model, tail_backend = make_binary_model(
                seed + fold_no * 1000
            )
            if tail_backend != backend:
                raise RuntimeError("Tail backend differs from up backend")
            tail_model = fit_binary(
                tail_model, backend,
                x[model_train], tail_y[model_train],
                x[calibration], tail_y[calibration],
            )
            tail_rounds_all[seed].append(
                best_round_count(tail_model, backend)
            )
            prediction = tail_model.predict_proba(x[strategy])[:, 1]
            tail_predictions.append(prediction)
            tail_aucs.append(roc_auc_score(tail_y[strategy], prediction))
            del tail_model
            gc.collect()
        tail_predictions = np.vstack(tail_predictions)
        print(
            "tail AUCs="
            + ", ".join(f"{v:.6f}" for v in tail_aucs)
            + f", mean={np.mean(tail_aucs):.6f}"
        )

        local_group_ids = group_ids[strategy]
        local_days = days[strategy]
        batches = build_independent_window_batches(
            local_group_ids,
            local_days,
            np.ones(strategy.sum(), dtype=bool),
            V31_BATCH_REPEATS,
            RANDOM_STATE + fold_no,
        )
        fold_payloads.append({
            "name": fold_name,
            "y": y[strategy],
            "paths": paths[strategy],
            "p_up": p_up_strategy,
            "p_tail": tail_predictions,
            "batches": batches,
        })

    print("\nRolling-fold gamma selection")
    gamma_results = []
    for gamma in V31_TAIL_GAMMAS:
        metrics = evaluate_gamma_across_folds(fold_payloads, gamma)
        gamma_results.append((metrics["robust_objective"], gamma, metrics))
        fold_text = ",".join(f"{v:.4f}" for v in metrics["fold_means"])
        print(
            f"  gamma={gamma:>5.3f} robust={metrics['robust_objective']:.6f} "
            f"mean={metrics['score_mean']:.6f} median={metrics['score_median']:.6f} "
            f"ret={metrics['return_mean']:.3%} mdd={metrics['mdd_mean']:.3%} "
            f"brier={metrics['brier_mean']:.6f} folds=[{fold_text}]"
        )
    gamma_results.sort(
        reverse=True, key=lambda z: (z[0], z[2]["score_median"])
    )
    _, best_gamma, best_metrics = gamma_results[0]
    print(f"\nSelected gamma={best_gamma:.4f}")
    print(f"  batches={best_metrics['batches']}")
    print(f"  fold score std={best_metrics['fold_score_std']:.6f}")

    final_up_rounds = max(50, int(np.median(up_rounds_all)))
    final_tail_rounds = {
        seed: max(50, int(np.median(rounds)))
        for seed, rounds in tail_rounds_all.items()
    }
    final_alpha = float(np.median(alphas))
    final_tail_threshold = float(np.quantile(ret20, TAIL_QUANTILE))
    final_tail_y = (ret20 > final_tail_threshold).astype(np.uint8)
    print("\nFinal settings")
    print(f"  up rounds={final_up_rounds}, alpha={final_alpha:.2f}")
    print(f"  tail threshold={final_tail_threshold:.4%}")
    print(f"  tail rounds={final_tail_rounds}")

    max_day = max(int(days.max()), 1)
    sample_weight = (
        0.60 + 0.40 * (days.astype(np.float32) / max_day) ** 2
    ).astype(np.float32)

    with timer("Refit final up model"):
        final_up, final_backend = make_binary_model(
            RANDOM_STATE, final_up_rounds
        )
        if final_backend == "sklearn_hist_gbdt":
            final_up.set_params(early_stopping=False)
        final_up = fit_binary(
            final_up, final_backend, x, y, sample_weight=sample_weight
        )
        raw_up_test = final_up.predict_proba(x_test)[:, 1]
        del final_up
        gc.collect()

    final_tail_predictions = []
    with timer(f"Refit {len(TAIL_SEEDS)} final tail models"):
        for seed in TAIL_SEEDS:
            print(f"  seed={seed}, rounds={final_tail_rounds[seed]}")
            model, this_backend = make_binary_model(
                seed, final_tail_rounds[seed]
            )
            if this_backend == "sklearn_hist_gbdt":
                model.set_params(early_stopping=False)
            model = fit_binary(
                model, this_backend,
                x, final_tail_y,
                sample_weight=sample_weight,
            )
            final_tail_predictions.append(
                model.predict_proba(x_test)[:, 1]
            )
            del model
            gc.collect()
    final_tail_predictions = np.vstack(final_tail_predictions)

    p_up_test = np.clip(
        NEUTRAL_PROBABILITY_CENTER
        + final_alpha * (raw_up_test - float(raw_up_test.mean())),
        *PROBABILITY_CLIP,
    )
    stable_tail_rank = tail_stability_rank(final_tail_predictions)
    test_probability = np.clip(
        p_up_test + best_gamma * (stable_tail_rank - 0.5),
        *PROBABILITY_CLIP,
    )

    submission = pd.DataFrame({
        "code": test_codes,
        "up_factor": test_probability.astype(np.float64),
    })
    validate_submission(submission, test_codes)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(
        OUTPUT_PATH, index=False, encoding="utf-8", float_format="%.8f"
    )
    check = pd.read_csv(OUTPUT_PATH)
    validate_submission(check, test_codes)

    print("\nV3.1 submission created")
    print(f"  path={OUTPUT_PATH}")
    print(f"  rows={len(check):,}")
    print(
        f"  probability range=[{check.up_factor.min():.6f}, "
        f"{check.up_factor.max():.6f}]"
    )
    print(
        check.sort_values(
            ["up_factor", "code"], ascending=[False, True]
        ).head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main_v31()
