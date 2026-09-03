# stock20-up-factor-prediction
Forecast 20-day upward probability for anonymized stocks using dual-model (up + absolute-tail) and cross-sectional ranking signals. Top-5 portfolio construction with Brier/MDD/Return scoring.

# Stock20 Up-Factor Prediction

> 20-day upward probability prediction for anonymized stocks | Top-5 portfolio construction | Brier / MDD / Return scoring

---

## Overview

This repository contains my solution for a quantitative finance competition (Kaggle-style).  
The task is to predict the probability that each stock will have a positive cumulative return over the next 20 trading days, based solely on the latest 20‑day OHLCV window.

The final evaluation score is a weighted combination of:

- **Top‑5 portfolio return** (weight 0.6)
- **Maximum drawdown** (penalty weight 0.2)
- **Brier score** (penalty weight 0.2)

**Key challenge**: The test set consists of 1,500 completely new stocks (anonymized codes) with only 20 days of price data – no historical code alignment, no industry/fundamental information. All features must be scale‑free and computed from a single 20‑day window.

---

## Data

- **Training set**: 4,375 stocks × ~2,794 days, ~8.04M rows (OHLCV only)
- **Test set**: 1,500 stocks × 20 days (anonymized, no overlap with training codes)
- **Target**: For each stock, predict `up_factor` ∈ [0,1] = probability that `close[t+20] / close[t] > 0`

**Important**: The test set is an independent cold‑start sample – training code identities cannot be used for alignment.

---

## Methodology

### Core Idea

A single classification model for “up/down” is insufficient because the top‑5 selection is extremely sensitive to extreme positive returns.  
We adopt a **dual‑model architecture**:

1. **Up model** – predicts the probability of positive return (serves Brier score and broad ranking).
2. **Absolute‑tail model** – predicts whether the 20‑day return exceeds a global 80th‑percentile threshold (serves Top‑5 selection).

Final score is:
final_score = calibrated_up_prob + gamma * (tail_rank_percentile - 0.5)

where `gamma` is tuned on a hold‑out validation set.

### Feature Engineering

All features are derived from the **last 20 days** of OHLCV and are scale‑free (ratios, log‑differences, ranks).  
The base feature set (59 dimensions) includes:

- 19 daily returns
- Multi‑horizon returns (1, 2, 3, 5, 10, 19 days)
- Rolling statistics (mean, std, downside deviation, skew)
- Linear trend slope and residual std
- Candle patterns (range, body, close location, gaps, true range)
- Volume relative metrics and volume‑return correlation

### Model

- **LightGBM** with binary classification (`logloss`) for both up and tail models.
- Early stopping with 60 rounds patience.
- Probability calibration via scaling (shrinkage) to a neutral 0.50 center using validation set Brier optimization.

### Validation Strategy

To avoid leakage, we use a **code‑bucket + time‑window** split:

- Stocks are hashed into 100 buckets (CRC32).
- Training: bucket < 60 & day ≤ 2200  
- Calibration: bucket 60‑78 & day 2241‑2400  
- Strategy validation: bucket ≥ 78 & day ≥ 2441  

This ensures that training, calibration, and validation sets are **disjoint in both stock identity and time**.

We also experimented with a more realistic **exact‑1,500 cold‑stock time‑fold** framework (three folds with non‑overlapping 20‑day future windows) to reduce false positives.

---

## Key Experiments & Results

| Version | Description | Online Score |
| :--- | :--- | :--- |
| V1 | Baseline up‑classifier | -0.0239 |
| V2 | Multi‑task / balanced / auto‑fusion | -0.05 ~ -0.07 |
| **V3** | **Up + absolute‑tail dual model** | **+0.0285** |
| V3.1 | Multi‑seed / rolling folds | 0.0194 |
| V3.2 | Enhanced tail sequence features (fallback to V3) | 0.0285 |
| V3.2‑A1 | Added cross‑sectional return features (fallback) | **0.03403** (best) |
| Pool50 | Candidate‑pool (Top50) + absolute re‑ranking | -0.1097 (failed) |

**Best online score**: **0.03403** (achieved by V3.2‑A1 fallback, which essentially reproduced V3 with a deterministic row‑order).

The A1 experiments (cross‑sectional return ranks / Z‑scores) showed that relative‑strength signals have decent AUC (~0.59) but failed to improve Top‑5 precision when naively overlaid. The main bottleneck remains **“good at picking strong candidates, but bad at ranking the top few”**.

---

## Project Structure
```text
├── baseline_v3_2.py          # Main stable script (V3 dual‑model)
├── baseline_v3_2_a1.py       # A1 cross‑sectional feature extension
├── baseline_v3_2_a1_1.py     # Relative‑tail target extension
├── validation_v5_*.py        # Strict time‑fold validation framework
├── diagnostic_*.py           # Diagnostic tools for model analysis
└── README.md
```

---

## How to Reproduce

1. Place the competition data in the expected paths:
   - Train: `test-data/train.csv`
   - Test: `true-test-data/test.csv`
2. Run `baseline_v3_2.py` (or `v3_2_a1.py` for the best version).
3. The submission file `submission_v3_2.csv` will be generated.

**Dependencies**: Python 3.8+, numpy, pandas, scikit‑learn, lightgbm.

---

## Lessons Learned

- **Classification alone is not enough**: You need a dedicated model to pick extreme winners.
- **Cross‑sectional information matters**, but should be integrated carefully – naive overlay can hurt.
- **Validation framework must mimic the test environment**: small validation sets can overstate performance.
- **Top‑5 selection is fragile**: a single replacement can cause huge score swings.
- **Simple baselines with proper calibration often outperform complex ensembles** in low‑signal, cold‑start settings.

---

## Future Ideas

- Two‑stage pipeline: recall (Top‑50) + re‑ranking using Learning‑to‑Rank (LambdaRank) with richer features.
- Multi‑grade labels (ordinal regression) instead of binary up/down.
- Cross‑stock attention / graph‑based features to capture stock inter‑dependencies.

---

## Acknowledgements

This work was done as part of a Kaggle‑style competition. Special thanks to the organizers and the open‑source community for providing valuable reference implementations.

---

## License

MIT

---

## Contact

Feel free to open an issue or reach out if you have questions about the methodology.  
(Add your email or GitHub handle here if you wish.)

License
MIT

