# VMD-CA-EWT-KANInformer: Short-Term Wind Speed Forecasting

Implementation of the complete forecasting pipeline from:

> **"Short-term Wind Speed Forecasting Based on a Novel KANInformer Model and Improved Dual Decomposition"**  
> *Energy (Elsevier, 2025)*

The pipeline combines:
- **VMD** (Variational Mode Decomposition) — decompose wind speed into IMFs
- **CA** (Component Aggregation) — novel step fusing high-entropy IMFs
- **EWT** (Empirical Wavelet Transform) — further decompose the fused component
- **KANInformer** — Informer with KAN layers replacing every FFN block

---

## Installation

```bash
pip install -r requirements.txt
```

> **Note:** `pykan` requires Python ≥ 3.9. If `pip install pykan` fails, try:
> ```bash
> pip install git+https://github.com/KindXiaoming/pykan.git
> ```

---

## Data Setup

1. Go to [https://cimis.water.ca.gov/](https://cimis.water.ca.gov/)
2. Select **Station 47 — Brentwood**
3. Download **hourly data** for the period **December 1, 2020 to December 1, 2021**
4. Export as CSV
5. Place the file at:
   ```
   data/hourly.csv
   ```

The CSV must contain these columns (CIMIS standard format with QC columns):
```
Stn Id, Stn Name, CIMIS Region, Date, Hour (PST), Jul,
ETo (in), [qc], Precip (in), [qc], Sol Rad (Ly/day), [qc],
Vap Pres (mBars), [qc], Air Temp (F), [qc], Rel Hum (%), [qc],
Dew Point (F), [qc], Wind Speed (mph), [qc], Wind Dir (0-360), [qc], Soil Temp (F), [qc]
```

Step 1 automatically drops station metadata (`Stn Id`, `Stn Name`, `CIMIS Region`, `Jul`) and all QC columns, then renames the weather fields to `ET`, `PCP`, `SR`, `VP`, `AT`, `RH`, `DPT`, `WS`, `WD`, `ST`.

---

## Running

```bash
python run_pipeline.py
```

This single command executes all steps in order:

| Step | Script | Description |
|------|--------|-------------|
| 1 | `step1_prepare_seasons.py` | Load CSV, convert units, split by season |
| 2 | `step2_pcc_selection.py` | Pearson correlation feature selection |
| 3 | `step3_vmd_decompose.py` | VMD decomposition, find optimal K |
| 4 | `step4_sample_entropy.py` | Sample entropy per IMF |
| 5 | `step5_component_aggregation.py` | Aggregate high-SE IMFs (CA step) |
| 6 | `step6_ewt_decompose.py` | EWT on fused IMF (8 modes) |
| 7 | `step7_build_feature_matrix.py` | Build final feature matrix |
| 8 | `step8_normalize_and_window.py` | MinMaxScaler + sliding windows |
| — | `KANInformer.py` | Train and evaluate model |

### Options

```bash
python run_pipeline.py --force           # Recompute everything from scratch
python run_pipeline.py --season spring   # Run only for spring season
```

---

## Output Structure

```
outputs/
├── spring_predictions.csv   ← actual vs predicted for 1/2/3-step
├── summer_predictions.csv
├── autumn_predictions.csv
└── winter_predictions.csv

results/
├── table9_results.csv         ← metrics vs paper Table 9
├── pcc_results_all_seasons.csv
└── se_values_all_seasons.csv

model/
├── spring_best_model.pt
├── summer_best_model.pt
├── autumn_best_model.pt
└── winter_best_model.pt
```

---

## Expected Results (Paper Table 9)

| Season | Step | RMSE | MAE | MAPE (%) |
|--------|------|------|-----|----------|
| Spring | 1    | 0.641 | 0.509 | 18.2 |
| Spring | 2    | 0.821 | 0.645 | 23.1 |
| Spring | 3    | 0.910 | 0.729 | 26.6 |
| Summer | 1    | 0.424 | 0.325 | 18.3 |
| Summer | 2    | 0.505 | 0.392 | 21.9 |
| Summer | 3    | 0.518 | 0.407 | 23.2 |
| Autumn | 1    | 0.495 | 0.330 | 22.9 |
| Autumn | 2    | 0.764 | 0.467 | 27.9 |
| Autumn | 3    | 0.978 | 0.562 | 31.2 |
| Winter | 1    | 0.725 | 0.527 | 25.5 |
| Winter | 2    | 0.969 | 0.693 | 30.6 |
| Winter | 3    | 1.093 | 0.784 | 34.4 |

All RMSE/MAE values are in **m/s** (wind speed converted from mph at ingestion).  
The final `results/table9_results.csv` compares your run against the paper values.

---

## Key Implementation Details

- **No data leakage**: MinMaxScaler fit on training rows only; applied to val/test.
- **Chronological splits**: 80/10/10 ratio, no shuffling anywhere.
- **WS always last column**: target Y always from the last column of the feature matrix.
- **VMD on full season**: IMFs computed on complete seasonal series, then split with features.
- **EWT on full fused IMF**: same logic as VMD.
- **Random seeds**: `torch.manual_seed(42)`, `np.random.seed(42)` before training.
- **Early stopping**: patience=3 epochs; best weights restored for evaluation.

---

## References

- Paper: Energy (Elsevier, 2025) — DOI to be added
- KANInformer source: https://github.com/375330014/lzy
- pykan (KAN library): https://github.com/KindXiaoming/pykan
- Informer base: https://github.com/zhouhaoyi/Informer2020
- Data source: https://cimis.water.ca.gov/ (Station 47, Brentwood, CA)
