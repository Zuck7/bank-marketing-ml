# Bank Marketing ML Pipeline

> End-to-end production ML pipeline using TFX and Apache Airflow to predict term deposit subscriptions from bank marketing campaign data.

Built for COMP315 (AI Software Test & ML/Ops) at Centennial College, Fall 2026.

---

## What This Project Does

A Portuguese bank ran phone-based marketing campaigns to sell term deposits. This project builds a fully automated ML pipeline that ingests the raw campaign data, validates its quality, engineers features, trains a neural network, evaluates the model for fairness across demographic slices, and deploys it to a serving directory — all orchestrated as a reproducible DAG.

The pipeline achieves **91.4% accuracy** and **0.90 AUC** on the validation set, with TFMA analysis revealing performance gaps across education levels, marital status, and job types.

---

## Results

### Model Performance

| Metric | Run 1 (500 steps) | Run 2 (1000 steps) |
|--------|-------------------|---------------------|
| Binary Accuracy | 91.4% | 90.1% |
| AUC | 0.900 | 0.907 |
| Training Loss | 0.273 | 0.260 |
| Validation Loss | 0.212 | 0.236 |

Run 1 achieves higher accuracy; Run 2 achieves higher AUC. The slight accuracy drop in Run 2 with improved AUC suggests better calibration across the full prediction range at the cost of marginal accuracy on the majority class. This is a common tradeoff with imbalanced datasets (88% negative, 12% positive).

### TFMA Slice Analysis

The model was evaluated across education, marital status, and job type slices to detect fairness gaps:

- **Education:** Tertiary-educated clients receive the most accurate predictions; primary education shows the widest performance gap
- **Job type:** Students are the hardest slice to predict (76.1% accuracy) — 15 percentage points below the best-performing category
- **Marital status:** Relatively uniform performance across married, single, and divorced groups

These gaps are driven by class imbalance within slices — students and primary-education clients have different subscription patterns that the model struggles to learn from limited examples.

---

## Pipeline Architecture

```
bank-full.csv (45,211 rows, 17 features)
       │
       ▼
┌─────────────┐     ┌────────────────┐     ┌────────────┐     ┌──────────────────┐
│ CsvExampleGen│────▶│ StatisticsGen  │────▶│ SchemaGen   │────▶│ ExampleValidator │
│ (Data Ingest)│     │ (Compute Stats)│     │(Infer Types)│     │ (Check Anomalies)│
└─────────────┘     └────────────────┘     └────────────┘     └──────────────────┘
       │                                          │
       ▼                                          ▼
┌─────────────┐                            ┌─────────────┐
│  Transform   │◀───────────────────────────│   Schema     │
│(Feature Eng.)│                            └─────────────┘
└─────────────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Trainer    │────▶│  Evaluator   │◀────│  Resolver    │     │   Pusher    │
│ (Train Model)│     │  (TFMA Eval) │     │(Get Baseline)│     │  (Deploy)   │
└─────────────┘     └─────────────┘                          └─────────────┘
                           │                                        ▲
                           │  blessed?                              │
                           └────────────────────────────────────────┘
```

Each component produces versioned artifacts tracked by ML Metadata (MLMD). The Resolver retrieves the latest blessed model so the Evaluator can compare new models against baselines — preventing deployment of regressions.

---

## Dataset

**Source:** [Bank Marketing Dataset](http://hdl.handle.net/1822/14838) (Moro et al., 2011)

| Property | Value |
|----------|-------|
| Records | 45,211 |
| Features | 16 input + 1 target |
| Target | `y` — subscribed to term deposit? (yes/no) |
| Class Balance | 88.3% no, 11.7% yes |
| Original Format | Semicolon-delimited CSV |

**Preprocessing required:** The original CSV uses semicolons and text labels (`yes`/`no`). TFX's CsvExampleGen expects comma-delimited files with numeric labels. A preprocessing step converts the delimiter and maps `yes→1`, `no→0`.

### Features

**Numeric (7):** age, balance, day, duration, campaign, pdays, previous

**Categorical (9):** job (12 vals), marital (3), education (4), default (2), housing (2), loan (2), contact (3), month (12), poutcome (4)

---

## Model Architecture

```
Numeric features (7) ── z-score scaling ──────────────────┐
                                                          │
Categorical features (9) ── vocab lookup ── Embedding ── Flatten ─┤
                                                          │
Age ── bucketize (5 bins) ── Embedding ── Flatten ────────┤
                                                          │
                                                   Concatenate (37 dims)
                                                          │
                                                   Dense(128) + BatchNorm + Dropout(0.3)
                                                          │
                                                   Dense(64)  + BatchNorm + Dropout(0.2)
                                                          │
                                                   Dense(32)
                                                          │
                                                   Dense(1, sigmoid)
```

| Property | Value |
|----------|-------|
| Total Parameters | 16,235 (63.42 KB) |
| Trainable Parameters | 15,851 |
| Optimizer | Adam (lr=0.001) |
| Loss | Binary Crossentropy |
| Callbacks | TensorBoard, Early Stopping (patience=5) |

**Why embeddings over one-hot?** One-hot encoding `job` (12 categories) creates a sparse 12-dimensional vector. An embedding maps it to a dense 6-dimensional space where semantically similar jobs can cluster together, giving the model more expressive power with fewer parameters.

**Why batch normalization?** Normalizes activations between layers to stabilize training and allow higher learning rates. Particularly helpful when combining numeric features (already z-scored) with embedding outputs (learned during training) that start on different scales.

---

## Feature Engineering (Transform Module)

All transformations are baked into the SavedModel's serving graph via TensorFlow Transform, eliminating training/serving skew:

| Transformation | Features | Purpose |
|---------------|----------|---------|
| `scale_to_z_score` | All 7 numeric | Center around 0 with std=1 so no feature dominates by scale |
| `compute_and_apply_vocabulary` | All 9 categorical | Convert strings to integer IDs with OOV bucket for unseen values |
| `bucketize` | age (5 bins) | Capture non-linear age-subscription relationship (students, working age, retired) |

---

## Airflow Orchestration

The pipeline runs as an Airflow DAG where each TFX component is an individual task with dependency tracking:

```
CsvExampleGen → StatisticsGen → SchemaGen → ExampleValidator
                                    ↓
                                Transform → Trainer → Resolver
                                                ↓         ↓
                                            Evaluator ←───┘
                                                ↓
                                             Pusher
```

Airflow provides scheduling, retry logic, dependency resolution, and monitoring. The DAG is configured with `schedule_interval=None` (manual trigger only) for this project.

**DAG test run:** All 9 tasks completed successfully, with each task logged as `Marking task as SUCCESS` and the run finishing with `DagRun Finished: state:success`.

---

## Project Structure

```
COMP315_Project/
├── data/
│   ├── bank-full.csv                  # Original dataset (semicolons)
│   └── processed/
│       └── bank_marketing.csv         # Preprocessed (commas, numeric labels)
├── modules/
│   ├── transform_module.py            # preprocessing_fn — feature engineering
│   └── trainer_module.py              # run_fn — Keras model + TensorBoard
├── pipeline/
│   └── pipeline.py                    # create_pipeline() for Airflow
├── dags/
│   └── bank_marketing_dag.py          # Airflow DAG definition
├── notebooks/
│   └── phase1-3_pipeline.ipynb        # Full pipeline execution notebook
├── outputs/
│   ├── artifacts/
│   │   ├── eval_run_1/                # TFMA results (500 steps)
│   │   ├── eval_run_2/                # TFMA results (1000 steps)
│   │   ├── model_run_1/               # TensorBoard logs (500 steps)
│   │   ├── model_run_2/               # TensorBoard logs (1000 steps)
│   │   └── trained_model/             # Exported SavedModel
│   └── screenshots/
│       ├── airflow_dag_graph.png       # DAG component graph
│       └── airflow_dag_run_log.txt     # Full Airflow run log
├── requirements.txt
└── README.md
```

---

## Environment Setup

TFX 1.15.0 requires Python 3.10 and has strict version pins across its entire dependency chain. Google Colab runs Python 3.12, so the pipeline runs inside a Python 3.10 virtual environment.

### Why Python 3.10?

`ml-metadata`, a C++ compiled package that TFX uses to track artifacts, only has pre-built wheels for Python ≤3.10. No amount of pip configuration can work around a missing binary wheel — the version constraint is hard.

### Setup (Google Colab)

```bash
# Create Python 3.10 venv (Colab has 3.10 via Ubuntu 22.04)
sudo apt install python3.10-venv -y
python3.10 -m venv /content/tfx_env --without-pip
curl -sS https://bootstrap.pypa.io/get-pip.py | /content/tfx_env/bin/python3.10

# Install TFX with pinned dependencies (--no-deps to bypass resolver)
/content/tfx_env/bin/pip install --no-deps tfx==1.15.0 tensorflow==2.15.0 ...
```

The full dependency list is in the notebook's Cell 2. The `--no-deps` flag is necessary because pip's dependency resolver enters an infinite backtracking loop when resolving the TFX + Apache Beam + Google Cloud dependency tree. Every package is pinned to an exact compatible version instead.

---

## Tech Stack

| Component | Version | Purpose |
|-----------|---------|---------|
| TFX | 1.15.0 | ML pipeline framework |
| TensorFlow | 2.15.0 | Model training and serving |
| TensorFlow Transform | 1.15.0 | Feature engineering with serving graph |
| TFMA | 0.46.0 | Model evaluation, slicing, fairness |
| TFDV | 1.15.1 | Data validation and anomaly detection |
| Apache Beam | 2.56.0 | Distributed data processing backend |
| Apache Airflow | 2.7.3 | Pipeline orchestration and scheduling |
| ML Metadata | 1.15.0 | Artifact lineage tracking |
| Keras | 2.15.0 | Neural network API |
| Python | 3.10.12 | Runtime (venv inside Colab's 3.12) |

---

## Challenges & Solutions

### TFX Won't Install on Python 3.12
**Problem:** `ml-metadata` has no wheel for Python 3.12. pip fails with `No matching distribution found`.
**Solution:** Created a Python 3.10 virtual environment inside Colab using Ubuntu's built-in Python 3.10, bootstrapped pip manually, and ran all TFX code through the venv via subprocess.

### pip Dependency Resolution Infinite Loop
**Problem:** TFX's dependency tree (TFX → Apache Beam[GCP] → 50+ Google Cloud packages) causes pip to backtrack through thousands of version combinations, eventually hitting `resolution-too-deep`.
**Solution:** Pre-installed all packages with exact pinned versions using `--no-deps`, then fixed missing transitive dependencies individually.

### InteractiveContext Only Works in Jupyter
**Problem:** TFX's `InteractiveContext.run()` is a no-op outside IPython/Jupyter. Since the pipeline runs in a subprocess (separate Python 3.10 process), `context.run()` silently does nothing.
**Solution:** Switched to `LocalDagRunner` which executes components as a proper pipeline without requiring a notebook kernel.

### Airflow Webserver Timeout on Colab Free Tier
**Problem:** Gunicorn workers timeout after 120 seconds on Colab's limited CPU, crashing the Airflow web UI before it finishes loading.
**Solution:** Used Airflow CLI commands (`airflow dags show`, `airflow dags test`) to generate the DAG graph image and run the pipeline directly, bypassing the web UI entirely.

### Protobuf Version Conflicts
**Problem:** Installing Google Cloud packages pulls in `protobuf 6.x`, but TFX 1.15.0 requires `protobuf <5`. pip's resolver doesn't catch this because of `--no-deps`.
**Solution:** Added a "fix overwrites" step that forces `protobuf==4.25.9` back after installing packages that might upgrade it.

---

## Key Takeaways

1. **Data validation catches what manual inspection misses.** ExampleValidator compares every feature's statistics against an inferred schema automatically. In production, this runs on every new data batch before training begins.

2. **Overall accuracy hides fairness gaps.** A model at 91% overall can be 76% for students. TFMA's slicing metrics make this visible without writing custom evaluation code.

3. **Transform eliminates training/serving skew.** Feature preprocessing (z-scoring, vocabulary encoding, bucketization) is baked into the SavedModel. The exact same transformations run at prediction time with zero additional code.

4. **ML infrastructure is harder than ML modeling.** The model architecture took 30 minutes to write. Getting TFX, its 50+ dependencies, Airflow, and Python version constraints to work together took significantly longer. This is representative of real-world ML engineering.

5. **Artifact lineage enables reproducibility.** Every pipeline run records which data was used, what schema was inferred, which model was trained, whether it was blessed, and where it was deployed. ML Metadata makes this queryable.

---

## How to Reproduce

1. Upload `bank-full.csv` to `COMP315_Project/data/` on Google Drive
2. Upload `transform_module.py` and `trainer_module.py` to `COMP315_Project/modules/`
3. Open `phase1-3_pipeline.ipynb` in Google Colab
4. Run cells 1–8 sequentially (total time: ~30 minutes)
5. Artifacts appear in `COMP315_Project/outputs/artifacts/` on Drive

---

## FAQ

**Q: Why not just use a Jupyter notebook with scikit-learn?**
A: The point of TFX is to build a production pipeline, not just train a model. In industry, models need automated data validation, reproducible training, baseline comparison before deployment, and artifact tracking. TFX provides all of this as infrastructure.

**Q: Why is the model relatively simple (3 dense layers)?**
A: For tabular data with 16 features, a 16K-parameter network is appropriately sized. Larger models overfit on this dataset. The architecture choices (embeddings for categoricals, batch normalization, dropout) matter more than depth.

**Q: Why does the second run (1000 steps) have slightly lower accuracy but higher AUC?**
A: More training steps improve the model's ranking ability (AUC) but can slightly overfit on the majority class (no-subscription), reducing accuracy on the minority class. For an imbalanced dataset, AUC is the more meaningful metric.

**Q: What would you change for production?**
A: Add data versioning (DVC), model monitoring for drift detection, A/B testing before full rollout, a CI/CD pipeline for model retraining, and class weighting or SMOTE to address the 88/12 imbalance.

**Q: Why LocalDagRunner instead of running Airflow end-to-end?**
A: Both use the same pipeline definition and produce identical results. LocalDagRunner runs components sequentially in a single process (simpler for development). Airflow runs them as separate tasks with scheduling, retries, and monitoring (necessary for production). We demonstrated both.

**Q: What does "blessed" mean in the context of the Evaluator?**
A: A model is "blessed" when it passes the Evaluator's quality checks — meaning it performs at least as well as the current baseline model. Only blessed models are deployed by the Pusher. This prevents deploying a model that regressed in performance.

**Q: How does the pipeline handle unseen categorical values at serving time?**
A: The Transform module's `compute_and_apply_vocabulary` uses `num_oov_buckets=1`, which creates a dedicated bucket for out-of-vocabulary values. Any category not seen during training gets mapped to this OOV bucket rather than causing an error.

**Q: Why does CsvExampleGen need a separate subfolder?**
A: CsvExampleGen reads every CSV file in the directory you point it to. If the original semicolon-delimited `bank-full.csv` is in the same folder as the preprocessed `bank_marketing.csv`, it tries to parse both and fails on the semicolons.

---

## References

Moro, S., Laureano, R., & Cortez, P. (2011). Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology. *Proceedings of the European Simulation and Modelling Conference — ESM'2011*, pp. 117-121, Guimaraes, Portugal. EUROSIS. [http://hdl.handle.net/1822/14838](http://hdl.handle.net/1822/14838)

---

## Authors

COMP315 AI Software Test & ML/Ops — Centennial College, Fall 2026
