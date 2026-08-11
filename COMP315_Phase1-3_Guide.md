# What I Did — Phases 1 to 3 Guide

This documents everything I built and ran for the COMP315 Term Project, covering pipeline construction (Phase 1), model training and deployment (Phase 2), and Airflow orchestration (Phase 3).

---

## Phase 1: Data Pipeline (Steps 1–4)

### What I Did

Built the data ingestion and validation pipeline — the foundation that every downstream step depends on. This takes raw CSV data and produces validated, profiled datasets ready for feature engineering.

### Step-by-Step

**Step 0 — Data Preprocessing**

The original `bank-full.csv` uses semicolon delimiters and text labels (`yes`/`no`). TFX's CsvExampleGen only reads comma-separated CSVs with numeric values. I wrote a preprocessing script that:
- Reads the semicolon-delimited file with pandas
- Maps the label column: `yes → 1`, `no → 0`
- Saves to a separate `data/processed/` subfolder as `bank_marketing.csv`

The separate subfolder matters because CsvExampleGen reads every CSV in the directory — if both the original and processed files are in the same folder, it tries to parse the semicolon file and crashes.

**Step 1 — ExampleGen (Data Ingestion)**

CsvExampleGen reads `bank_marketing.csv`, automatically splits it into train (2/3) and eval (1/3) sets, and converts everything to TFRecord format. TFRecords are TensorFlow's binary storage format — they're faster to read during training than CSV because the data is pre-serialized.

The split is deterministic and recorded in ML Metadata, so every downstream component knows exactly which data it's working with.

**Step 2 — StatisticsGen (Data Profiling)**

Computes per-feature statistics across both train and eval splits: min, max, mean, standard deviation, missing value counts, histograms, and distribution shapes. This gives a quantitative profile of the entire dataset.

Output: statistical artifacts stored in the pipeline directory, used by SchemaGen and ExampleValidator.

**Step 3 — SchemaGen (Schema Inference)**

Automatically infers what the data should look like based on the computed statistics:
- Which features exist and their types (INT, FLOAT, STRING)
- Expected value ranges for numeric features
- Vocabulary lists for categorical features (all 12 job types, 3 marital statuses, etc.)

The schema acts as a contract. In production, every new batch of data gets validated against this schema before training begins.

**Inferred schema for the Bank Marketing dataset:**
- Numeric (INT/FLOAT): age, balance, day, duration, campaign, pdays, previous, y
- Categorical (STRING): job, marital, education, default, housing, loan, contact, month, poutcome
- Notable: `pdays` has a sentinel value of -1 meaning "never previously contacted"

**Step 4 — ExampleValidator (Anomaly Detection)**

Compares actual data statistics against the inferred schema and flags anything suspicious — missing values, wrong types, out-of-range values, unexpected categories.

**Result: No anomalies detected.** The Bank Marketing dataset is clean — no missing values, no type mismatches, all categorical values within expected vocabularies.

### How I Ran It

All four components were assembled into a TFX Pipeline and executed using `LocalDagRunner`. I originally tried `InteractiveContext` but discovered it's a no-op when run outside a Jupyter kernel (our code runs in a subprocess through the Python 3.10 venv). `LocalDagRunner` executes the same components as a proper pipeline without requiring IPython.

### Output

```
=== PHASE 1 COMPLETE ===

Examples (3 artifacts)
ExampleStatistics (2 artifacts)
Schema (2 artifacts)
ExampleAnomalies (2 artifacts)
```

---

## Phase 2: Model Training & Deployment (Steps 5–9)

### What I Did

Built the feature engineering, model training, evaluation, and deployment pipeline. This is the core ML work — transforming raw features, training a neural network, evaluating it for fairness, and deploying it if it passes quality checks.

### Step-by-Step

**Step 5 — Transform (Feature Engineering)**

Created `transform_module.py` with a `preprocessing_fn()` function that defines three types of transformations:

1. **Z-score scaling** on all 7 numeric features — centers each around 0 with std=1 so no feature dominates by magnitude
2. **Vocabulary encoding** on all 9 categorical features — converts strings to integer IDs with an out-of-vocabulary bucket for unseen values at serving time
3. **Bucketization** on age — splits into 5 bins to capture the non-linear relationship between age groups and subscription behavior

These transformations get baked into the SavedModel's serving graph. At prediction time, the same preprocessing runs automatically — no separate preprocessing code needed, no risk of training/serving skew.

**Step 6 — Trainer (Model Training)**

Created `trainer_module.py` with a `run_fn()` function that:
1. Builds a Keras model with embedding layers for categorical features, batch normalization, and dropout
2. Trains with Adam optimizer and binary crossentropy loss
3. Logs metrics to TensorBoard via a callback
4. Saves the model in SavedModel format with a serving signature that includes the Transform preprocessing

**Model architecture:** 7 numeric inputs + 9 categorical embeddings + 1 bucketized age embedding → concatenated (37 dims) → Dense(128) + BatchNorm + Dropout(0.3) → Dense(64) + BatchNorm + Dropout(0.2) → Dense(32) → Dense(1, sigmoid)

**Total parameters:** 16,235 (63.42 KB) — deliberately small for tabular data with 16 features.

**Step 7 — Resolver (Baseline Lookup)**

Queries ML Metadata to find the most recently "blessed" (approved) model. On the first run, there's no baseline — the Evaluator auto-blesses. On subsequent runs, the new model must perform at least as well as this baseline to be blessed. This prevents deploying a regression.

**Step 8 — Evaluator (TFMA Analysis)**

Runs TensorFlow Model Analysis with slicing across three dimensions:
- **Education:** primary, secondary, tertiary, unknown
- **Marital status:** married, single, divorced
- **Job type:** 12 categories from admin to unemployed

Metrics computed per slice: BinaryAccuracy, AUC, ExampleCount.

Both models were **blessed** — they passed quality checks.

**Step 9 — Pusher (Model Deployment)**

Copies the blessed model to the serving directory. In production this would push to TF Serving or a cloud endpoint; for this project it copies to a local filesystem path.

### Two Pipeline Runs

I ran the full pipeline twice with different hyperparameters to enable TFMA comparison:

**Run 1 (500 training steps):**
```
500/500 ━━━━━━━━━━━━━━━━━━━━ 9s
Training:     loss: 0.2727  accuracy: 88.4%  AUC: 0.848
Validation:   loss: 0.2118  accuracy: 91.4%  AUC: 0.900
Model: BLESSED and PUSHED
```

**Run 2 (1000 training steps):**
```
1000/1000 ━━━━━━━━━━━━━━━━━━━━ 12s
Training:     loss: 0.2600  accuracy: 88.7%  AUC: 0.867
Validation:   loss: 0.2359  accuracy: 90.1%  AUC: 0.907
Model: BLESSED and PUSHED
```

Run 1 has higher accuracy (91.4% vs 90.1%); Run 2 has higher AUC (0.907 vs 0.900). More training steps improved the model's ranking ability but slightly overfitted on the majority class.

### Artifacts Saved to Google Drive

After each run, I copied key artifacts to the shared Drive folder so groupmates could use them:

```
outputs/artifacts/
├── eval_run_1/      ← TFMA results from 500-step run
├── eval_run_2/      ← TFMA results from 1000-step run
├── model_run_1/     ← TensorBoard logs from 500-step run
├── model_run_2/     ← TensorBoard logs from 1000-step run
└── trained_model/   ← Exported SavedModel
```

### Output

```
=== RUN 1 COMPLETE (ALL 9 STEPS) ===

Examples (10 artifacts)
ExampleStatistics (14 artifacts)
Schema (14 artifacts)
ExampleAnomalies (10 artifacts)
TransformGraph (4 artifacts)
TransformCache (4 artifacts)
Model (4 artifacts)
ModelRun (4 artifacts)
ModelBlessing (4 artifacts)
ModelEvaluation (4 artifacts)
PushedModel (4 artifacts)

Evaluation saved to Drive
TensorBoard logs saved to Drive
```

---

## Phase 3: Airflow Orchestration

### What I Did

Wrapped the TFX pipeline in an Apache Airflow DAG so each component runs as a managed task with dependency tracking, scheduling, and monitoring. Generated the DAG graph image and ran the full pipeline through Airflow to demonstrate orchestrated execution.

### Step-by-Step

**Step 1 — Install Airflow 2.7.3**

TFX 1.15.0's `AirflowDagRunner` requires Airflow 2.x. The earlier dependency installs had pulled in Airflow 3.3.0, which is incompatible. I installed Airflow 2.7.3 specifically, using Apache's official constraints file to pin compatible versions, then re-pinned protobuf and grpcio to prevent overwrites.

**Step 2 — Initialize Airflow**

- Initialized the Airflow metadata database (`airflow db init`)
- Created an admin user (admin/admin)
- Both run against the Python 3.10 venv's Airflow installation

**Step 3 — Create the DAG File**

Wrote `bank_marketing_dag.py` that:
1. Imports all 9 TFX components
2. Defines the same pipeline as Phase 2 (with reduced training steps for faster Airflow runs)
3. Configures `schedule_interval=None` (manual trigger only)
4. Uses `AirflowDagRunner` to convert the TFX pipeline into an Airflow DAG

**Bug fix:** The initial DAG file was missing `import tfx`, causing a `NameError` when Airflow tried to parse it. Added the import and verified with `airflow dags list`.

**Step 4 — Fix Dependency Issues**

Two additional fixes were needed:
- `jupyter_client` was too new (v8.x) for the venv's Python 3.10 setup — downgraded to `jupyter_client<8`
- The DAG file referenced `tfx.orchestration.metadata.sqlite_metadata_connection_config` but hadn't imported `tfx` — added the missing import

**Step 5 — Verify DAG Loads**

Ran `airflow dags list` and confirmed `bank_marketing_pipeline` appeared in the DAG list without import errors.

**Step 6 — Generate DAG Graph**

Used `airflow dags show bank_marketing_pipeline --save airflow_dag_graph.png` to generate a PNG of the DAG graph showing all 9 TFX components and their dependency relationships. This was saved directly to Google Drive.

**Step 7 — Run the DAG**

Used `airflow dags test bank_marketing_pipeline 2024-01-01` to execute all tasks sequentially. This is equivalent to triggering the DAG from the Airflow UI — it runs every task in dependency order and logs the results.

**Why CLI instead of the web UI:** The Airflow web UI (Gunicorn-based) consistently timed out on Colab's free tier — the worker would crash after 120 seconds before the UI finished loading. The CLI commands produce identical results without needing the webserver running.

### Output

```
✅ DAG graph saved to outputs/screenshots/airflow_dag_graph.png

Running DAG...
Marking task as SUCCESS: CsvExampleGen
Marking task as SUCCESS: StatisticsGen
Marking task as SUCCESS: SchemaGen
Marking task as SUCCESS: ExampleValidator
Marking task as SUCCESS: Transform
Marking task as SUCCESS: Trainer
Marking task as SUCCESS: Resolver
Marking task as SUCCESS: Evaluator
Marking task as SUCCESS: Pusher
DagRun Finished: state:success

✅ DAG RUN SUCCESSFUL
Full log saved to outputs/screenshots/airflow_dag_run_log.txt
```

### Deliverables Produced

```
outputs/screenshots/
├── airflow_dag_graph.png       ← Visual graph of all 9 components
└── airflow_dag_run_log.txt     ← Complete Airflow execution log
```

---

## Environment Challenges & Solutions

### Problem: TFX Won't Install on Colab (Python 3.12)

`ml-metadata`, a C++ compiled dependency, has no wheel for Python 3.12. Every version of TFX that exists requires Python ≤3.10 for this package.

**Solution:** Created a Python 3.10 virtual environment inside Colab using Ubuntu 22.04's built-in Python 3.10, bootstrapped pip manually with `get-pip.py`, and ran all TFX code through the venv via subprocess calls.

### Problem: pip Resolver Infinite Loop

TFX depends on Apache Beam, which depends on 50+ Google Cloud packages. pip's dependency resolver tried to check every version combination and hit `resolution-too-deep` after 15+ minutes of backtracking.

**Solution:** Pre-installed every package with exact pinned versions using `--no-deps`, bypassing pip's resolver entirely. Then installed missing transitive dependencies one by one as import errors surfaced (`nbformat`, `defusedxml`, `ipykernel`, `google-api-python-client`, etc.).

### Problem: InteractiveContext is a No-Op in Subprocess

TFX's `InteractiveContext.run()` only works inside an IPython/Jupyter kernel. Since our Python 3.10 code runs as a subprocess (not in Colab's kernel), `context.run()` silently does nothing — no error, no output, just `WARNING: Method "run" is a no-op when invoked outside of IPython`.

**Solution:** Switched to `LocalDagRunner` which executes the pipeline as a proper DAG without requiring a notebook kernel. The pipeline definition is identical — only the runner changes.

### Problem: Package Version Overwrites

Installing `google-api-python-client`, `kubernetes`, and other packages without `--no-deps` pulled in `protobuf 6.x`, `grpcio 1.83`, and `attrs 26.x` — all incompatible with TFX 1.15.0 (which needs `protobuf <5`, `grpcio ~1.62`, `attrs <24`).

**Solution:** Added a "fix overwrites" step that force-reinstalls the correct versions with `--no-deps` after every batch of package installations.

### Problem: Airflow Webserver Timeout

Gunicorn (Airflow's webserver) spawns 4 worker processes by default. On Colab's free tier, the workers timeout (120s) before they finish initializing, crashing the webserver.

**Solution:** Bypassed the webserver entirely. Used `airflow dags show` for the DAG graph and `airflow dags test` for execution — both run from the CLI without needing the web UI.

---

## Files I Created

| File | Purpose |
|------|---------|
| `modules/transform_module.py` | Feature engineering: z-score scaling, vocabulary encoding, age bucketization |
| `modules/trainer_module.py` | Keras model definition, training loop, TensorBoard callback, serving signature |
| `pipeline/pipeline.py` | `create_pipeline()` function bundling all 9 components |
| `dags/bank_marketing_dag.py` | Airflow DAG definition using AirflowDagRunner |
| `notebooks/phase1-3_pipeline.ipynb` | Complete Colab notebook running everything |
| `requirements.txt` | Pinned package versions for reproducibility |
| `data/processed/bank_marketing.csv` | Preprocessed dataset (comma-delimited, numeric labels) |

---

## What My Groupmates Received

After Phases 1–3, I shared the following on Google Drive for my groupmates to complete their portions:

**For TFMA Analysis (Phase 4):**
- `outputs/artifacts/eval_run_1/` — evaluation results from 500-step run
- `outputs/artifacts/eval_run_2/` — evaluation results from 1000-step run
- Both ready to load with `tfma.load_eval_result(path)`

**For TensorBoard (Phase 5):**
- `outputs/artifacts/model_run_1/` — training logs from 500-step run
- `outputs/artifacts/model_run_2/` — training logs from 1000-step run

**For What-If Tool (Phase 6):**
- `outputs/artifacts/trained_model/` — exported SavedModel
- Eval TFRecords at known paths in the pipeline directory

**For the Report:**
- All screenshots from Phase 1–3 execution
- DAG graph PNG and run log
- Model training output showing accuracy, AUC, and architecture
