# Bank Marketing ML Pipeline — TFX & Apache Airflow

An end-to-end machine learning pipeline built with **TensorFlow Extended (TFX)** and **Apache Airflow** that predicts whether a bank client will subscribe to a term deposit. Built as part of COMP315 (AI Software Test & ML/Ops) at Centennial College.

## Problem

A Portuguese bank ran direct marketing campaigns via phone calls. The goal is to predict whether a client will subscribe to a term deposit (`yes`/`no`) based on demographic, financial, and campaign-related features. The dataset is imbalanced — 88% negative, 12% positive — which makes slice-level evaluation critical.

**Dataset:** [Bank Marketing Dataset](http://hdl.handle.net/1822/14838) (Moro et al., 2011) — 45,211 records, 16 features

## Pipeline Architecture

```
CSV Data
  │
  ▼
ExampleGen ──► StatisticsGen ──► SchemaGen ──► ExampleValidator
                                                     │
                                                     ▼
                                               Transform
                                                     │
                                                     ▼
                                                Trainer ──► Resolver
                                                     │          │
                                                     ▼          ▼
                                                Evaluator (TFMA)
                                                     │
                                                     ▼
                                                  Pusher
                                                     │
                                                     ▼
                                              Serving Directory
```

Each component is a standalone, reusable unit managed by ML Metadata (MLMD). The pipeline runs on Apache Airflow in production and via `LocalDagRunner` for development.

## Components

| Step | Component | Purpose |
|------|-----------|---------|
| 1 | **ExampleGen** | Ingests CSV, splits into train/eval (2:1), converts to TFRecords |
| 2 | **StatisticsGen** | Computes per-feature statistics (min, max, mean, distributions) |
| 3 | **SchemaGen** | Infers data schema (types, ranges, vocabularies) |
| 4 | **ExampleValidator** | Flags anomalies by comparing statistics against schema |
| 5 | **Transform** | Feature engineering — z-score scaling, vocabulary encoding, age bucketization |
| 6 | **Trainer** | Trains a Keras neural network with embeddings, batch normalization, dropout |
| 7 | **Resolver** | Retrieves the latest blessed model as a baseline for comparison |
| 8 | **Evaluator** | Runs TFMA with slicing by education, marital status, and job type |
| 9 | **Pusher** | Deploys the model to a serving directory if blessed |

## Model Architecture

```
Numeric features (7) ──────────────────────┐
                                           │
Categorical features (9) → Embedding → Flatten ─┤
                                           │
Age bucket → Embedding → Flatten ──────────┤
                                           │
                                    Concatenate
                                           │
                                   Dense(128) + BatchNorm + Dropout(0.3)
                                           │
                                   Dense(64) + BatchNorm + Dropout(0.2)
                                           │
                                       Dense(32)
                                           │
                                   Dense(1, sigmoid)
```

**Features:**
- 7 numeric features scaled to z-scores: `age`, `balance`, `day`, `duration`, `campaign`, `pdays`, `previous`
- 9 categorical features encoded via vocabulary lookup + embeddings: `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`
- 1 engineered feature: `age_bucket` (5 bins)

**Results (Run 1):** 90.9% binary accuracy, 0.89 AUC on validation set

## TFMA Evaluation

Model performance is evaluated across multiple data slices to detect fairness gaps:

- **Overall** dataset metrics
- **Education** slices (primary, secondary, tertiary, unknown)
- **Marital status** slices (married, single, divorced)
- **Job type** slices (12 categories)

Metrics tracked: BinaryAccuracy, AUC, ExampleCount

## Project Structure

```
COMP315_Project/
├── data/
│   ├── bank-full.csv              # Original dataset (semicolon-delimited)
│   └── processed/
│       └── bank_marketing.csv     # Preprocessed (comma-delimited, numeric labels)
├── modules/
│   ├── transform_module.py        # preprocessing_fn for feature engineering
│   └── trainer_module.py          # run_fn with Keras model and TensorBoard
├── pipeline/
│   └── pipeline.py                # create_pipeline() function (Step 10)
├── dags/
│   └── comp315_dag.py             # Airflow DAG definition
├── notebooks/
│   └── phase1_2_pipeline.ipynb    # Colab notebook running all 9 steps
├── outputs/
│   ├── artifacts/
│   │   ├── eval_run_1/            # TFMA evaluation results
│   │   └── model_run_1/           # TensorBoard training logs
│   └── screenshots/
├── report/
├── requirements.txt
└── README.md
```

## Setup & Reproduction

### Prerequisites

TFX 1.15.0 requires **Python 3.10**. On Google Colab (which runs Python 3.12), a Python 3.10 virtual environment is needed.

### Google Colab Setup

```bash
# Install Python 3.10 venv
sudo apt-get update -qq
sudo apt install python3.10-venv -y -qq
python3.10 -m venv /content/tfx_env --without-pip
curl -sS https://bootstrap.pypa.io/get-pip.py | /content/tfx_env/bin/python3.10

# Install TFX and dependencies (pinned versions, --no-deps to avoid resolver issues)
/content/tfx_env/bin/pip install --no-deps \
  tfx==1.15.0 tensorflow==2.15.0 ml-pipelines-sdk==1.15.0 \
  tensorflow-data-validation==1.15.1 tensorflow-model-analysis==0.46.0 \
  tensorflow-transform==1.15.0 tfx-bsl==1.15.1 ml-metadata==1.15.0 \
  tensorflow-metadata==1.15.0 keras==2.15.0
# ... (see notebook for full dependency list)
```

### Data Preprocessing

```python
import pandas as pd

df = pd.read_csv('data/bank-full.csv', sep=';')
df['y'] = df['y'].map({'yes': 1, 'no': 0})
df.to_csv('data/processed/bank_marketing.csv', index=False)
```

The original CSV uses semicolons and text labels — TFX requires commas and numeric labels.

### Running the Pipeline

```python
from tfx.orchestration.local.local_dag_runner import LocalDagRunner
from tfx.orchestration import pipeline as pipeline_module

# ... (define components)

p = pipeline_module.Pipeline(
    pipeline_name="bank_marketing_pipeline",
    pipeline_root="pipelines/bank_marketing_pipeline",
    components=components,
    metadata_connection_config=tfx.orchestration.metadata
        .sqlite_metadata_connection_config(metadata_path),
)

LocalDagRunner().run(p)
```

## Tech Stack

- **TFX 1.15.0** — ML pipeline framework
- **TensorFlow 2.15.0** — Model training
- **Apache Beam 2.56.0** — Data processing backend
- **Apache Airflow** — Pipeline orchestration
- **TFMA 0.46.0** — Model evaluation and fairness analysis
- **TFDV 1.15.1** — Data validation
- **TensorFlow Transform 1.15.0** — Feature engineering
- **ML Metadata 1.15.0** — Artifact tracking
- **Python 3.10** — Runtime

## Key Learnings

- **Data validation catches what manual inspection misses.** ExampleValidator automates anomaly detection across every feature — a step that's easy to skip but critical in production.
- **Slice-level evaluation reveals what overall accuracy hides.** A model can be 90% accurate overall but perform poorly on underrepresented groups. TFMA slicing makes this visible.
- **Transform preprocessing baked into the SavedModel eliminates training/serving skew.** The same feature engineering runs at training time and prediction time automatically.
- **Dependency management for TFX is nontrivial.** TFX 1.15.0 has strict version pins across its ecosystem. Python version, protobuf version, and gRPC version must all align — a real-world lesson in ML infrastructure.

## References

Moro, S., Laureano, R., & Cortez, P. (2011). Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology. In P. Novais et al. (Eds.), *Proceedings of the European Simulation and Modelling Conference — ESM'2011*, pp. 117-121, Guimarães, Portugal. EUROSIS.

## Authors

COMP315 AI Software Test & ML/Ops — Centennial College, Fall 2026
