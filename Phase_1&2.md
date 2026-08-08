# COMP315 — Phases 1 & 2 Full Implementation Guide
## Bank Marketing Dataset (Term Deposit Prediction)

---

## Your Dataset at a Glance

| Property | Value |
|----------|-------|
| File | `bank-full.csv` (45,211 rows) |
| Goal | Predict if client subscribes a term deposit (`y`: yes/no) |
| Features | 7 numeric, 9 categorical |
| Label balance | 88.3% no, 11.7% yes (imbalanced) |
| Delimiter | Semicolon (`;`) — must convert to comma for TFX |

**Numeric features:** age, balance, day, duration, campaign, pdays, previous
**Categorical features:** job (12 vals), marital (3), education (4), default (2), housing (2), loan (2), contact (3), month (12), poutcome (4)

---

## Step 0: Data Preprocessing

TFX's `CsvExampleGen` expects comma-separated CSVs with numeric labels. Your data uses semicolons and the label is text. This script fixes both.

### `preprocess_data.py`

```python
"""
Preprocess bank-full.csv for TFX pipeline.
- Converts semicolon delimiter to comma
- Converts label y: yes/no → 1/0
- Saves to data/ folder for CsvExampleGen
"""
import pandas as pd
import os

# Read the semicolon-delimited CSV
df = pd.read_csv('bank-full.csv', sep=';')

# Convert label to numeric (required for TFX binary classification)
df['y'] = df['y'].map({'yes': 1, 'no': 0})

# Verify conversion
print(f"Shape: {df.shape}")
print(f"Label distribution:\n{df['y'].value_counts()}")
print(f"Any nulls: {df.isnull().any().any()}")

# Save as comma-separated CSV
os.makedirs('data', exist_ok=True)
df.to_csv('data/bank_marketing.csv', index=False)

print(f"\nSaved to data/bank_marketing.csv")
```

**Run it:**
```bash
python preprocess_data.py
```

Your folder structure should now look like:
```
project/
├── data/
│   └── bank_marketing.csv     ← CsvExampleGen reads from this folder
├── modules/
│   ├── transform_module.py    ← Step 5
│   └── trainer_module.py      ← Step 6
├── pipeline.py                ← Step 10
├── preprocess_data.py
└── dags/
    └── comp315_dag.py         ← Airflow DAG
```

---

## Step 1: ExampleGen — Ingest Your Data

**What it does:** Reads the CSV from the `data/` folder, splits it into train (2/3) and eval (1/3) sets, and converts everything into TFRecord format. TFRecords are TensorFlow's optimized binary storage — they're faster to read during training than CSV.

**Why you don't split manually:** TFX handles the split for you deterministically, and records the split metadata in ML Metadata (MLMD) so every downstream component knows exactly which data it's working with. This reproducibility is the whole point of a managed pipeline.

### `pipeline_interactive.py` (Steps 1–9 in one notebook-ready script)

```python
"""
COMP315 Term Project — Interactive Pipeline Runner
Bank Marketing Dataset: Predict term deposit subscription
Run this in Google Colab or a Jupyter notebook.
"""

import os
import tensorflow as tf
import tfx
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext

# ──────────────────────────────────────────────
# CONFIGURATION — adjust these paths as needed
# ──────────────────────────────────────────────
_pipeline_name = 'bank_marketing_pipeline'
_project_root = os.getcwd()
_pipeline_root = os.path.join(_project_root, 'pipelines', _pipeline_name)
_data_root = os.path.join(_project_root, 'data')  # folder containing bank_marketing.csv
_metadata_path = os.path.join(_project_root, 'metadata', _pipeline_name, 'metadata.db')
_serving_model_dir = os.path.join(_project_root, 'serving_model', _pipeline_name)
_transform_module_file = os.path.join(_project_root, 'modules', 'transform_module.py')
_trainer_module_file = os.path.join(_project_root, 'modules', 'trainer_module.py')

# Create directories
for d in [_pipeline_root, os.path.dirname(_metadata_path), _serving_model_dir,
          os.path.dirname(_transform_module_file)]:
    os.makedirs(d, exist_ok=True)

print(f"TFX version: {tfx.__version__}")
print(f"TensorFlow version: {tf.__version__}")
print(f"Data root: {_data_root}")
print(f"Pipeline root: {_pipeline_root}")

# ──────────────────────────────────────────────
# INTERACTIVE CONTEXT
# This lets you run each TFX component one at a time in a notebook.
# In production you'd use Airflow instead, but the components are identical.
# ──────────────────────────────────────────────
context = InteractiveContext(
    pipeline_name=_pipeline_name,
    pipeline_root=_pipeline_root,
    metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(
        _metadata_path
    )
)

# ════════════════════════════════════════════════
# STEP 1: ExampleGen — Data Ingestion
# ════════════════════════════════════════════════
from tfx.components import CsvExampleGen

example_gen = CsvExampleGen(input_base=_data_root)
context.run(example_gen)

# Display the URIs of the data splits (required by assignment)
print("\n--- ExampleGen Output Artifacts ---")
for artifact in example_gen.outputs['examples'].get():
    print(f"  Artifact URI: {artifact.uri}")
    print(f"  Split names: {artifact.split_names}")


# ════════════════════════════════════════════════
# STEP 2: StatisticsGen — Compute Data Statistics
# ════════════════════════════════════════════════
from tfx.components import StatisticsGen

statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
context.run(statistics_gen)

# Render interactive statistics visualization
# This shows histograms, means, min/max, missing counts for every feature
context.show(statistics_gen.outputs['statistics'])

print("\n--- StatisticsGen Output Artifacts ---")
for artifact in statistics_gen.outputs['statistics'].get():
    print(f"  Statistics URI: {artifact.uri}")


# ════════════════════════════════════════════════
# STEP 3: SchemaGen — Infer Data Schema
# ════════════════════════════════════════════════
from tfx.components import SchemaGen

schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'])
context.run(schema_gen)

# Render the inferred schema
context.show(schema_gen.outputs['schema'])

print("\n--- SchemaGen Output Artifacts ---")
for artifact in schema_gen.outputs['schema'].get():
    print(f"  Schema URI: {artifact.uri}")

# ─── Schema Summary (for the report) ───
# After viewing the rendered schema, write your summary.
# Expected schema for this dataset:
#   Numeric (INT/FLOAT): age, balance, day, duration, campaign, pdays, previous, y
#   Categorical (STRING): job, marital, education, default, housing, loan, contact, month, poutcome
#   Note: "day" is numeric (1-31) but represents day-of-month, could also be bucketized
#   Note: "pdays" has value -1 meaning "not previously contacted" — a special sentinel value


# ════════════════════════════════════════════════
# STEP 4: ExampleValidator — Detect Anomalies
# ════════════════════════════════════════════════
from tfx.components import ExampleValidator

example_validator = ExampleValidator(
    statistics=statistics_gen.outputs['statistics'],
    schema=schema_gen.outputs['schema']
)
context.run(example_validator)

# Render anomalies visualization
context.show(example_validator.outputs['anomalies'])

# Print whether data pipeline is healthy
print("\n--- Data Validation Results ---")
import tensorflow_data_validation as tfdv

anomalies_artifact = example_validator.outputs['anomalies'].get()[0]
anomalies = tfdv.load_anomalies_text(
    os.path.join(anomalies_artifact.uri, 'SchemaDiff.pb')
)

if len(anomalies.anomaly_info) == 0:
    print("✅ Data pipeline is HEALTHY — no anomalies detected.")
else:
    print("⚠️  Anomalies detected:")
    for feature_name, anomaly_info in anomalies.anomaly_info.items():
        print(f"  Feature: {feature_name}")
        print(f"    Description: {anomaly_info.description}")
        print(f"    Severity: {anomaly_info.severity}")


# ════════════════════════════════════════════════
# STEP 5: Transform — Feature Engineering
# ════════════════════════════════════════════════
from tfx.components import Transform

# The Transform component reads the module file you created separately
# (see modules/transform_module.py below)
transform = Transform(
    examples=example_gen.outputs['examples'],
    schema=schema_gen.outputs['schema'],
    module_file=_transform_module_file
)
context.run(transform)

print("\n--- Transform Output Artifacts ---")
print(f"  Transformed examples: {transform.outputs['transformed_examples'].get()[0].uri}")
print(f"  Transform graph: {transform.outputs['transform_graph'].get()[0].uri}")


# ════════════════════════════════════════════════
# STEP 6: Trainer — Train the Model
# ════════════════════════════════════════════════
from tfx.components import Trainer
from tfx.proto import trainer_pb2

trainer = Trainer(
    module_file=_trainer_module_file,
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    schema=schema_gen.outputs['schema'],
    train_args=trainer_pb2.TrainArgs(num_steps=500),  # Adjust for second run
    eval_args=trainer_pb2.EvalArgs(num_steps=100)
)
context.run(trainer)

print("\n--- Trainer Output Artifacts ---")
print(f"  Model URI: {trainer.outputs['model'].get()[0].uri}")
print(f"  Model run URI: {trainer.outputs['model_run'].get()[0].uri}")
# ↑ model_run URI is where TensorBoard logs are stored


# ════════════════════════════════════════════════
# STEP 7: Resolver — Find Latest Blessed Model
# ════════════════════════════════════════════════
from tfx.components import Resolver
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy,
)

model_resolver = Resolver(
    strategy_class=LatestBlessedModelStrategy,
    model=Channel(type=Model),
    model_blessing=Channel(type=ModelBlessing)
).with_id('latest_blessed_model_resolver')
context.run(model_resolver)

# On first run, there's no blessed model yet — that's expected.
# The Evaluator will auto-bless the first model.
print("\n--- Resolver Output ---")
try:
    baseline = model_resolver.outputs['model'].get()[0]
    print(f"  Baseline model URI: {baseline.uri}")
except IndexError:
    print("  No baseline model found (first run) — Evaluator will auto-bless.")


# ════════════════════════════════════════════════
# STEP 8: Evaluator — Evaluate with TFMA
# ════════════════════════════════════════════════
import tensorflow_model_analysis as tfma
from tfx.components import Evaluator

eval_config = tfma.EvalConfig(
    model_specs=[
        tfma.ModelSpec(label_key='y')
    ],
    slicing_specs=[
        # Overall dataset (no slicing)
        tfma.SlicingSpec(),
        # Slice by education level — interesting because education correlates
        # with income and financial product adoption
        tfma.SlicingSpec(feature_keys=['education']),
        # Slice by marital status — relevant for fairness analysis
        tfma.SlicingSpec(feature_keys=['marital']),
        # Slice by job type — see if model performs differently across occupations
        tfma.SlicingSpec(feature_keys=['job']),
    ],
    metrics_specs=[
        tfma.MetricsSpec(metrics=[
            tfma.MetricConfig(class_name='BinaryAccuracy'),
            tfma.MetricConfig(class_name='AUC'),
            # ExampleCount helps you see how many examples are in each slice
            tfma.MetricConfig(class_name='ExampleCount'),
        ])
    ],
    options=tfma.Options(
        compute_confidence_intervals=True
    )
)

evaluator = Evaluator(
    examples=example_gen.outputs['examples'],
    model=trainer.outputs['model'],
    baseline_model=model_resolver.outputs['model'],
    eval_config=eval_config
)
context.run(evaluator)

print("\n--- Evaluator Output ---")
eval_output = evaluator.outputs['evaluation'].get()[0]
print(f"  Evaluation URI: {eval_output.uri}")
blessing = evaluator.outputs['blessing'].get()[0]
print(f"  Blessing URI: {blessing.uri}")

# Check if model was blessed
if os.path.exists(os.path.join(blessing.uri, 'BLESSED')):
    print("  ✅ Model was BLESSED (approved for serving)")
else:
    print("  ❌ Model was NOT blessed (did not beat baseline)")


# ════════════════════════════════════════════════
# STEP 9: Pusher — Deploy the Blessed Model
# ════════════════════════════════════════════════
from tfx.components import Pusher
from tfx.proto import pusher_pb2

pusher = Pusher(
    model=trainer.outputs['model'],
    model_blessing=evaluator.outputs['blessing'],
    push_destination=pusher_pb2.PushDestination(
        filesystem=pusher_pb2.PushDestination.Filesystem(
            base_directory=_serving_model_dir
        )
    )
)
context.run(pusher)

print("\n--- Pusher Output ---")
push_artifact = pusher.outputs['pushed_model'].get()[0]
print(f"  Pushed model URI: {push_artifact.uri}")
print(f"  Serving directory: {_serving_model_dir}")

# ════════════════════════════════════════════════
# DONE — Print summary of all models created
# ════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PIPELINE RUN COMPLETE — Summary")
print("=" * 60)
print(f"  Data ingested from:    {_data_root}")
print(f"  Pipeline artifacts:    {_pipeline_root}")
print(f"  ML Metadata DB:        {_metadata_path}")
print(f"  Trained model:         {trainer.outputs['model'].get()[0].uri}")
print(f"  Evaluation results:    {eval_output.uri}")
print(f"  Serving model:         {_serving_model_dir}")
print(f"  TensorBoard logs:      {trainer.outputs['model_run'].get()[0].uri}")
print()
print("Next steps:")
print("  1. Run TensorBoard:  tensorboard --logdir <model_run_uri>")
print("  2. Run the pipeline again with different hyperparameters for TFMA comparison")
print("  3. Load eval results in TFMA notebook for analysis")
```

---

## Module File: `modules/transform_module.py`

```python
"""
Transform module for the Bank Marketing TFX pipeline.
Defines preprocessing_fn() which specifies how raw features are transformed
before training. These transformations get baked into the SavedModel's serving
graph, so the same preprocessing runs at prediction time — no training/serving skew.

Dataset features:
  Numeric:     age, balance, day, duration, campaign, pdays, previous
  Categorical: job, marital, education, default, housing, loan, contact, month, poutcome
  Label:       y (0 or 1)
"""

import tensorflow as tf
import tensorflow_transform as tft

# ── Feature lists ──
# These must match the column names in your preprocessed CSV exactly.

NUMERIC_FEATURES = [
    'age',       # client age
    'balance',   # average yearly balance in euros
    'day',       # last contact day of month (1-31)
    'duration',  # last contact duration in seconds
    'campaign',  # contacts during this campaign
    'pdays',     # days since last contact from previous campaign (-1 = never)
    'previous',  # contacts before this campaign
]

CATEGORICAL_FEATURES = [
    'job',        # 12 categories (admin., blue-collar, technician, etc.)
    'marital',    # 3 categories (married, single, divorced)
    'education',  # 4 categories (primary, secondary, tertiary, unknown)
    'default',    # 2 categories (yes, no) — has credit in default?
    'housing',    # 2 categories (yes, no) — has housing loan?
    'loan',       # 2 categories (yes, no) — has personal loan?
    'contact',    # 3 categories (cellular, telephone, unknown)
    'month',      # 12 categories (jan, feb, ..., dec)
    'poutcome',   # 4 categories (success, failure, other, unknown)
]

LABEL_KEY = 'y'


def preprocessing_fn(inputs):
    """
    Transform raw features into model-ready format.

    What each transformation does and why:
    - scale_to_z_score: Centers numeric features around 0 with std=1.
      Neural networks train better when features are on similar scales.
    - compute_and_apply_vocabulary: Converts string categories to integer IDs.
      Neural networks can't consume strings — they need numbers.
      top_k limits the vocabulary to the most frequent values.
      num_oov_buckets=1 creates a catch-all bucket for unseen categories at serving time.
    - bucketize: Splits a numeric range into discrete bins.
      Used for 'age' here because the relationship between age and
      term deposit subscription isn't linear — it's grouped (students, working age, retired).
    """
    outputs = {}

    # Scale all numeric features to z-scores (mean=0, std=1)
    for feature in NUMERIC_FEATURES:
        outputs[feature] = tft.scale_to_z_score(inputs[feature])

    # Convert categorical strings to integer indices via vocabulary lookup
    for feature in CATEGORICAL_FEATURES:
        outputs[feature] = tft.compute_and_apply_vocabulary(
            inputs[feature],
            top_k=100,         # keep top 100 most frequent values (more than enough)
            num_oov_buckets=1  # 1 bucket for out-of-vocabulary values at serving time
        )

    # Bucketize age into 5 bins (adds an extra feature the model can use)
    # Bins will be roughly: 18-30, 30-40, 40-50, 50-60, 60+
    outputs['age_bucket'] = tft.bucketize(inputs['age'], num_buckets=5)

    # Pass through the label unchanged
    outputs[LABEL_KEY] = inputs[LABEL_KEY]

    return outputs
```

---

## Module File: `modules/trainer_module.py`

```python
"""
Trainer module for the Bank Marketing TFX pipeline.
Defines run_fn() which builds a Keras model, trains it, and saves it
in SavedModel format with a serving signature.

The serving signature is important — it means the saved model can accept
raw serialized tf.Examples (the format TFX uses) and internally applies
the Transform preprocessing before making predictions. This is how you
avoid training/serving skew in production.
"""

import os
from typing import List

import tensorflow as tf
from tensorflow import keras
import tensorflow_transform as tft
from tfx_bsl.public import tfxio

# ── Feature configuration (must match transform_module.py) ──

NUMERIC_FEATURES = [
    'age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous',
]

CATEGORICAL_FEATURES = [
    'job', 'marital', 'education', 'default', 'housing', 'loan',
    'contact', 'month', 'poutcome',
]

# Max vocabulary sizes per categorical feature.
# These are the number of unique values + 1 for the OOV bucket.
# If unsure, set them higher than needed — embeddings will just have unused rows.
VOCAB_SIZES = {
    'job': 13,        # 12 categories + 1 OOV
    'marital': 4,     # 3 + 1
    'education': 5,   # 4 + 1
    'default': 3,     # 2 + 1
    'housing': 3,     # 2 + 1
    'loan': 3,        # 2 + 1
    'contact': 4,     # 3 + 1
    'month': 13,      # 12 + 1
    'poutcome': 5,    # 4 + 1
}

# Extra engineered feature from Transform
BUCKET_FEATURES = ['age_bucket']
BUCKET_SIZES = {'age_bucket': 6}  # 5 buckets + 1 (bucketize is 0-indexed)

LABEL_KEY = 'y'


def _input_fn(
    file_pattern: List[str],
    data_accessor,
    tf_transform_output: tft.TFTransformOutput,
    batch_size: int = 64,
) -> tf.data.Dataset:
    """
    Creates a tf.data.Dataset from transformed TFRecord files.
    data_accessor handles the low-level deserialization — you don't need to
    parse the TFRecords manually.
    """
    return data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(
            batch_size=batch_size,
            label_key=LABEL_KEY,
        ),
        tf_transform_output.transformed_metadata.schema,
    )


def _build_model(tf_transform_output: tft.TFTransformOutput) -> keras.Model:
    """
    Builds a Keras model for binary classification.

    Architecture:
    - Each numeric feature enters as a single float input
    - Each categorical feature enters as an integer, gets embedded into a dense vector
    - The bucketized age also gets embedded
    - All are concatenated → Dense(128) → Dropout → Dense(64) → Dropout → sigmoid output

    Why embeddings instead of one-hot encoding?
    One-hot for 'job' (12 categories) creates a sparse 12-dim vector.
    An embedding maps it to a dense 8-dim vector where similar jobs can end up
    near each other. This gives the model more capacity to learn relationships
    between categories while using fewer parameters.
    """
    inputs = {}
    encoded_features = []

    # Numeric features — each is a single z-scored float
    for feature in NUMERIC_FEATURES:
        inp = keras.layers.Input(shape=(1,), name=feature, dtype=tf.float32)
        inputs[feature] = inp
        encoded_features.append(inp)

    # Categorical features — integer IDs → embedding → flatten
    for feature in CATEGORICAL_FEATURES:
        inp = keras.layers.Input(shape=(1,), name=feature, dtype=tf.int64)
        inputs[feature] = inp
        vocab_size = VOCAB_SIZES[feature]
        # Embedding dimension: min(50, vocab_size // 2) is a common heuristic
        embed_dim = min(8, max(2, vocab_size // 2))
        embedding = keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )(inp)
        embedding = keras.layers.Flatten()(embedding)
        encoded_features.append(embedding)

    # Bucketized features — same treatment as categorical
    for feature in BUCKET_FEATURES:
        inp = keras.layers.Input(shape=(1,), name=feature, dtype=tf.int64)
        inputs[feature] = inp
        bucket_size = BUCKET_SIZES[feature]
        embedding = keras.layers.Embedding(
            input_dim=bucket_size, output_dim=4
        )(inp)
        embedding = keras.layers.Flatten()(embedding)
        encoded_features.append(embedding)

    # Concatenate all encoded features
    x = keras.layers.Concatenate()(encoded_features)

    # Hidden layers with dropout for regularization
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Dense(32, activation='relu')(x)

    # Sigmoid output for binary classification
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            keras.metrics.BinaryAccuracy(name='binary_accuracy'),
            keras.metrics.AUC(name='auc'),
        ],
    )

    model.summary()
    return model


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """
    Creates a serving function that accepts raw serialized tf.Examples.

    Why this matters: When this model is deployed (via Pusher or TF Serving),
    clients send raw feature data. This function ensures the Transform
    preprocessing is applied automatically before the model makes predictions.
    Without this, you'd have to reimplement all the preprocessing logic
    in your serving infrastructure — a common source of bugs called
    "training/serving skew."
    """
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def serve_tf_examples_fn(serialized_tf_examples):
        # Parse raw features from the serialized examples
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)  # Remove label — not available at serving time
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

        # Apply the same transformations used during training
        transformed_features = model.tft_layer(parsed_features)

        # Run the model
        return model(transformed_features)

    return serve_tf_examples_fn


def run_fn(fn_args):
    """
    Entry point called by the TFX Trainer component.
    fn_args contains paths to data, transform outputs, and configuration
    set by the Trainer component.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    # Build datasets
    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        tf_transform_output,
        batch_size=64,
    )
    eval_dataset = _input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        tf_transform_output,
        batch_size=64,
    )

    # Build the model
    model = _build_model(tf_transform_output)

    # ── Callbacks ──

    # TensorBoard — REQUIRED by the assignment
    # Logs training metrics so you can visualize loss/accuracy curves
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=fn_args.model_run_dir,
        update_freq='batch',  # log every batch (vs every epoch)
    )

    # Early stopping — stops training if validation loss stops improving
    # patience=5 means "wait 5 evaluation rounds before stopping"
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
    )

    # Train the model
    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback, early_stopping],
    )

    # Save in SavedModel format with the serving signature
    signatures = {
        'serving_default': _get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
        ),
    }
    model.save(
        fn_args.serving_model_dir,
        save_format='tf',
        signatures=signatures,
    )

    print(f"\nModel saved to: {fn_args.serving_model_dir}")
```

---

## Step 10: Pipeline Definition Function

### `pipeline.py`

```python
"""
Pipeline definition for the Bank Marketing TFX pipeline.
This function bundles all 9 components into a single Pipeline object
that can be run by any TFX orchestrator (Airflow, Beam, Vertex, etc.).

Keeping this as a standalone function (not a script) is a TFX best practice —
it decouples the pipeline definition from the orchestrator. You can test it
locally with InteractiveContext, then deploy the exact same pipeline to
Airflow without changing any component code.
"""

import os
from typing import Optional

import tensorflow_model_analysis as tfma
import tfx.v1 as tfx
from tfx.components import (
    CsvExampleGen,
    Evaluator,
    ExampleValidator,
    Pusher,
    Resolver,
    SchemaGen,
    StatisticsGen,
    Trainer,
    Transform,
)
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy,
)
from tfx.proto import pusher_pb2, trainer_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing


def create_pipeline(
    pipeline_name: str,
    pipeline_root: str,
    data_root: str,
    transform_module_file: str,
    trainer_module_file: str,
    serving_model_dir: str,
    metadata_path: str,
    train_steps: int = 500,
    eval_steps: int = 100,
) -> tfx.dsl.Pipeline:
    """
    Creates and returns a TFX pipeline for the Bank Marketing dataset.

    Args:
        pipeline_name: Name used in MLMD and Airflow
        pipeline_root: Root directory for pipeline artifacts
        data_root: Folder containing bank_marketing.csv
        transform_module_file: Path to transform_module.py
        trainer_module_file: Path to trainer_module.py
        serving_model_dir: Where Pusher copies blessed models
        metadata_path: Path to the SQLite MLMD database
        train_steps: Number of training steps (change for second run)
        eval_steps: Number of evaluation steps
    """

    # Step 1: Ingest data
    example_gen = CsvExampleGen(input_base=data_root)

    # Step 2: Compute statistics
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

    # Step 3: Infer schema
    schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'])

    # Step 4: Validate data
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema'],
    )

    # Step 5: Feature engineering
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=transform_module_file,
    )

    # Step 6: Train model
    trainer = Trainer(
        module_file=trainer_module_file,
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(num_steps=train_steps),
        eval_args=trainer_pb2.EvalArgs(num_steps=eval_steps),
    )

    # Step 7: Resolve latest blessed model for baseline comparison
    model_resolver = Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing),
    ).with_id('latest_blessed_model_resolver')

    # Step 8: Evaluate with TFMA
    eval_config = tfma.EvalConfig(
        model_specs=[
            tfma.ModelSpec(label_key='y')
        ],
        slicing_specs=[
            tfma.SlicingSpec(),  # overall
            tfma.SlicingSpec(feature_keys=['education']),
            tfma.SlicingSpec(feature_keys=['marital']),
            tfma.SlicingSpec(feature_keys=['job']),
        ],
        metrics_specs=[
            tfma.MetricsSpec(metrics=[
                tfma.MetricConfig(class_name='BinaryAccuracy'),
                tfma.MetricConfig(class_name='AUC'),
                tfma.MetricConfig(class_name='ExampleCount'),
            ])
        ],
        options=tfma.Options(compute_confidence_intervals=True),
    )

    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config,
    )

    # Step 9: Push blessed model to serving directory
    pusher = Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir,
            )
        ),
    )

    # Assemble all components in dependency order
    components = [
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        trainer,
        model_resolver,
        evaluator,
        pusher,
    ]

    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        metadata_connection_config=(
            tfx.orchestration.metadata.sqlite_metadata_connection_config(metadata_path)
        ),
    )
```

---

## Airflow DAG File: `dags/comp315_dag.py`

```python
"""
Airflow DAG for the Bank Marketing TFX pipeline.
Place this file in your $AIRFLOW_HOME/dags/ directory.

What Airflow does here: It takes the same pipeline you ran interactively
and orchestrates it as a DAG (Directed Acyclic Graph). Each TFX component
becomes an Airflow task. Airflow handles scheduling, retries, dependency
resolution, and gives you a web UI to monitor everything.

schedule_interval=None means this DAG only runs when you manually trigger it.
In production you might set this to '@daily' or a cron expression.
"""

import os
from datetime import datetime

from tfx.orchestration.airflow.airflow_dag_runner import (
    AirflowDagRunner,
    AirflowPipelineConfig,
)

# Import your pipeline definition
from pipeline import create_pipeline

# ── Configuration ──
_pipeline_name = 'bank_marketing_pipeline'
_project_root = os.path.join(os.environ['HOME'], 'tfx_project')
_pipeline_root = os.path.join(_project_root, 'pipelines', _pipeline_name)
_data_root = os.path.join(_project_root, 'data')
_transform_module = os.path.join(_project_root, 'modules', 'transform_module.py')
_trainer_module = os.path.join(_project_root, 'modules', 'trainer_module.py')
_serving_model_dir = os.path.join(_project_root, 'serving_model', _pipeline_name)
_metadata_path = os.path.join(
    _project_root, 'metadata', _pipeline_name, 'metadata.db'
)

# Airflow-specific config
_airflow_config = {
    'schedule_interval': None,  # manual trigger only
    'start_date': datetime(2024, 1, 1),
    'catchup': False,
}

# Create the pipeline
_pipeline = create_pipeline(
    pipeline_name=_pipeline_name,
    pipeline_root=_pipeline_root,
    data_root=_data_root,
    transform_module_file=_transform_module,
    trainer_module_file=_trainer_module,
    serving_model_dir=_serving_model_dir,
    metadata_path=_metadata_path,
    train_steps=500,
    eval_steps=100,
)

# Convert TFX pipeline to Airflow DAG
DAG = AirflowDagRunner(
    AirflowPipelineConfig(airflow_dag_config=_airflow_config)
).run(_pipeline)
```

---

## Running the Second Pipeline (for TFMA Comparison)

After your first successful run, change hyperparameters and run again.
The second run is required for TFMA Step 3 (comparing two runs).

In `pipeline_interactive.py`, change these before re-running:

```python
# FIRST RUN (already done):
# train_args=trainer_pb2.TrainArgs(num_steps=500)
# eval_args=trainer_pb2.EvalArgs(num_steps=100)

# SECOND RUN — change at least one of:
trainer = Trainer(
    module_file=_trainer_module_file,
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    schema=schema_gen.outputs['schema'],
    train_args=trainer_pb2.TrainArgs(num_steps=1000),   # ← doubled
    eval_args=trainer_pb2.EvalArgs(num_steps=200)       # ← doubled
)
```

Or change the learning rate in `trainer_module.py`:
```python
# First run:  learning_rate=0.001
# Second run: learning_rate=0.0005
```

Save the evaluation artifact URIs from both runs — you'll need them
for the TFMA comparison notebook.

---

## Execution Order Checklist

```
□  1. Run preprocess_data.py to create data/bank_marketing.csv
□  2. Create modules/transform_module.py
□  3. Create modules/trainer_module.py
□  4. Run pipeline_interactive.py Steps 1-4 (data pipeline)
       □ Screenshot: StatisticsGen visualization
       □ Screenshot: SchemaGen visualization
       □ Screenshot: ExampleValidator anomalies
       □ Note the ExampleValidator results for the Data Validation Report
□  5. Run Step 5 (Transform) — verify no errors
□  6. Run Step 6 (Trainer) — verify model trains
       □ Note the model_run URI for TensorBoard
□  7. Run Step 7 (Resolver) — expect "no baseline" on first run
□  8. Run Step 8 (Evaluator) — verify model is blessed
       □ Note the evaluation URI for TFMA notebook
□  9. Run Step 9 (Pusher) — verify model copied to serving dir
□ 10. Save all artifact URIs printed by the summary
□ 11. Change hyperparameters and run Steps 6-9 again for the second run
       □ Save second evaluation URI for TFMA comparison
□ 12. Set up Airflow and get DAG screenshots
       □ Screenshot: DAG graph view
       □ Screenshot: All-green task run
       □ Screenshot: One task log with no errors
```
