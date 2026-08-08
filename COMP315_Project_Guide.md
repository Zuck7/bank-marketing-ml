C# COMP315 Term Project — Step-by-Step Completion Guide

## Overview

This guide walks your team of 3 through every step of the COMP315 TFX + Airflow ML pipeline project. It covers environment setup, all 10 pipeline steps, TFMA analysis, What-If Tool exploration, and every deliverable.

---

## Phase 0: Environment Setup (Do This First — Day 1)

### Why Colab?

TFX has heavy dependencies tied to x86 Linux. On Apple Silicon (ARM64), many TFX components fail to install or crash at runtime. Google Colab gives you a free x86 Linux VM with GPU access — it's the path of least resistance.

### 0.1 — Create a shared Google Drive folder

```
COMP315_Project/
├── data/               ← your assigned CSV dataset
├── notebooks/          ← Colab notebooks (.ipynb)
├── modules/            ← Python module files (.py)
├── pipeline/           ← pipeline definition code (.py)
├── outputs/            ← screenshots, artifacts
├── report/             ← report drafts
└── meeting_log.md      ← meeting register
```

### 0.2 — Colab: Install TFX and Airflow

Run this cell at the top of every Colab notebook:

```python
# Pin versions for reproducibility
!pip install tfx==1.15.0
!pip install apache-airflow==2.7.3
!pip install tensorflow-model-analysis
!pip install tensorflow-data-validation
!pip install witwidget  # What-If Tool

# Verify
import tfx
import airflow
print(f"TFX: {tfx.__version__}")
print(f"Airflow: {airflow.__version__}")
```

> **Note:** If the professor requires a local Airflow webserver (for the DAG screenshots), you can run Airflow in a Docker container — see Phase 3 below.

### 0.3 — Create requirements.txt

```
tfx==1.15.0
apache-airflow==2.7.3
tensorflow==2.15.0
tensorflow-model-analysis
tensorflow-data-validation
witwidget
```

This is a required deliverable — the instructor needs to reproduce your environment.

---

## Phase 1: Pipeline Components (Steps 1–4) — Data Ingestion & Validation

These steps are about getting your data into the TFX ecosystem and automatically checking its quality. The idea is that in production, you never trust raw data — you compute statistics, infer what the data *should* look like (schema), then validate every new batch against that schema.

### Step 1: ExampleGen — Ingest Your Data

**What this does:** Reads your CSV, splits it into train/eval sets (default 2:1 ratio), and converts it to TFRecord format (TensorFlow's efficient binary format for storing data).

```python
import os
from tfx.components import CsvExampleGen
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext

# Set paths
_pipeline_name = 'comp315_pipeline'
_pipeline_root = os.path.join('pipelines', _pipeline_name)
_metadata_path = os.path.join('metadata', _pipeline_name, 'metadata.db')
_data_root = 'data/'  # folder containing your CSV

# Create interactive context (runs components one at a time in a notebook)
context = InteractiveContext(
    pipeline_name=_pipeline_name,
    pipeline_root=_pipeline_root,
    metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(_metadata_path)
)

# ExampleGen
example_gen = CsvExampleGen(input_base=_data_root)
context.run(example_gen)

# Display the URIs of the data splits (required by the assignment)
for artifact in example_gen.outputs['examples'].get():
    print(f"Split: {artifact.split_names}")
    print(f"URI: {artifact.uri}")
```

### Step 2: StatisticsGen — Compute Data Statistics

**What this does:** Computes per-feature stats (min, max, mean, missing values, histograms) for both train and eval splits. This gives you a quantitative profile of your data.

```python
from tfx.components import StatisticsGen

statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
context.run(statistics_gen)

# Visualize statistics (this renders interactive histograms)
context.show(statistics_gen.outputs['statistics'])
```

> Take a screenshot of the rendered statistics — you'll need this for the report.

### Step 3: SchemaGen — Infer the Data Schema

**What this does:** Automatically infers what your data *should* look like — which features exist, their types (int, float, string), expected ranges, and vocabularies for categorical columns. Think of it as a contract for your data.

```python
from tfx.components import SchemaGen

schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'])
context.run(schema_gen)

# Visualize the schema
context.show(schema_gen.outputs['schema'])
```

**For the deliverable — write a short summary:**
- List the features and their inferred types
- Note any categorical features and their vocabulary sizes
- Note any value ranges that were inferred
- Flag anything that looks wrong (e.g., a numeric feature inferred as string)

### Step 4: ExampleValidator — Detect Data Anomalies

**What this does:** Compares your actual data statistics against the schema and flags anything suspicious — missing values, wrong types, values outside expected ranges. In production, this is your early warning system that something is wrong with incoming data.

```python
from tfx.components import ExampleValidator

example_validator = ExampleValidator(
    statistics=statistics_gen.outputs['statistics'],
    schema=schema_gen.outputs['schema']
)
context.run(example_validator)

# Show anomalies
context.show(example_validator.outputs['anomalies'])

# Print whether the data is healthy
from tensorflow_data_validation.utils.display_util import get_anomalies_dataframe
anomalies = example_validator.outputs['anomalies'].get()[0]
anomalies_df = get_anomalies_dataframe(anomalies)
if anomalies_df.empty:
    print("✅ Data pipeline is healthy — no anomalies detected.")
else:
    print("⚠️ Anomalies detected:")
    print(anomalies_df)
```

**For the Data Validation Report (1 page):**
- What anomalies were found (or that none were found)
- What each anomaly means in context of your dataset
- Whether you fixed them or adjusted the schema to accommodate them

---

## Phase 2: Pipeline Components (Steps 5–9) — Model Training & Evaluation

### Step 5: Transform — Feature Engineering

**What this does:** Defines how raw features get transformed before training — normalizing numbers, encoding categories, creating buckets. The key idea is that these transformations get *baked into the model's serving graph*, so the same preprocessing runs automatically at prediction time. No training/serving skew.

Create a file called `transform_module.py`:

```python
import tensorflow as tf
import tensorflow_transform as tft

# Adjust these to match YOUR dataset's columns
NUMERIC_FEATURES = ['age', 'income', 'hours_per_week']  # example
CATEGORICAL_FEATURES = ['workclass', 'education', 'occupation']  # example
LABEL_KEY = 'label'  # your target column

def preprocessing_fn(inputs):
    outputs = {}

    # Scale numeric features to z-scores
    for feature in NUMERIC_FEATURES:
        outputs[feature] = tft.scale_to_z_score(inputs[feature])

    # Vocabulary lookup for categorical features
    for feature in CATEGORICAL_FEATURES:
        outputs[feature] = tft.compute_and_apply_vocabulary(
            inputs[feature],
            top_k=100,
            num_oov_buckets=1
        )

    # Example: bucketize a numeric feature (optional but shows range)
    # outputs['age_bucket'] = tft.bucketize(inputs['age'], num_buckets=5)

    # Pass through the label
    outputs[LABEL_KEY] = inputs[LABEL_KEY]

    return outputs
```

Wire it into the pipeline:

```python
from tfx.components import Transform

_transform_module_file = 'modules/transform_module.py'

transform = Transform(
    examples=example_gen.outputs['examples'],
    schema=schema_gen.outputs['schema'],
    module_file=_transform_module_file
)
context.run(transform)
```

### Step 6: Trainer — Train Your Model

**What this does:** Builds and trains a Keras model on the transformed data, logs metrics to TensorBoard, and saves the model in SavedModel format (TensorFlow's standard format for deployment).

Create a file called `trainer_module.py`:

```python
import os
import tensorflow as tf
from tensorflow import keras
import tensorflow_transform as tft
from tfx_bsl.public import tfxio

# Adjust to match your dataset
NUMERIC_FEATURES = ['age', 'income', 'hours_per_week']
CATEGORICAL_FEATURES = ['workclass', 'education', 'occupation']
LABEL_KEY = 'label'
NUM_CATEGORICAL_VOCAB_SIZES = {  # Set after running Transform
    'workclass': 10,
    'education': 20,
    'occupation': 15,
}

def _input_fn(file_pattern, data_accessor, tf_transform_output, batch_size=64):
    return data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(batch_size=batch_size, label_key=LABEL_KEY),
        tf_transform_output.transformed_metadata.schema
    )

def _build_model(tf_transform_output):
    feature_spec = tf_transform_output.transformed_feature_spec().copy()
    feature_spec.pop(LABEL_KEY)

    inputs = {}
    encoded_features = []

    for feature in NUMERIC_FEATURES:
        inputs[feature] = keras.layers.Input(shape=(1,), name=feature, dtype=tf.float32)
        encoded_features.append(inputs[feature])

    for feature in CATEGORICAL_FEATURES:
        inputs[feature] = keras.layers.Input(shape=(1,), name=feature, dtype=tf.int64)
        vocab_size = NUM_CATEGORICAL_VOCAB_SIZES.get(feature, 100) + 1  # +1 for OOV
        embedding = keras.layers.Embedding(input_dim=vocab_size, output_dim=8)(inputs[feature])
        embedding = keras.layers.Flatten()(embedding)
        encoded_features.append(embedding)

    x = keras.layers.Concatenate()(encoded_features)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            keras.metrics.BinaryAccuracy(),
            keras.metrics.AUC(name='auc')
        ]
    )
    return model

def run_fn(fn_args):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(fn_args.train_files, fn_args.data_accessor,
                               tf_transform_output, batch_size=64)
    eval_dataset = _input_fn(fn_args.eval_files, fn_args.data_accessor,
                              tf_transform_output, batch_size=64)

    model = _build_model(tf_transform_output)

    # TensorBoard callback — REQUIRED by the assignment
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=fn_args.model_run_dir,
        update_freq='batch'
    )

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback]
    )

    # Save in SavedModel format with serving signature
    signatures = {
        'serving_default': _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
        )
    }
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)

def _get_serve_tf_examples_fn(model, tf_transform_output):
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)

    return serve_tf_examples_fn
```

Wire it in:

```python
from tfx.components import Trainer
from tfx.proto import trainer_pb2

_trainer_module_file = 'modules/trainer_module.py'

trainer = Trainer(
    module_file=_trainer_module_file,
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    schema=schema_gen.outputs['schema'],
    train_args=trainer_pb2.TrainArgs(num_steps=500),
    eval_args=trainer_pb2.EvalArgs(num_steps=100)
)
context.run(trainer)
```

### Step 7: Resolver — Find Latest Blessed Model

**What this does:** Looks up ML Metadata to find the most recently "blessed" (approved) model. On your first run, there won't be one — the Evaluator will automatically bless the first model. On subsequent runs, your new model must beat this baseline to be blessed. This is how production systems prevent deploying a worse model.

```python
from tfx.components import Resolver
from tfx.dsl.components.common.resolver import ResolverStrategy
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import LatestBlessedModelStrategy

model_resolver = Resolver(
    strategy_class=LatestBlessedModelStrategy,
    model=Channel(type=Model),
    model_blessing=Channel(type=ModelBlessing)
).with_id('latest_blessed_model_resolver')
context.run(model_resolver)
```

### Step 8: Evaluator — Evaluate with TFMA

**What this does:** Runs TensorFlow Model Analysis on your model, computing metrics (accuracy, AUC) not just overall but broken down by *slices* — subgroups of your data defined by a feature. This is how you catch a model that's 90% accurate overall but only 60% accurate for one demographic group.

```python
import tensorflow_model_analysis as tfma
from tfx.components import Evaluator

# Configure evaluation — ADJUST the slicing feature to your dataset
eval_config = tfma.EvalConfig(
    model_specs=[
        tfma.ModelSpec(label_key='label')  # your label column
    ],
    slicing_specs=[
        tfma.SlicingSpec(),  # overall dataset
        tfma.SlicingSpec(feature_keys=['race']),     # slice by race (example)
        tfma.SlicingSpec(feature_keys=['sex']),      # slice by sex (example)
    ],
    metrics_specs=[
        tfma.MetricsSpec(metrics=[
            tfma.MetricConfig(class_name='BinaryAccuracy'),
            tfma.MetricConfig(class_name='AUC'),
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
```

### Step 9: Pusher — Deploy the Blessed Model

**What this does:** If (and only if) the Evaluator blessed the model, the Pusher copies it to a serving directory. In production this would push to TF Serving or a cloud endpoint; for this project a local directory is fine.

```python
from tfx.components import Pusher
from tfx.proto import pusher_pb2

_serving_model_dir = os.path.join('serving_model', _pipeline_name)

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
```

### Step 10: Define the Pipeline Function

**What this does:** Bundles all components into a reusable function. This is what Airflow (or any orchestrator) calls to get the pipeline definition.

Create `pipeline.py`:

```python
import os
from typing import List, Optional
import tfx.v1 as tfx
from tfx.components import (
    CsvExampleGen, StatisticsGen, SchemaGen, ExampleValidator,
    Transform, Trainer, Resolver, Evaluator, Pusher
)
from tfx.proto import trainer_pb2, pusher_pb2
from tfx.dsl.components.common.resolver import ResolverStrategy
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import LatestBlessedModelStrategy
import tensorflow_model_analysis as tfma

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

    # Step 1
    example_gen = CsvExampleGen(input_base=data_root)

    # Step 2
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

    # Step 3
    schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'])

    # Step 4
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
    )

    # Step 5
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=transform_module_file
    )

    # Step 6
    trainer = Trainer(
        module_file=trainer_module_file,
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(num_steps=train_steps),
        eval_args=trainer_pb2.EvalArgs(num_steps=eval_steps),
    )

    # Step 7
    model_resolver = Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing)
    ).with_id('latest_blessed_model_resolver')

    # Step 8 — adjust eval_config to your dataset
    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key='label')],
        slicing_specs=[
            tfma.SlicingSpec(),
            tfma.SlicingSpec(feature_keys=['race']),
            tfma.SlicingSpec(feature_keys=['sex']),
        ],
        metrics_specs=[
            tfma.MetricsSpec(metrics=[
                tfma.MetricConfig(class_name='BinaryAccuracy'),
                tfma.MetricConfig(class_name='AUC'),
            ])
        ]
    )

    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config
    )

    # Step 9
    pusher = Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        )
    )

    components = [
        example_gen, statistics_gen, schema_gen, example_validator,
        transform, trainer, model_resolver, evaluator, pusher,
    ]

    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(metadata_path),
    )
```

---

## Phase 3: Airflow Integration

The assignment requires Airflow DAG screenshots. Here's how to get them.

### Option A: Airflow in Docker (Recommended)

```bash
# Create a docker-compose.yml
docker compose up -d

# Access Airflow UI at http://localhost:8080
# Default credentials: airflow / airflow
```

### Option B: Standalone Airflow

```bash
export AIRFLOW_HOME=~/airflow
airflow db init
airflow users create --username admin --password admin --firstname Admin \
    --lastname User --role Admin --email admin@example.com
airflow webserver --port 8080 &
airflow scheduler &
```

### Create the Airflow DAG file

Save this as `dags/comp315_dag.py` in your Airflow DAGs folder:

```python
import os
from datetime import datetime
from airflow import DAG
from tfx.orchestration.airflow.airflow_dag_runner import AirflowDagRunner
from tfx.orchestration.airflow.airflow_dag_runner import AirflowPipelineConfig
from pipeline import create_pipeline  # your Step 10 function

_pipeline_name = 'comp315_pipeline'
_pipeline_root = os.path.join(os.environ['HOME'], 'tfx', 'pipelines', _pipeline_name)
_data_root = os.path.join(os.environ['HOME'], 'tfx', 'data')
_transform_module = os.path.join(os.environ['HOME'], 'tfx', 'modules', 'transform_module.py')
_trainer_module = os.path.join(os.environ['HOME'], 'tfx', 'modules', 'trainer_module.py')
_serving_model_dir = os.path.join(os.environ['HOME'], 'tfx', 'serving_model', _pipeline_name)
_metadata_path = os.path.join(os.environ['HOME'], 'tfx', 'metadata', _pipeline_name, 'metadata.db')

airflow_config = {
    'schedule_interval': None,  # manual trigger only
    'start_date': datetime(2024, 1, 1),
}

dag = DAG(
    dag_id=_pipeline_name,
    default_args=airflow_config,
    schedule_interval=None,
    catchup=False,
)

pipeline = create_pipeline(
    pipeline_name=_pipeline_name,
    pipeline_root=_pipeline_root,
    data_root=_data_root,
    transform_module_file=_transform_module,
    trainer_module_file=_trainer_module,
    serving_model_dir=_serving_model_dir,
    metadata_path=_metadata_path,
)

AirflowDagRunner(AirflowPipelineConfig()).run(pipeline)
```

**Screenshots to capture:**
1. Full DAG graph view showing all tasks and their dependencies
2. DAG run view with all tasks green (successful)
3. At least one individual task log showing no errors
4. A second full DAG run (needed for TFMA comparison later)

---

## Phase 4: TFMA Analysis Notebook

Create a separate Jupyter notebook for this section.

### TFMA Step 1: Load and Render Slicing Metrics

```python
import tensorflow_model_analysis as tfma

# Path to the evaluator output — get this from evaluator.outputs
eval_result_path = '<path_to_evaluator_output>/eval'  # adjust this

eval_result = tfma.load_eval_result(eval_result_path)

# Render interactive visualization
tfma.view.render_slicing_metrics(eval_result)
```

**Write a summary covering:**
- Overall accuracy and AUC
- Accuracy and AUC for each slice (e.g., by race, by sex)
- Which slice performs best/worst and by how much
- Why certain slices might perform differently (data imbalance, feature correlation)

### TFMA Step 2: Fairness Indicator

```python
tfma.view.render_fairness_indicator(eval_result)
```

**Write an analysis covering:**
- Which slice has the lowest performance
- Why this might be the case (hypothesis)
- What data collection or preprocessing steps could fix the gap

### TFMA Step 3: Compare Two Runs

Run the pipeline a second time with different hyperparameters (e.g., change `train_steps` from 500 to 1000, or change the learning rate). Then:

```python
eval_result_1 = tfma.load_eval_result('<first_run_path>/eval')
eval_result_2 = tfma.load_eval_result('<second_run_path>/eval')

# Side-by-side comparison
tfma.view.render_slicing_metrics(
    eval_result_2,
    baseline_eval_result=eval_result_1
)
```

**Write a summary:** Did the changes improve or degrade performance? On which slices? Was the improvement uniform or did some slices benefit more than others?

---

## Phase 5: TensorBoard

```python
# In Colab
%load_ext tensorboard
%tensorboard --logdir <model_run_dir_path>
```

**Screenshots to capture:**
- Training loss curve
- Validation loss curve
- Training accuracy curve
- Validation accuracy curve
- AUC curve (if visible)

**Write a brief analysis:**
- Is the model converging? (Loss decreasing over time)
- Is there overfitting? (Training loss keeps dropping but validation loss plateaus or rises)
- Is there underfitting? (Both losses are high and flat)
- What would you change? (More epochs, different architecture, regularization)

---

## Phase 6: What-If Tool

```python
import witwidget
from witwidget.notebook.visualization import WitConfigBuilder, WitWidget

# Load eval data
eval_examples = []
# Read from your eval TFRecords
for record in tf.data.TFRecordDataset('<eval_tfrecord_path>'):
    eval_examples.append(record.numpy())

config = WitConfigBuilder(eval_examples[:500]).set_custom_predict_fn(
    lambda examples: model.predict(examples)
).set_label_vocab(['Negative', 'Positive'])  # adjust labels

WitWidget(config, height=800)
```

### What-If Step 1: Counterfactual Experiments

Pick 3 data points and manually change feature values to see how the prediction changes. For each:

1. Screenshot the original prediction
2. Screenshot after changing a feature
3. Write a paragraph: What did you change? How did the prediction shift? What does this tell you about what the model learned?

**Good experiments to try:**
- Change a demographic feature (age, race, sex) and see if the prediction flips
- Change a numeric feature to an extreme value
- Find the minimum change needed to flip a prediction

### What-If Step 2: Feature Distributions + Partial Dependence

- Screenshot the feature distribution view
- Screenshot partial dependence plots for 2-3 key features
- Screenshot the fairness threshold analysis from the Performance & Fairness tab

---

## Phase 7: Final Reflection Report (2–3 pages)

Answer these five questions with specific references to YOUR results:

1. **ExampleValidator insights:** "ExampleValidator caught [specific anomaly] that we would have missed. This matters because [explanation]."

2. **TFMA slicing insights:** "Overall accuracy was X%, but when we sliced by [feature], the [slice_value] group only had Y% accuracy. This reveals [insight]."

3. **What-If Tool insights:** "When we changed [feature] from [value_A] to [value_B], the prediction changed from [X] to [Y]. This suggests the model [learned/relies on ...]."

4. **Production improvements:** "If deploying to production, we would add [data versioning / model monitoring / A/B testing / CI/CD pipeline / etc.] because [reason]."

5. **Limitations:** "Our model is limited by [small dataset / class imbalance / missing features / potential bias in training data]. The training data [specific limitation]."

---

## Phase 8: Project Report Assembly

### Structure

```
Cover Page
Table of Contents
1. Rationale and Scope
   - Problem statement
   - Dataset description
   - Pipeline architecture overview
2. Pipeline Implementation
   - Data ingestion (ExampleGen)
   - Data validation (StatisticsGen, SchemaGen, ExampleValidator)
   - Feature engineering (Transform)
   - Model training (Trainer)
   - Model evaluation (Evaluator, Resolver)
   - Model deployment (Pusher)
3. TFMA Analysis
   - Slicing metrics results + interpretation
   - Fairness indicator results + interpretation
   - Two-run comparison + interpretation
4. TensorBoard Analysis
   - Training curves + interpretation
5. What-If Tool Analysis
   - Counterfactual experiments + interpretation
   - Fairness threshold analysis
6. Final Reflection
   - (the 5 questions above)
7. Conclusion
8. Assumptions
9. References
Appendix 1: Meeting Register
```

### Meeting Register Template

| Date | Time | Attendees | Topics Discussed | Assignments |
|------|------|-----------|-----------------|-------------|
| Week 9 | 2:00 PM | All | Project kickoff, dataset review, task split | Person A: Steps 1-4, Person B: Steps 5-9, Person C: Analysis |
| Week 10 | ... | ... | ... | ... |

---

## Phase 9: Presentation (Max 8 Slides)

```
Slide 1: Title (project name, team members, date)
Slide 2: Problem & Dataset (what you're predicting, dataset overview)
Slide 3: Pipeline Architecture (diagram of the 9 components)
Slide 4: Key Results — Data Validation (what ExampleValidator found)
Slide 5: Key Results — Model Performance (accuracy, AUC, TensorBoard curves)
Slide 6: TFMA & Fairness (slicing results, fairness indicator findings)
Slide 7: What-If Tool (most interesting counterfactual finding)
Slide 8: Lessons Learned & Production Considerations
```

---

## Task Split for 3 People — Week-by-Week

### Week 9 (Now)
- **All:** Set up shared Drive folder, install TFX in Colab, review dataset
- **Person A:** Run Steps 1–4 (data pipeline), take screenshots
- **Person B:** Start writing transform_module.py and trainer_module.py
- **Person C:** Set up report template, start meeting log

### Week 10
- **Person A:** Set up Airflow (Docker or local), create DAG file
- **Person B:** Run Steps 5–9, debug model training, get first blessed model
- **Person C:** Begin TFMA analysis notebook once evaluator outputs exist

### Week 11
- **Person A:** Get Airflow screenshots (all-green DAG), run pipeline second time
- **Person B:** Adjust hyperparameters for second run, help with What-If Tool
- **Person C:** Complete TFMA 3-step analysis, TensorBoard screenshots, What-If Tool experiments

### Week 12 (Submission)
- **All:** Write reflection report sections, review code comments
- **Person A:** Finalize requirements.txt, clean up code, verify reproducibility
- **Person B:** Write pipeline implementation sections of report
- **Person C:** Assemble final report PDF, create PowerPoint

### Week 13 (Presentation)
- **All:** Practice and deliver presentation, demonstrate working code

---

## Common Pitfalls & Fixes

| Problem | Fix |
|---------|-----|
| TFX won't install on ARM64 Mac | Use Google Colab |
| `ModuleNotFoundError` for module files | Use absolute paths or upload to Colab |
| Evaluator doesn't bless first model | Normal on first run if no baseline exists — check logs |
| TFMA visualizations don't render | Run in Jupyter, not plain Python; install `jupyter_widgets` |
| What-If Tool widget blank | Use Colab; local Jupyter needs `witwidget` extension enabled |
| Airflow DAG not appearing | Check file is in `$AIRFLOW_HOME/dags/`, restart scheduler |
| Schema infers wrong types | Manually edit the schema proto after first SchemaGen run |
| Out of memory during training | Reduce batch_size, reduce train_steps, use Colab GPU |
