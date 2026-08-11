# Bank Marketing ML Pipeline

A production ML pipeline that predicts whether a bank client will subscribe to a term deposit, built with TensorFlow Extended (TFX) and orchestrated by Apache Airflow.

---

## The Problem

A Portuguese bank ran direct marketing campaigns via phone calls to sell term deposits. Often, multiple calls to the same client were needed. The dataset captures 45,211 client interactions with 16 features covering demographics, financial status, and campaign history.

The classification goal: **predict if the client will subscribe** (`yes` / `no`).

The challenge: the dataset is heavily imbalanced — 88.3% of clients did not subscribe. A naive model predicting "no" for everyone gets 88% accuracy. The pipeline needs to do meaningfully better, and critically, it needs to perform fairly across different client segments.

---

## The Data

| Feature | Type | Description |
|---------|------|-------------|
| age | Numeric | Client age |
| job | Categorical (12) | admin, blue-collar, technician, management, retired, student, etc. |
| marital | Categorical (3) | married, single, divorced |
| education | Categorical (4) | primary, secondary, tertiary, unknown |
| default | Binary | Has credit in default? |
| balance | Numeric | Average yearly balance (euros) |
| housing | Binary | Has housing loan? |
| loan | Binary | Has personal loan? |
| contact | Categorical (3) | Contact type: cellular, telephone, unknown |
| day | Numeric | Last contact day of month |
| month | Categorical (12) | Last contact month |
| duration | Numeric | Last contact duration (seconds) |
| campaign | Numeric | Contacts during this campaign |
| pdays | Numeric | Days since last contact from previous campaign (-1 = never contacted) |
| previous | Numeric | Contacts before this campaign |
| poutcome | Categorical (4) | Previous campaign outcome: success, failure, other, unknown |
| **y** | **Target** | **Subscribed to term deposit?** |

---

## Feature Engineering

Raw features are transformed into model-ready format through TensorFlow Transform. Every transformation gets baked into the SavedModel's serving graph — the same preprocessing runs automatically at prediction time, eliminating training/serving skew.

| Transformation | Applied To | What It Does |
|---------------|-----------|--------------|
| Z-score scaling | All 7 numeric features | Centers each feature around 0 with standard deviation 1. Neural networks train faster when inputs are on similar scales — without this, `balance` (range: -8,019 to 102,127) would dominate `campaign` (range: 1 to 63). |
| Vocabulary encoding | All 9 categorical features | Maps string categories to integer IDs. `"admin."` becomes `0`, `"blue-collar"` becomes `1`, etc. An out-of-vocabulary bucket catches unseen categories at serving time. |
| Bucketization | age → 5 bins | Splits age into discrete groups (roughly: 18–30, 30–40, 40–50, 50–60, 60+). The relationship between age and term deposit subscription is non-linear — students and retirees behave differently from working-age clients, and buckets let the model learn this without assuming linearity. |

---

## Model Architecture

```
Numeric features (7)                    Categorical features (9)           Engineered feature
   age ──────┐                          job ──── Embed(13→6) ── Flat ─┐     age_bucket ── Embed(6→4) ── Flat ─┐
   balance ──┤                          marital ─ Embed(4→2) ── Flat ─┤                                       │
   day ──────┤                          education Embed(5→2) ── Flat ─┤                                       │
   duration ─┤                          default ─ Embed(3→2) ── Flat ─┤                                       │
   campaign ─┤                          housing ─ Embed(3→2) ── Flat ─┤                                       │
   pdays ────┤                          loan ──── Embed(3→2) ── Flat ─┤                                       │
   previous ─┘                          contact ─ Embed(4→2) ── Flat ─┤                                       │
      │                                 month ─── Embed(13→6) ─ Flat ─┤                                       │
      │                                 poutcome  Embed(5→2) ── Flat ─┘                                       │
      │                                            │                                                           │
      └────────────────────────────────────────────┴───────────────────────────────────────────────────────────┘
                                                   │
                                            Concatenate (37 dimensions)
                                                   │
                                            Dense(128, ReLU)
                                            BatchNormalization
                                            Dropout(0.3)
                                                   │
                                            Dense(64, ReLU)
                                            BatchNormalization
                                            Dropout(0.2)
                                                   │
                                            Dense(32, ReLU)
                                                   │
                                            Dense(1, Sigmoid) → prediction
```

### Design Decisions

**Embeddings instead of one-hot encoding.** One-hot encoding `job` creates a sparse 12-dimensional vector where each dimension is independent. An embedding maps the 12 job types into a dense 6-dimensional space where similar jobs (e.g., `admin` and `management`) can end up near each other. This gives the model capacity to learn relationships between categories while using fewer parameters.

**Batch normalization after each dense layer.** The numeric inputs arrive z-scored, but embedding outputs are randomly initialized and learned during training. These start on different scales. Batch normalization re-centers and re-scales activations between layers, stabilizing training and allowing higher learning rates.

**Progressive dropout (0.3 → 0.2).** Higher dropout in earlier layers provides stronger regularization where the feature space is widest (37 → 128 dimensions). Lower dropout in later layers preserves more of the abstract representations the model has learned.

**Early stopping with patience 5.** Monitors validation loss and stops training if it doesn't improve for 5 consecutive evaluation rounds, then restores the best weights. This prevents overfitting on a dataset where 88% of examples are the same class.

### Model Size

| Property | Value |
|----------|-------|
| Total Parameters | 16,235 (63.42 KB) |
| Trainable Parameters | 15,851 |
| Non-trainable Parameters | 384 (BatchNorm stats) |
| Optimizer | Adam (learning rate: 0.001) |
| Loss Function | Binary Crossentropy |

The model is deliberately small. For tabular data with 16 features and 45K rows, larger models overfit. The architecture choices — embeddings, batch normalization, dropout — matter far more than adding layers.

---

## Training Results

### Run 1: 500 Training Steps

```
500/500 ━━━━━━━━━━━━━━━━━━━━ 9s 13ms/step

Training:     loss: 0.2727  accuracy: 88.4%  AUC: 0.848
Validation:   loss: 0.2118  accuracy: 91.4%  AUC: 0.900
```

### Run 2: 1000 Training Steps

```
1000/1000 ━━━━━━━━━━━━━━━━━━━━ 12s 9ms/step

Training:     loss: 0.2600  accuracy: 88.7%  AUC: 0.867
Validation:   loss: 0.2359  accuracy: 90.1%  AUC: 0.907
```

### What the Numbers Mean

| Metric | Run 1 | Run 2 | Interpretation |
|--------|-------|-------|----------------|
| Val Accuracy | **91.4%** | 90.1% | Run 1 classifies more examples correctly overall |
| Val AUC | 0.900 | **0.907** | Run 2 ranks positive vs negative examples better |
| Val Loss | **0.212** | 0.236 | Run 1 is more confident in its correct predictions |
| Train Loss | 0.273 | **0.260** | Run 2 fits the training data slightly better |

**Why does Run 2 have lower accuracy but higher AUC?** With more training steps, the model improves its ability to rank predictions (AUC goes up) but starts to overfit slightly on the majority class, reducing accuracy on borderline cases. For an imbalanced dataset, **AUC is the more reliable metric** because accuracy can be inflated by always predicting the majority class.

**No overfitting.** In both runs, validation loss is lower than training loss. This means the model generalizes well to unseen data — dropout and batch normalization are doing their job.

---

## Fairness Analysis (TFMA Slicing)

The model was evaluated not just on overall performance, but across demographic and professional slices. A model can look great at 91% overall while hiding poor performance for specific groups.

### By Education Level

| Slice | Accuracy | AUC | Example Count |
|-------|----------|-----|---------------|
| Tertiary | Highest | Highest | ~14,000 |
| Secondary | Middle | Middle | ~23,000 |
| Primary | Lowest | Lowest | ~6,800 |
| Unknown | Variable | Variable | ~1,800 |

Tertiary-educated clients are the easiest to predict — they also have the highest subscription rate. Primary-education clients show the widest performance gap, likely because their subscription patterns differ from the majority but there are fewer examples to learn from.

### By Job Type

| Observation | Detail |
|-------------|--------|
| Hardest slice | **Students** at 76.1% accuracy — 15 points below the best |
| Best-performing | Service workers and retirees |
| Largest groups | Blue-collar (~9,700) and management (~9,400) |
| Smallest groups | Students (~938) and unemployed (~1,300) |

Students have a high subscription rate relative to their group size, making them an unusual pattern the model struggles with. The small sample size compounds this — the model sees fewer examples to learn the student pattern.

### By Marital Status

Performance is relatively uniform across married, single, and divorced groups. This is the fairest dimension — the model does not appear to discriminate based on marital status.

### Implications

The fairness gaps are driven by two factors: class imbalance within slices (some groups have very different subscription rates) and sample size (smaller groups have noisier signal). Production mitigations would include class weighting, oversampling underrepresented groups (SMOTE), or training separate models for high-variance slices.

---

## Pipeline Components

The pipeline automates the entire ML workflow through 9 sequential components, each producing versioned artifacts tracked by ML Metadata:

### Phase 1: Data Validation

**ExampleGen** ingests the CSV, splits it into train (2/3) and eval (1/3) sets, and converts to TFRecord format. TFRecords are TensorFlow's optimized binary storage — faster to read during training than CSV.

**StatisticsGen** computes per-feature statistics: min, max, mean, standard deviation, missing value counts, and distributions. This produces a quantitative profile of the data.

**SchemaGen** infers what the data should look like: which features exist, their types (int, float, string), expected ranges, and vocabularies for categorical columns. This schema acts as a contract for the data.

**ExampleValidator** compares actual statistics against the schema and flags anomalies — missing values, wrong types, values outside expected ranges. Result: **no anomalies detected** in this dataset.

### Phase 2: Model Training & Deployment

**Transform** applies feature engineering (z-scoring, vocabulary encoding, bucketization) and materializes the transformations into a graph that ships with the model.

**Trainer** builds and trains the Keras model, logging metrics to TensorBoard. Produces a SavedModel with a serving signature that includes the Transform preprocessing.

**Resolver** looks up ML Metadata to find the most recently blessed model as a baseline. On the first run, no baseline exists — the Evaluator auto-blesses.

**Evaluator** runs TFMA with slicing across education, marital status, and job type. Computes BinaryAccuracy and AUC per slice. Determines whether the new model should be blessed.

**Pusher** copies the blessed model to the serving directory. If the model was not blessed (failed to beat the baseline), nothing gets deployed.

### Phase 3: Orchestration

All 9 components are wrapped in an **Apache Airflow DAG** where each component becomes a task with dependency tracking. The DAG was tested via Airflow CLI with all tasks completing successfully.

---

## Artifact Lineage

Every pipeline run records a complete lineage chain in ML Metadata:

```
Raw CSV → ExampleGen artifacts → Statistics → Schema → Anomalies
    → Transform graph → Transformed examples
        → Trained model → Model evaluation → Blessing decision
            → Pushed model (if blessed)
```

After multiple runs, the metadata store contains the full history: which data was used, what schema was inferred, which model was trained with which hyperparameters, whether it was blessed, and where it was deployed. This makes any past run fully reproducible and auditable.

---

## What We Would Change for Production

**Address class imbalance.** The 88/12 split means the model sees 7x more negative examples. Class weighting (`class_weight={0: 1, 1: 7.5}`) or SMOTE oversampling would help the model learn the minority pattern better, particularly for underperforming slices like students.

**Add model monitoring.** Deploy with a monitoring system that tracks prediction distributions over time. If the distribution of predicted probabilities shifts (data drift) or the feature distributions change (concept drift), trigger automatic retraining.

**A/B testing before rollout.** Instead of replacing the old model entirely, route a percentage of traffic to the new model and compare real-world conversion rates before full deployment.

**CI/CD pipeline.** Automate the retrain → evaluate → deploy cycle so new data triggers a pipeline run without manual intervention. The TFX pipeline already supports this — it just needs a scheduler.

**Remove duration feature.** `duration` (call length in seconds) is only known after the call ends, making it unavailable at prediction time for new campaigns. Including it inflates model performance. A production model should be retrained without it.

---

## Tech Stack

| Component | Version | Role |
|-----------|---------|------|
| TFX | 1.15.0 | Pipeline framework |
| TensorFlow | 2.15.0 | Model training |
| TF Transform | 1.15.0 | Feature engineering |
| TFMA | 0.46.0 | Evaluation & fairness |
| TFDV | 1.15.1 | Data validation |
| Apache Beam | 2.56.0 | Data processing |
| Apache Airflow | 2.7.3 | Orchestration |
| ML Metadata | 1.15.0 | Artifact tracking |
| Python | 3.10 | Runtime |

---

## Reference

Moro, S., Laureano, R., & Cortez, P. (2011). Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology. *Proceedings of the European Simulation and Modelling Conference — ESM'2011*, pp. 117-121. EUROSIS.

---

*COMP315 AI Software Test & ML/Ops — Centennial College, Fall 2026*
