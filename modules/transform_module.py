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
