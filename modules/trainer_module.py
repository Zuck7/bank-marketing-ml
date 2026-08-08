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
    - All are concatenated -> Dense(128) -> Dropout -> Dense(64) -> Dropout -> sigmoid output

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

    # Categorical features — integer IDs -> embedding -> flatten
    for feature in CATEGORICAL_FEATURES:
        inp = keras.layers.Input(shape=(1,), name=feature, dtype=tf.int64)
        inputs[feature] = inp
        vocab_size = VOCAB_SIZES[feature]
        # Embedding dimension: min(8, vocab_size // 2) is a common heuristic
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
