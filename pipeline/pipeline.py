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
