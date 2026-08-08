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
# Make sure pipeline.py is importable (e.g. on PYTHONPATH or in same directory)
import sys
sys.path.insert(0, os.path.join(os.environ['HOME'], 'tfx_project'))
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
