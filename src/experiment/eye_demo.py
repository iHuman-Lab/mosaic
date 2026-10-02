"""Eye-tracking demonstration: one keyless SAR mission with the Tobii streaming beside it.

    PYTHONPATH=src python -m experiment.eye_demo

Registers and calibrates the eye tracker, then runs a single short mission.
Gaze (``TobiiEyeTracker``) and game state (``SARGame``) are published as LSL
streams on one clock; record them with LabRecorder. The study itself
(``experiment.py`` and ``configs/experiment.yaml``) is left as it is.
"""

from pathlib import Path

import ray
import yaml
from ixp.experiment import Experiment

from .game import SARGameDemo
from .sensors.eye_tracker.tobii import TobiiEyeTracker

DEMO_MINUTES = 2

with Path("configs/experiment.yaml").open() as f:
    config = yaml.safe_load(f)

ray.init(ignore_reinit_error=True, _system_config={"metrics_report_interval_ms": 0})
experiment = Experiment(config)

experiment.register_sensor(
    name="TobiiEyeTracker", sensor_cls=TobiiEyeTracker, sensor_config={}
)
experiment.calibrate_sensor(
    "TobiiEyeTracker", screen=config["display"], fullscreen=config["fullscreen"]
)

experiment.add_task(
    name="main_game",
    task_cls=SARGameDemo,
    task_config={"config": {**config["game"], "max_time": DEMO_MINUTES}},
)

experiment.run()
experiment.close()
