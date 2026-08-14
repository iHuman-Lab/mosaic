from pathlib import Path

import ray
import yaml
from dotenv import load_dotenv
from ixp.experiment import Experiment
from ixp.individual_difference.mot import MOT
from ixp.individual_difference.vs import VS
from ixp.surveys.nasa_tlx import NasaTLX
from ixp.surveys.sart import SART

from .game import SARGame
from .sensors.eye_tracker.tobii import TobiiEyeTracker
from .tutorial import SARTutorial
from .utils import skip_run

load_dotenv()

with Path("configs/experiment.yaml").open() as f:
    config = yaml.safe_load(f)

with Path("src/experiment/instructions.yaml").open() as f:
    instructions = yaml.safe_load(f)


with skip_run("skip", "tobii_calibration") as check, check():
    tobii = TobiiEyeTracker()
    tobii.initialize()
    tobii.calibrate()

with skip_run("skip", "tobii_test") as check, check():
    ray.init(ignore_reinit_error=True, _system_config={"metrics_report_interval_ms": 0})
    experiment = Experiment(config)

    experiment.register_sensor(
        name="TobiiEyeTracker", sensor_cls=TobiiEyeTracker, sensor_config={"config": {}}
    )
    experiment.calibrate_sensor(
        "TobiiEyeTracker", screen=config["display"], fullscreen=config["fullscreen"]
    )
    experiment.run()
    experiment.close()


with skip_run("skip", "sar_experiment_test") as check, check():
    ray.init(ignore_reinit_error=True, _system_config={"metrics_report_interval_ms": 0})
    experiment = Experiment(config)

    experiment.add_task(
        name="main_game",
        task_cls=SARGame,
        task_config={"config": config["game"]},
        order=3,
        instructions=instructions["main_game"],
    )

    # Run the experiment
    experiment.run()
    experiment.close()


with skip_run("run", "instruction_test") as check, check():
    ray.init(ignore_reinit_error=True, _system_config={"metrics_report_interval_ms": 0})
    experiment = Experiment(config)

    # Add visual search and mot
    experiment.add_task(
        name="visual_search",
        task_cls=VS,
        task_config={"config": config["vs"]},
        order=1,
        instructions=instructions["visual_search"],
    )

    # Run the experiment
    experiment.run()
    experiment.close()


with skip_run("skip", "sar_experiment") as check, check():
    ray.init(ignore_reinit_error=True, _system_config={"metrics_report_interval_ms": 0})
    experiment = Experiment(config)

    # Add sensors
    experiment.register_sensor(
        name="TobiiEyeTracker", sensor_cls=TobiiEyeTracker, sensor_config={"config": {}}
    )
    experiment.calibrate_sensor(
        "TobiiEyeTracker", screen=config["display"], fullscreen=config["fullscreen"]
    )

    # Add visual search and mot
    experiment.add_task(
        name="visual_search",
        task_cls=VS,
        task_config={"config": config["vs"]},
        order=1,
        instructions=instructions["visual_search"],
    )
    experiment.add_task(
        name="multi_object_tracking",
        task_cls=MOT,
        task_config={"config": config["mot"]},
        order=2,
        instructions=instructions["multi_object_tracking"],
    )
    experiment.add_task(
        name="practice",
        task_cls=SARTutorial,
        task_config={"config": config["game"]},
        order=2,
        instructions=instructions["practice"],
    )
    # Main game
    experiment.add_task(
        name="main_game",
        task_cls=SARGame,
        task_config={"config": config["game"]},
        order=3,
        instructions=instructions["main_game"],
    )
    experiment.add_task(
        name="sart",
        task_cls=SART,
        task_config={"config": config["surveys"]},
        order=5,
        instructions=instructions["sart"],
    )
    experiment.add_task(
        name="nasa_tlx",
        task_cls=NasaTLX,
        task_config={"config": config["surveys"]},
        order=6,
        instructions=instructions["nasa_tlx"],
    )

    # Run the experiment
    experiment.run()
    experiment.close()
