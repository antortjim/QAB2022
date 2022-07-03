import logging
import logging.config
import os.path
import yaml
import numpy as np
import shlex
# import imgstore.stores.multi as imgstore

LOGGING_FILE=os.path.join(os.environ["HOME"], ".config", "qab2022.yaml")

from get_files import get_experiments
from library import generate_dataset
import utils

from confapp import conf
try:
    import local_settings
    conf += local_settings
except ImportError:
    pass

logger = logging.getLogger(__name__)

with open(LOGGING_FILE, "r") as filehandle:
    config = yaml.load(filehandle, yaml.SafeLoader)

logging.config.dictConfig(config)


def main():
    
    metadata=get_experiments()
    experiments = metadata["experiment"]

    if conf.EXPERIMENT_PROCESS:
        import joblib

        joblib.Parallel(n_jobs=conf.N_JOBS_EXPERIMENTS)(
            joblib.delayed(generate_dataset)(
                experiment, compute_thresholds=conf.COMPUTE_THRESHOLDS, tolerance=conf.TOLERANCE, crop=conf.CROP, rotate=conf.ROTATE
            ) for i, experiment in enumerate(experiments)
        )
        
    elif conf.EXPERIMENT_SUBPROCESS:
        import subprocess

        subprocesses=[]
        for i, experiment in enumerate(experiments):
            cmd = f"python main_sp.py --experiment {experiment}"
            cmd_l = shlex.split(cmd)
            subprocesses.append(
                subprocess.Popen(
                    cmd_l
                )
            )
    else:
        for i, experiment in enumerate(experiments):
            generate_dataset(experiment, compute_thresholds=conf.COMPUTE_THRESHOLDS, tolerance=conf.TOLERANCE, crop=conf.CROP, rotate=conf.ROTATE)
    
if __name__ == "__main__":
    main()
