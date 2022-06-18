import logging
import logging.config
import os.path
import yaml
import joblib
import numpy as np
# import imgstore.stores.multi as imgstore

LOGGING_FILE=os.path.join(os.environ["HOME"], ".config", "qab2022.yaml")

from get_files import get_experiments
from library import generate_dataset
import utils
from timepoints import TimePoints

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
    
    experiments=get_experiments()
    # sampling_points_msec = np.linspace(start = utils.hours(1), stop=utils.hours(36), num=100)[:2] *1000
    # sampling_points_msec = (utils.hours(1.1) * 1000, utils.hours(2.1) * 1000)

    sampling_points_msec = (utils.hours(1.1) * 1000, utils.hours(1.15) * 1000)
    sampling_points_msec = (4133681, 4134681)

    joblib.Parallel(n_jobs=conf.N_JOBS_EXPERIMENTS)(
        joblib.delayed(generate_dataset)(
            experiment, TimePoints(sampling_points_msec), compute_thresholds=False, tolerance=conf.TOLERANCE, crop=True, rotate=True
        ) for experiment in experiments[:2]
    )
    
if __name__ == "__main__":
    main()