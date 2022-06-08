import logging
import logging.config
import joblib
import numpy as np
# import imgstore.stores.multi as imgstore


from get_files import get_experiments
from library import generate_dataset
import utils

from confapp import conf
try:
    import local_settings
    conf += local_settings
except ImportError:
    pass

logging.config.dictConfig({
    "version": 1.0,
    "root":  {"level": "DEBUG"}
})

logger = logging.getLogger(__name__)

def main():
    
    experiments=get_experiments()
    sampling_points = np.linspace(start = utils.hours(1), stop=utils.hours(36), num=100)
    sampling_points_msec = sampling_points*1000


    joblib.Parallel(n_jobs=conf.N_JOBS)(
        joblib.delayed(generate_dataset)(
            experiment, sampling_points_msec
        ) for experiment in experiments[:2]
    )
    
if __name__ == "__main__":
    main()