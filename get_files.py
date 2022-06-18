import json
import logging
import os.path
from confapp import conf
import numpy as np
try:
    import local_settings
    conf += local_settings
except ImportError:
    pass


logger = logging.getLogger(__name__)

def get_collections_file(experiment, chunk):
    path=os.path.join(conf.VIDEO_FOLDER, experiment, "idtrackerai", f"session_{str(chunk).zfill(6)}", "preprocessing", "blobs_collection.npy")
    assert os.path.exists(path), f"{path} does not exist"
    return path

def get_video_object(experiment, chunk):
    try:
        cwd=os.getcwd()
        os.chdir(conf.VIDEO_FOLDER)
        path=os.path.join(experiment, "idtrackerai", f"session_{str(chunk).zfill(6)}", "video_object.npy")
        logger.debug(f"Loading {path}")
        assert os.path.exists(path), f"{path} not found"
        video_object = np.load(path, allow_pickle=True).item()
        os.chdir(cwd)

        return video_object
    
    except Exception as error:
        os.chdir(cwd)
        raise error

def get_trajectories_file(experiment):

    return os.path.join(
        conf.VIDEO_FOLDER,
        experiment,
        "fh_quant",
        f"{experiment.split('/')[1]}_trajectories.npy"
    )
def get_timestamps_file(experiment):
    
    return os.path.join(
        conf.VIDEO_FOLDER,
        experiment,
        "fh_quant",
        f"{experiment.split('/')[1]}_timestamps.npy"
    )


def get_store_path(experiment):
    return os.path.join(conf.VIDEO_FOLDER, experiment)
    
def load_conf(experiment):
    with open(
        os.path.join(conf.VIDEO_FOLDER, experiment, os.path.basename(experiment)+".conf"), "r") as filehandle:
        conf=json.load(filehandle)
    
    return {
        "area": conf["_area"]["value"],
         "intensity": conf["_intensity"]["value"]
    }



def get_experiments():

    with open("experiments.txt", "r") as filehandle:
        experiments = [e.strip("\n") for e in filehandle.readlines()]
    experiments = [e for e in experiments if e[0] != "#"]
    print(experiments)    
    return experiments
