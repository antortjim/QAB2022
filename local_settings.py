import os.path

OUTPUT_FOLDER="output"
VIDEO_FOLDER="/Users/FlySleepLab_Dropbox/Data/flyhostel_data/videos/"
DEBUG_FOLDER=os.path.join(OUTPUT_FOLDER, "debug")
DATASET_FOLDER={
        "sleap": os.path.join(OUTPUT_FOLDER, "sleap_data"),
        "mmpy":  os.path.join(OUTPUT_FOLDER, "mmpy_data")
}
DATA_FOLDER=os.path.join(OUTPUT_FOLDER, "mmpy_timeseries")
HISTOGRAM_FOLDER=os.path.join(OUTPUT_FOLDER, "intensity_histograms")
MASKS_FOLDER=os.path.join(OUTPUT_FOLDER, "masks")
CONTOURS_FOLDER=os.path.join(OUTPUT_FOLDER, "contours")
SLEAP_BOX=200
MMPY_BOX=100
DEBUG=False
DEBUG_COORDS=False
DEBUG_FIND_WING=False
DEBUG_HIDE_BACKGROUND=False
CENTRAL_BOX_SIZE=None
N_JOBS_EXPERIMENTS=1
N_JOBS_CHUNKS=20
MULTIPROCRESSING=False
MULTITHREADING=False
TOLERANCE=200 # ms
