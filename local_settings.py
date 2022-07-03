import os.path

OUTPUT_FOLDER="output"
VIDEO_FOLDER="/Users/FlySleepLab_Dropbox/Data/flyhostel_data/videos/"
DEBUG_FOLDER=os.path.join(OUTPUT_FOLDER, "debug")
DATASET_FOLDER={
        "sleap": os.path.join(OUTPUT_FOLDER, "sleap_data"),
        "mmpy":  os.path.join(OUTPUT_FOLDER, "mmpy_data")
}
# DATA_FOLDER=os.path.join(OUTPUT_FOLDER, "mmpy_timeseries")
HISTOGRAM_FOLDER=os.path.join(OUTPUT_FOLDER, "intensity_histograms")
MASKS_FOLDER=os.path.join(OUTPUT_FOLDER, "masks")
CONTOURS_FOLDER=os.path.join(OUTPUT_FOLDER, "contours")
DEBUG=False
DEBUG_FIND_CONTOUR=False
DEBUG_FIND_WING=False
DEBUG_HIDE_BACKGROUND=False
# taken from the size of the animal
CENTRAL_BOX_SIZE=None
# length of half a side of the square surrounding
# the fly in the sleap frames
SLEAP_BOX=200
# length of half a side of the square surrounding
# the fly in the mmpy frames
MMPY_BOX=100

# only if EXPERIMENT_PROCESS
N_JOBS_EXPERIMENTS=20
N_JOBS_CHUNKS=20
N_JOBS_RADON_TRANSFORM=10
MULTIPROCRESSING=False
MULTITHREADING=False
TOLERANCE=400 # ms
# whether to process each experiment as a process
EXPERIMENT_PROCESS=False
# whether to process each experiment as a subprocess ->
# this allows processing the chunks in parallel
EXPERIMENT_SUBPROCESS=True

HISTOGRAM_SAMPLE_SIZE=20
COMPUTE_THRESHOLDS=False
CROP=False
ROTATE=True
assert COMPUTE_THRESHOLDS and (not CROP and not ROTATE) or not COMPUTE_THRESHOLDS and (CROP or ROTATE)