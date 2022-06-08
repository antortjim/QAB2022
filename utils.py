import os.path
import numpy as np
import imgstore
import cv2
import idtrackerai.list_of_blobs

from get_files import (
    get_collections_file,
    get_trajectories_file,
    get_video_object
)

from contour import find_contour
from confapp import conf
try:
    import local_settings
    conf += local_settings
except ImportError:
    pass


def reproduce_example(animal, frame_number, experiment):
    # get the frame
    store = imgstore.new_for_filename(
        os.path.join(conf.VIDEO_FOLDER, experiment)
    )
    frame, _  = store.get_image(frame_number)

    # get the contour and centroid
    chunk = store._chunk_n
    blobs_in_video = idtrackerai.list_of_blobs.ListOfBlobs.load(
        get_collections_file(experiment, chunk)
    ).blobs_in_video
    
    body_size=round(get_video_object(experiment, chunk).median_body_length)

    frame_index = store._get_chunk_metadata(chunk)["frame_number"].index(frame_number)
    trajectories=np.load(get_trajectories_file(experiment))
    blobs_in_frame = blobs_in_video[frame_index]
    tr = trajectories[frame_number, animal, :]            

    centroid = tuple([round(e) for e in tr])
    contour, other_contours = find_contour(blobs_in_frame, centroid)
    filepath="test.png"
    
    return frame, contour, other_contours, centroid, body_size, filepath

def hours(x):
    return x*3600


def package_frame_for_labeling(frame, center, box_size):
    
    # bbox = [tl_x, tl_y, br_x, br_y]
    bbox = [
            center[0] - box_size,
            center[1] - box_size,
            center[0] + box_size,
            center[1] + box_size,
        ]

    bbox = [
        max(0, bbox[0]),
        max(0, bbox[1]),
        min(frame.shape[1], bbox[2]),
        min(frame.shape[0], bbox[3])
    ]
    
    if conf.DEBUG:
        print(f"Final box: {bbox}")
        
    frame=frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    target_height = target_width = box_size*2
    actual_height = (bbox[3]-bbox[1])
    actual_width = (bbox[2]-bbox[0])
    

    # pad with black to ensure all img have equal size
    pad_bottom = round(target_height - actual_height)
    pad_right = round(target_width - actual_width)
    
    if conf.DEBUG:
        print(f"Padding: 0x{pad_bottom}x0x{pad_right}")
    
    frame=cv2.copyMakeBorder(frame, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, 255)
    return frame, bbox
