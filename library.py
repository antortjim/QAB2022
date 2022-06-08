import os.path
import warnings
import logging

import numpy as np
import cv2
import idtrackerai.list_of_blobs
import imgstore
import tqdm

from contour import center_blob_in_mask, find_contour
from plotting import plot_rotation
from get_files import (
    get_video_object,
    get_trajectories_file,
    get_timestamps_file,
    get_collections_file,
    get_store_path
)
from utils import package_frame_for_labeling

from confapp import conf


try:
    import local_settings
    conf += local_settings
except ImportError:
    pass


logger = logging.getLogger(__name__)

def crop_animal_in_time_and_space(frame, contour, other_contours, centroid, body_size, filepath):
    """
    Crop a box in the frame around the focal fly and rotate it so the animal points east 
    
    Arguments:
    
        * frame (np.ndarray): raw frame from the dataset
        * contour (np.ndarray): outline of the contour of one of the animals (focal) in the frame, in raw coordinates
        * centroid (tuple): x and y coordinates of the center of the focal animal
        * filepath (str): Path to a .png file
    
    Returns:
        * rotated (np.ndarray): crop of the raw frame with the focal animal pointing east 
    """

    if contour is None:
        warnings.warn(f"No contour detected in {filepath}", stacklevel=2)
        return
   
    # mask the other contours
    easy_frame=cv2.drawContours(frame.copy(), other_contours, -1, 255, -1)

    if conf.CENTRAL_BOX_SIZE is None:
        CENTRAL_BOX_SIZE=body_size*3
    else:
        CENTRAL_BOX_SIZE=conf.CENTRAL_BOX_SIZE
    
    easy_crop, bbox = package_frame_for_labeling(
        easy_frame, centroid, CENTRAL_BOX_SIZE
    )
    raw_crop, bbox = package_frame_for_labeling(
        frame, centroid, CENTRAL_BOX_SIZE
    )

    top_left_corner = bbox[:2]
    centered_contour=contour-top_left_corner

    angle, (T, mask, cloud, cloud_centered, cloud_center)=find_angle(easy_crop, centered_contour, body_size=body_size)
    rotated, rotate_matrix = rotate_frame(raw_crop, angle)
    
    easy_rotated, _ = rotate_frame(easy_crop, angle)

    mmpy_frame, _ = package_frame_for_labeling(easy_rotated, center=([e//2 for e in rotated.shape[:2][::-1]]), box_size=conf.MMPY_BOX)
    sleap_frame, _ = package_frame_for_labeling(rotated, center=([e//2 for e in rotated.shape[:2][::-1]]), box_size=conf.SLEAP_BOX)
    
    frames = {"sleap": sleap_frame, "mmpy": mmpy_frame}

    if conf.DEBUG:
        plot_rotation(raw_crop, mask, T, cloud_centered, filepath)

    # save
    for folder in conf.DATASET_FOLDER:
        os.makedirs(conf.DATASET_FOLDER[folder], exist_ok=True)
        path=os.path.join(conf.DATASET_FOLDER[folder], filepath.replace(".png", "_05_final.png"))
        
        if conf.DEBUG:
            print(f'Saving -> {path}')

        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, frames[folder])

    return raw_crop, rotated, (T, mask, cloud, cloud_centered, cloud_center, rotate_matrix)


def find_angle(crop, contour, body_size, point_to=90):
    """
    Take a crop with a centered animal and some surrounding space
    and rotate so the animal points east
    
    Arguments:  
    * crop (np.ndarray): Grayscale image of the animal
    * contour (np.ndarray): Contour of the animal in the image
    * point_to (int): Direction to which the animal should point to.
      0 means east, 90 means north, -90 means south and 180 means west

    Returns:
    * rotated (np.ndarray): Grayscale image rotated
    
    Details:
    
    * a contour in opencv has always shape nx1x2
      where n is the number of points
      and we have two dimensions for x and y in the 3rd axis
    """
    
    mask = np.zeros_like(crop)
    
    # contour should have shape ?x1x2
    assert contour.shape[1] == 1
    assert contour.shape[2] == 2
    
    mask=cv2.drawContours(mask, [contour], -1, 255, -1)
    cloud=np.stack(
        np.where(mask == 255)[::-1],
        axis=1
    )
    # np.where returns the coordinates along the first axis,
    # then the second, third, and so on i.e. first rows and then columns
    # this means the first column of cloud contains the coordinates along the
    # first axis of mask (rows), i.e. the height (y)
    # this means we need to flip the order of the columns
    # so the first column represents the x coordinates
    # and the second the y coordinates
    
    # now cloud's first column represents the x (horizontal axis)
    # and the second 
    
    # also, the origin now is set to the top left corner
    # (because the first row is the row on top)
    # and we want it in the bottom left, so we need to flip the y axis
    cloud[:, 1] = mask.shape[0] - cloud[:, 1]

    cloud_center = center_blob_in_mask(mask)
    
    if conf.DEBUG:
        print(f"Cloud center {cloud_center}")

    # center the contour around its mean
    cloud_centered=cloud-cloud_center

    # compute the covariance matrix
    cov_matrix = np.cov(cloud_centered.T)
    
    # Eigendecomposition
    ################################
    # compute the eigenvectors of the covariance matrix
    vals, T = np.linalg.eig(cov_matrix)
    
    T=T[:, np.argsort(vals)[::-1]]
    vals=vals[np.argsort(vals)[::-1]]
    
    
    # get the first eigen vector
    v1 = T[:, 0]
    v2 = T[:, 1]

    if conf.DEBUG:
        print(f"Eigenvalues: {vals}")
        print(f"First eigenvector {v1}")
        print(f"Second eigenvector {v2}")

    # Singular Value Decomposition
    ################################
    # u, s, T = np.linalg.svd(cloud_centered)
    # v1 = T[:,0]
   
    # compute the angle of the first eigenvector in radians
    
    angle = np.arctan2(v1[1], v1[0])
    # transform to degrees
    angle_deg=angle*360/(np.pi*2)-point_to
    
    if conf.DEBUG:
        print(f"Angle {angle_deg} degrees")


    rotated, rotate_matrix = rotate_frame(crop, angle_deg)
    flip=find_polarity(crop, mask, rotated, rotate_matrix, body_size, filepath=None)
    if flip:
        angle_deg-=180

    return angle_deg, (T, mask, cloud, cloud_centered, cloud_center)

def rotate_frame(img, angle):
    """
    Rotate the img the given angle
    
    Arguments:
    
        * img (np.ndarray)
        * angle (float): degrees
    Returns:
    
        * rotated (np.ndarray): img after applying the rotation
        * rotate_matrix (np.ndarray): rotation matrix that was used to perform the rotation
    """

    # compute the rotation matrix needed to cancel the angle
    rotate_matrix = cv2.getRotationMatrix2D(center=tuple([e//2 for e in img.shape]), angle=-angle, scale=1)
    # apply the rotation
    rotated = cv2.warpAffine(src=img, M=rotate_matrix, dsize=img.shape[:2][::-1])
    return rotated, rotate_matrix


def find_polarity(crop, mask, rotated, rotate_matrix, body_size, filepath):
    rotated_mask = cv2.warpAffine(src=mask, M=rotate_matrix, dsize=crop.shape[:2][::-1])
    boolean_mask = rotated_mask == 255
    
    coord_max=np.where(boolean_mask)[1].max()
    coord_min=np.where(boolean_mask)[1].min()
    width = np.where(boolean_mask)[0].max() - np.where(boolean_mask)[0].min()
    height = coord_max - coord_min
    
    top_mask = np.zeros_like(rotated)
    bottom_mask = np.zeros_like(rotated)
    mask_center = tuple([e // 2 for e in rotated.shape[:2][::-1]])
    
    # radius=round(height*0.6)
    # size = int(rotated.shape[0]*0.65 / 2)
    radius=int(0.5 * body_size)
    vertical_offset=int(0.8*body_size)
    
    
    cv2.circle(top_mask, (mask_center[0], mask_center[1] - vertical_offset), radius=radius, color=255, thickness=-1)
    cv2.circle(bottom_mask, (mask_center[0], mask_center[1] + vertical_offset), radius=radius, color=255, thickness=-1)
    
    top_area=cv2.bitwise_and(rotated.copy(), top_mask)
    bottom_area=cv2.bitwise_and(rotated.copy(), bottom_mask)
    
    if conf.DEBUG:
        print(f"Intensity in top circle: {top_area.mean()}")
        print(f"Intensity in bottom circle: {bottom_area.mean()}")
    
    if top_area.mean() < bottom_area.mean():
        flip=True
    else:
        flip=False
    
    if filepath is not None and conf.DEBUG:
        cv2.imwrite(os.path.join(conf.DEBUG_FOLDER, filepath.replace(".png", "_top-area_5.png")), top_area)
        cv2.imwrite(os.path.join(conf.DEBUG_FOLDER, filepath.replace(".png", "_bottom-area_5.png")), bottom_area)    
    return flip

def generate_dataset(experiment, sampling_points, tolerance=100):
    """
    Generate a dataset of frames for POSE labeling
    from a flyhostel experiment
    
    Arguments:
    
    * experiment (str): Path to folder containing an imgstore dataset (set of .mp4, .npz and .extra.json) files
    and an fh_quant folder with two .npy files that summarise the idtrackerai output. These files should be called
    {experiment_datetime}_trajectories.npy and {experiment_datetime}_timestamps.npy
    
    * sampling_points (list): List of integers which define the time at which a frame is sampled in the experiment,
    in ms since the start of the experiment
    
    * tolerance (int): Tolerance between wished sampling time and available time in dataset, in msec
    """
    
    print(experiment)
    
    tr_file = get_trajectories_file(experiment)
    time_file = get_timestamps_file(experiment)
    
    assert os.path.exists(tr_file), f"{tr_file} does not exist"
    assert os.path.exists(time_file), f"{time_file} does not exist"
    
    # load .mp4 dataset
    store = imgstore.new_for_filename(get_store_path(experiment))

    # load trajectories and timestamps
    trajectories = np.load(tr_file)
    trajectories_int = np.array(trajectories, np.int64)
    timestamps = np.load(time_file)
    
    assert len(timestamps) == trajectories.shape[0]
    
    
    lists_of_blobs = {}
    videos = {}
    
    # missing_data = np.isnan(trajectories).any(axis=2).mean(axis=0)
    
    for msec in tqdm.tqdm(sampling_points):

        frame, (frame_number, frame_time) = store.get_nearest_image(msec)
        chunk = store._chunk_n
        
        if not chunk in lists_of_blobs:
            lists_of_blobs[chunk] = idtrackerai.list_of_blobs.ListOfBlobs.load(
                get_collections_file(experiment, chunk)
            )
        
        if not chunk in videos:
            videos[chunk] = get_video_object(experiment, chunk)
    
        try:
            frame_index = store._get_chunk_metadata(chunk)["frame_number"].index(frame_number)
        except ValueError:
            warnings.warn(f"{frame_number} not found in {experiment}-{chunk}")
            continue

        blobs_in_frame = lists_of_blobs[chunk].blobs_in_video[frame_index]
        
        try:
            trajectories_frame=trajectories[frame_number,:, :]
        except IndexError as error:
            logger.debug(error)
            warnings.warn(f"{experiment} is not analyzed after {round(msec/1000/3600, 2)} hours", stacklevel=2)
            break
        
        
        error = np.abs(msec - frame_time)
        
        assert error < tolerance, f"{error} is higher than {tolerance} ms"
    
        for animal in np.arange(trajectories.shape[1]):
        
            # define where the files will be saved
            filepath = os.path.join(
                experiment,
                os.path.join(experiment, str(frame_number).zfill(10), str(animal) + ".png").replace("/", "-")
            ) 
            # pick the data for this animal
            tr = trajectories_frame[animal, :]

            try:
                centroid = tuple([round(e) for e in tr])
            except ValueError:
                # contains NaN
                warnings.warn(f"Animal #{animal} not found in frame {frame_number} of {experiment}", stacklevel=2)
                continue               
                
            contour, other_contours = find_contour(blobs_in_frame, centroid)
            body_size=round(videos[chunk].median_body_length)

            crop_animal_in_time_and_space(frame.copy(), contour, other_contours, centroid, body_size, filepath)