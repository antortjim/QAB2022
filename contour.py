import numpy as np
import cv2
from decorators import timeit
import scipy.ndimage
from confapp import conf

try:
    import local_settings
    conf += local_settings
except ImportError:
    pass


@timeit
def isin(centroid, contour):
    """
    Return True if the centroid is contained by the contour    
    """

    centroid = tuple([int(e) for e in centroid])
    x_max, y_max = contour.max(axis=0).flatten()
    mask=np.zeros((y_max, x_max), np.uint8)
    mask=cv2.drawContours(
        mask,
        [contour],
        # all contours
        -1,
        # white
        255,
        # fill inside
        -1
    )

    try:
        return mask[centroid[1], centroid[0]] == 255
    except:
        return False
    

def compute_contour(img, intensity, area):
    mask = np.zeros_like(img, np.uint8)
    
    mask[
        np.bitwise_and(
            img > intensity[0],
            img < intensity[1]
        )
    ] = 255
    
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    contours_pass=[]
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if area[0] < contour_area < area[1]:
            contours_pass.append(contour)

    contours = sorted(contours_pass, key=lambda contour: cv2.contourArea(contour))

    try:
        contour = contours[-1]
    except IndexError:
        contour = None
    
    return contour

def find_contour(blobs_in_frame, centroid):
    """
    Arguments:
    
    * blobs_in_frame: (list): A list where every element is an idtrackerai.Blob instance
    * centroid (tuple): x and y coordinates of the contour center
    
    Returns:
    
    * contour (np.ndarray): Blob in blobs_in_frame that overlaps the centroid
    """

    other_contours=[]
    contour=None

    for blob in blobs_in_frame:
        overlaps, msec = isin(centroid, blob.contour)
        if conf.DEBUG:
            print(f"isin took {msec} msecs")
        if overlaps:
            contour=blob.contour
        else:
            other_contours.append(blob.contour)
    
    return contour, other_contours


def center_contour(contour):
    try:
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    except ZeroDivisionError:
        cX, cY = contour.mean(axis=0).flatten()
        
    return (cX, cY)



def center_blob_in_mask(mask):
    """
    Return the center of mass of a blob,
    represented by a binary mask
    
    Arguments:
        * cloud (np.ndarray): binary mask
    Returns:
        * center (tuple): x, y coordinates of the center of the blob in the image
        the coordinates are rounded to int
        the coordinates are relative to the bottom left corner
    """
    assert mask.shape[1] != 2
    
    # this function returns the center of mass in y, x coordinates
    # with the origin being placed in the top left cell (0,0)
    # example: a center at 3.5, 4 means
    # between the 4th and 5th rows and exactly on the 5th column
    y, x = scipy.ndimage.center_of_mass(mask)
    y = mask.shape[0] - y
    center = (round(x), round(y))

    return center


def test_isin():
    
    contour = np.array([
        [0, 0],
        [10, 0],
        [10, 10],
        [0, 10]
    ])
    
    contour = contour[:, np.newaxis, :]
    
    centroid = (5, 5)
    overlaps, msec = isin(centroid, contour)
    assert overlaps
