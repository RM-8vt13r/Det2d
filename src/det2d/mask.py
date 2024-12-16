import numpy as np
from .keys import Keys

def tracklet_confidence_mask(tracklet: dict, confidence_threshold: float=0) -> np.ndarray:
    '''
    Given a tracklet, find all positions where keypoints satisfy a confidence threshold and are greater than 0
    
    Inputs:
    - tracklet: the tracklet
    - confidence_threshold: the confidence threshold
    
    Outputs:
    - array which is True where keypoints satisfy the confidence threshold and False elsewhere, shape [F,K] where F is the number of frames and K the number of keypoints
    '''
    assert confidence_threshold >= 0 and confidence_threshold <= 1, f"confidence_threshold must be at least 0 and at most 1, but was {confidence_threshold}"
    return (tracklet[Keys.keypoints][...,2] >= confidence_threshold) & (tracklet[Keys.keypoints][...,2] > 0)

def tracklet_unpadded_mask(tracklet: dict) -> np.ndarray:
    '''
    Given a tracklet or stacked tracklets, find all positions where keypoints are original, i.e., not padded by windowing
    
    Inputs:
    - tracklet: the tracklet or stacked tracklets
    
    Outputs:
    - array which is True where the tracklet or stacked tracklets weren't padded and False elsewhere, shape [F,] where F is the number of frames
    '''
    frames = np.arange(tracklet[Keys.keypoints].shape[0])
    mask = (frames >= tracklet[Keys.prepadding]) & (tracklet[Keys.keypoints].shape[0]-1-frames >= tracklet[Keys.postpadding])
    return mask

def tracklet_confidence_and_unpadded_mask(tracklet: dict, confidence_threshold: float=0) -> np.ndarray:
    '''
    Given a tracklet, find all positions where keypoints satisfy a confidence threshold and weren't padded.
    
    Inputs:
    - tracklet: the tracklet
    - confidence_threshold: the confidence threshold
    
    Outputs:
    - array which is True where keypoints satisfy the confidence_threshold and weren't padded, and False elsewhere. Shape [F,K] where F is the number of frames and K the number of keypoints
    '''
    return tracklet_confidence_mask(tracklet, confidence_threshold) & tracklet_unpadded_mask(tracklet)[:,None]

def stacked_tracklets_confidence_mask(stacked_tracklets: dict, confidence_threshold: float=0) -> np.ndarray:
    '''
    Given stacked tracklets, find all positions where keypoints satisfy a confidence threshold and are greater than 0
    
    Inputs:
    - stacked_tracklets: the stacked tracklets
    - confidence_threshold: the confidence threshold
    
    Outputs:
    - array which is True where keypoints satisfy the confidence threshold and False elsewhere, shape [D,F,K] where D is the number of stacked tracklets, F is the number of frames and K the number of keypoints
    '''
    return tracklet_confidence_mask(stacked_tracklets, confidence_threshold)

def stacked_tracklets_unpadded_mask(stacked_tracklets: dict) -> np.ndarray:
    '''
    Given stacked tracklets, find all positions where keypoints are original, i.e., not padded by windowing
    
    Inputs:
    - stacked_tracklets: the stacked tracklets
    
    Outputs:
    - array which is True where the tracklet or stacked tracklets weren't padded and False elsewhere, shape [D,F] where D is the number of stacked tracklets and F is the number of frames
    '''
    frames = np.arange(stacked_tracklets[Keys.keypoints].shape[1])
    mask = (frames[None,:] >= stacked_tracklets[Keys.prepaddings][:,None]) & (stacked_tracklets[Keys.keypoints].shape[1]-1-frames[None,:] >= stacked_tracklets[Keys.postpaddings][:,None])
    return mask

def stacked_tracklets_confidence_and_unpadded_mask(stacked_tracklets: dict, confidence_threshold: float=0) -> np.ndarray:
    '''
    Given stacked tracklets, find all positions where keypoints satisfy a confidence threshold and weren't padded.
    
    Inputs:
    - stacked_tracklets: the stacked tracklets
    - confidence_threshold: the confidence threshold
    
    Outputs:
    - array which is True where keypoints satisfy the confidence_threshold and weren't padded, and False elsewhere. Shape [D,F,K] where D is the number of tracklets, F is the number of frames and K the number of keypoints
    '''
    return stacked_tracklets_confidence_mask(stacked_tracklets, confidence_threshold) & stacked_tracklets_unpadded_mask(stacked_tracklets)[:,:,None]