from types import FunctionType
import numpy as np
import scipy as sp
from .keys import Keys

transpose_frames_keypoints = (1,0,2)

def interpolate_tracklet_gaps(tracklet: dict, confidence_threshold: float=0) -> dict:
    '''
    Use linear spline interpolation to fill missing frames in a single tracklet (after indexing tracklets by confidence and id).
    New frames and keypoints with low confidence are interpolated, as long as they have two neighbours with sufficient confidence.
    
    Inputs:
    - tracklet: tracklet to fill gaps in
    - confidence_threshold: threshold between 0 and 1, below which a keypoint is assumed not to be detected
    
    Outputs:
    - tracklet whose gaps were interpolated
    '''
    filled_tracklet, keyframes = prepare_filled_tracklet(tracklet, confidence_threshold)
    start, keypoints = filled_tracklet[Keys.start], filled_tracklet[Keys.keypoints]
    
    interpolated_tracklet = {
        Keys.keypoints: np.zeros_like(keypoints),
        Keys.start: start,
        Keys.prepadding: filled_tracklet[Keys.prepadding],
        Keys.postpadding: filled_tracklet[Keys.postpadding]
    }
    
    for k, (keypoints_per_class, keyframes_per_class) in enumerate(zip(keypoints.transpose(transpose_frames_keypoints), keyframes.T)):
        keyframes_per_class = np.nonzero(keyframes_per_class)[0]
        
        if len(keyframes_per_class) >= 2:
            interpolator = sp.interpolate.interp1d(keyframes_per_class, keypoints_per_class[keyframes_per_class], axis=0, bounds_error=False, fill_value=0)
            interpolated_tracklet[Keys.keypoints][:,k] = interpolator(range(keypoints.shape[0]))
        interpolated_tracklet[Keys.keypoints][keyframes_per_class,k] = keypoints_per_class[keyframes_per_class]
        
    return interpolated_tracklet
    
def zero_tracklet_gaps(tracklet: dict, confidence_threshold: float=0) -> dict:
    '''
    Add missing frames to a single tracklet (after indexing tracklets by confidence and id).
    New frames and keypoints with low confidence are written with 0 coordinates and confidence.
    
    Inputs:
    - tracklet: tracklet to fill gaps in
    - confidence_threshold: threshold between 0 and 1, below which a keypoint is assumed not to be detected
    
    Outputs:
    - tracklet whose gaps were filled with zeroes
    '''
    filled_tracklet, _ = prepare_filled_tracklet(tracklet, confidence_threshold)
    return filled_tracklet
    
def prepare_filled_tracklet(tracklet: dict, confidence_threshold: float=0) -> dict:
    '''
    Prepare for gap filling or interpolation by extracting keypoints from a tracklet, adding missing frames, and setting low-confidence keypoints to zero.
    
    Inputs:
    - tracklet: tracklet to fill gaps in
    - confidence_threshold: threshold between 0 and 1, below which a keypoint is assumed not to be detected
    
    Outputs:
    - tracklet whose gaps were filled with zeroes
    - Mask which is False where keypoint gaps were filled and True elsewhere, shape [F,K] where F is the number of frames and K the number of keypoints.
    '''
    assert confidence_threshold>=0 and confidence_threshold<=1, f"confidence_threshold must be between 0 and 1, but was {confidence_threshold}"
    
    if Keys.start in tracklet.keys():
        return tracklet, tracklet[Keys.keypoints][:,:,2] >= confidence_threshold
    
    keypoints = tracklet[Keys.keypoints]
    frames = tracklet[Keys.frames] if Keys.frames in tracklet.keys() else np.arange(len(keypoints))+tracklet[Keys.start]
    
    sort_frames = np.argsort(frames)
    frames = frames[sort_frames]
    keypoints = keypoints.copy()[sort_frames]
    
    detected = keypoints[:,:,2]>=confidence_threshold
    keypoints[~detected]=0
    
    filled_frames = np.arange(frames[0], frames[-1]+1)
    filled_keypoints = np.zeros(shape=(len(filled_frames), keypoints.shape[1], 3), dtype=float)
    filled_keypoints[frames-frames[0]] = keypoints
    
    keypoint_keyframes = np.zeros(shape=(*filled_frames.shape, keypoints.shape[1]), dtype=bool)
    keypoint_keyframes[frames-frames[0]] = True
    keypoint_keyframes[frames-frames[0]][~detected] = False
    
    filled_tracklet = {
        Keys.start: int(filled_frames[0]),
        Keys.keypoints: filled_keypoints,
        Keys.prepadding: tracklet[Keys.prepadding] if Keys.prepadding in tracklet.keys() else 0,
        Keys.postpadding: tracklet[Keys.postpadding] if Keys.postpadding in tracklet.keys() else 0
    }
    
    return filled_tracklet, keypoint_keyframes
    
def interpolate_tracklets_gaps(tracklets: dict, confidence_threshold: float=0) -> dict:
    '''
    Use linear spline interpolation to fill missing frames in multiple tracklets (after indexing tracklets by category, confidence and id).
    New frames and keypoints with low confidence are interpolated, as long as they have two neighbours with sufficient confidence.
    
    Inputs:
    - tracklets: tracklets to fill gaps in
    - confidence_threshold: threshold between 0 and 1, below which a keypoint is assumed not to be detected
    
    Outputs:
    - tracklets whose gaps were interpolated
    '''
    return fill_tracklets(interpolate_tracklet_gaps, tracklets, confidence_threshold)

def zero_tracklets_gaps(tracklet: dict, confidence_threshold: float=0) -> dict:
    '''
    Add missing frames to multiple tracklets (after indexing tracklets by category, confidence and id).
    New frames and keypoints with low confidence are written with 0 coordinates and confidence.
    
    Inputs:
    - tracklets: tracklets to fill gaps in
    - confidence_threshold: threshold between 0 and 1, below which a keypoint is assumed not to be detected
    
    Outputs:
    - tracklets whose gaps were filled with zeroes
    '''
    return fill_tracklets(zero_tracklet_gaps, tracklets, confidence_threshold)

def fill_tracklets(fill_function: FunctionType, tracklets: dict, confidence_threshold: float=0) -> dict:
    '''
    Fill missing frames in multiple tracklets (after indexing tracklets by category, confidence and id).
    
    Inputs:
    - tracklets: tracklets to fill gaps in
    - confidence_threshold: threshold between 0 and 1, below which a keypoint is assumed not to be detected
    - fill_function: function to use to fill tracklet gaps
    
    Outputs:
    - tracklets whose gaps were interpolated
    '''
    filled_tracklets = {}
    for category, category_tracklets in tracklets.items():
        filled_tracklets[category] = {}
        for tracklet_id, tracklet in category_tracklets.items():
            filled_tracklets[category][tracklet_id] = fill_function(tracklet, confidence_threshold)
            
    return filled_tracklets