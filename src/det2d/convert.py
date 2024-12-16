import numpy as np
from types import FunctionType
from tqdm import tqdm
from .fill import zero_tracklet_gaps
from .verify import assert_tracklet_valid
from .keys import Keys

def detections2tracklets(poses: dict, fill_function: FunctionType=zero_tracklet_gaps, confidence_threshold: float=0, frame_range: range=None, verbose: bool=False) -> dict:
    '''
    Transform an imported dictionary of pose detections to a dictionary of tracklets, ensuring that tracklet poses are sorted by frame number
    
    Inputs:
    - poses: dictionary of poses
    - fill_function: function to use to fill gaps in the tracklet
    - confidence_threshold: keypoint confidence threshold to use in fill_function
    - frame_range: range of frames to convert to a tracklet, must have step size 1
    - verbose: if True, show a progress bar
    
    Outputs:
    - dictionary of tracklets
    '''
    assert callable(fill_function), f"fill_function must be callable, but has type {type(fill_function)}"
    assert frame_range is None or frame_range.step==1, f"frame_range must have a step size of 1, but this was {frame_range.step}"
    
    if frame_range is not None: pose_items = {frame: poses[frame] for frame in frame_range if frame in poses.keys()}.items()
    else: pose_items = poses.items()
    
    if verbose: pose_items = tqdm(pose_items, desc="detections -> tracklets")
    
    tracklets = {}
    
    for frame, frame_dictionary in pose_items:
        for category, category_list in frame_dictionary.items():
            if category not in tracklets.keys(): tracklets[category] = {}
            for pose in category_list:
                pose_id, pose_keypoints = pose[Keys.id], pose[Keys.keypoints]
                
                if pose_id not in tracklets[category].keys():
                    tracklets[category][pose_id]={
                        Keys.keypoints: np.zeros(shape=(0,*pose_keypoints.shape)),
                        Keys.frames: np.zeros(shape=(0,), dtype=int)
                    }
                
                tracklets[category][pose_id][Keys.keypoints] = np.vstack((tracklets[category][pose_id][Keys.keypoints], pose[Keys.keypoints][None,:,:]))
                tracklets[category][pose_id][Keys.frames]    = np.append(tracklets[category][pose_id][Keys.frames], frame)
    
    for category, category_dict in tracklets.items():
        for tracklet_id in category_dict.keys():
            category_dict[tracklet_id] = fill_function(category_dict[tracklet_id], confidence_threshold)
    
    return tracklets
    
def tracklets2detections(tracklets: dict, verbose: bool=False) -> dict:
    '''
    Transform a dictionary of tracklets to a dictionary of pose detections, ensuring that poses are sorted by frame number
    
    Inputs:
    - tracklets: dictionary of tracklets
    - verbose: if True, show a progress bar
    
    Outputs:
    - dictionary of pose detections
    '''
    detections = {}
    for category, category_dictionary in tracklets.items():
        category_dictionary_items = tqdm(category_dictionary.items(), desc=f"tracklets -> detections ({category})") if verbose else category_dictionary.items()
        for tracklet_id, tracklet in category_dictionary_items:
            assert_tracklet_valid(tracklet)
            
            keypoints = tracklet[Keys.keypoints].copy()
            frames = np.arange(keypoints.shape[0])+tracklet[Keys.start]
            nonzero_pose_mask = np.sum(keypoints[:,:,2], axis=1)>0
            keypoints = keypoints[nonzero_pose_mask]
            frames = frames[nonzero_pose_mask]
            for frame, keypoint in zip(frames, keypoints):
                if frame not in detections.keys(): detections[frame] = {}
                if category not in detections[frame].keys(): detections[frame][category] = []
                detections[frame][category].append({
                    Keys.keypoints: keypoint,
                    Keys.id: tracklet_id
                })
                
    detections = dict(sorted(detections.items()))
    
    return detections