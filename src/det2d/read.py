from types import FunctionType
import os
import json
import numpy as np

from .convert import detections2tracklets
from .fill import zero_tracklet_gaps
from .keys import Keys

def read_tracklets(path: str, fill_function: FunctionType=zero_tracklet_gaps, confidence_threshold: float=0, frame_range: range=None, verbose: bool=False) -> dict:
    '''
    Import tracklet data from a .det2d.json file.
    
    Tracklet structure:
    {
    <category0>: {
        <id0>: {
            "start":<frame0>,
            "keypoints":<keypoints0>,
            ...
        },
        <id1>: {...},
        ...
    ],
    <category1>: ...,
    ...
    }
    
    Where each category dictionary has T elements where T is the number of tracklets.
    "start" contains the first frame number where the pose appears.
    "keypoints" contains an array of shape [F,K,3] where F is the number of frames, K is the number of keypoints, and the last axis contains x,y,confidence.
    
    Inputs:
    - path: path to the .det2d.json file to read
    - fill_function: function to use to fill gaps in the tracklet
    - confidence_threshold: keypoint confidence threshold to use in fill_function
    - frame_range: range of frames to convert to a tracklet, must have step size 1
    - verbose: if True, show a progress bar
    
    Outputs:
    - Tracklet data as described above
    '''
    poses = read_detections(path, frame_range)
    tracklets = detections2tracklets(poses, fill_function, confidence_threshold, frame_range, verbose=verbose)
    return tracklets
    
def read_detections(path: str, frame_range: range=None) -> dict:
    '''
    Import pose detection data from a .det2d.json file.
    
    Pose structure:
    {
    <frame0>: {
        <category0>: [
            {"keypoints": <keypoints0>, "id": <id0>},
            ...
        ],
        <category1>: ...,
        ...
    },
    <frame1>: ...,
    ...
    }
    
    Where each frame dictionary has C elements where C is the number of categories.
    Each category list has P elements where P is the number of poses.
    "id" contains a scalar tracking identity.
    "keypoints" contains an array of shape [K,3], where K is the number of keypoints in the pose and the last dimension contains x,y,confidence.
    It contains keypoint x and y coordinates and confidence repeatedly in that order.
    
    Inputs:
    - path: path to the .det2d.json file to read
    - frame_range: range of frames to read, must have step size 1
    
    Outputs:
    - Pose data as described above.
    '''
    assert isinstance(path, str), f"Argument 'path' must be a str, but was a {type(path)}"
    assert os.path.isfile(path), f"Path \"{path}\" does not exist"
    assert frame_range is None or frame_range.step==1, f"frame_range must have a step size of 1, but this was {frame_range.step}"
    
    with open(path, 'r') as file:
        poses = json.load(file, object_hook=lambda dictionary: {int(key) if key.isdigit() else key: np.array(value) if isinstance(value, list | tuple) else value for key, value in dictionary.items()})
    
    if frame_range is not None: poses = {frame: poses[frame] for frame in frame_range if frame in poses.keys()}
    
    for frame, frame_dict in poses.items():
        _process_frame_detections_dict(frame, frame_dict)
        
        # frame_index += 1 #
    
    return poses
    
def _process_frame_detections_dict(frame: int, frame_detections_dict: str) -> dict:
    """
    Verify a single det2d line after indexing by frame number, and reshape keypoints from [3*K] to [K,3]. Modifies the dict directly, and does not return anything.
    
    Inputs:
    - frame: the frame number, used in error messages
    - frame_detections_dict: loaded dictionary after indexing by frame number
    """
    for category, category_list in frame_detections_dict.items():
        for p, pose in enumerate(category_list):
            assert Keys.id in pose.keys(), f"Pose does not have id (frame {frame}, category {category}, pose {p})"
            assert Keys.keypoints in pose.keys(), f"Pose does not have keypoints (frame {frame}, category {category}, id {pose[Keys.id]})"
            assert np.isclose(len(pose[Keys.keypoints])%3, 0), f"Last axis dimension of keypoints must be divisible by 3, but was {len(pose[Keys.keypoints])} (frame {frame}, category {category}, id {pose[Keys.id]})"
            pose[Keys.keypoints] = pose[Keys.keypoints].reshape((-1,3))