import numpy as np
from .window import tracklet_window
from .keys import Keys
from .verify import assert_tracklets_comparable

def stack_tracklets(tracklets: dict, window: bool=False) -> (np.ndarray, np.ndarray):
    '''
    Stack the keypoints of multiple tracklets from a single category for vectorized computing
    
    Inputs:
    - tracklets: the tracklets dictionary, already indexed by category
    - window: whether to apply windowing to make tracklets span the same frame range. If False, throws an error if the tracklets don't span the same frames
    
    Outputs:
    - stacked_tracklets: dict with keys:
        - ids: the ids of the tracklets
        - start: the start frame
        - keypoints: the keypoints of the tracklets where the first dimension concatenates the tracklets, shape [D,F,K,3] where D is the number of IDs
    '''
    start = tuple(tracklets.values())[0][Keys.start] if len(tracklets) else 0
    stop = start+tuple(tracklets.values())[0][Keys.keypoints].shape[0] if len(tracklets) else 0
    
    for tracklet in tuple(tracklets.values())[1:]:
        if window:
            start = min(start, tracklet[Keys.start])
            stop  = max(stop, tracklet[Keys.start]+tracklet[Keys.keypoints].shape[0])
        else:
            assert_tracklets_comparable(tracklet, tuple(tracklets.values())[0])

    stacked_tracklets = {
        Keys.ids: np.zeros(shape=len(tracklets), dtype=int),
        Keys.start: start,
        Keys.keypoints: np.zeros(shape=(len(tracklets), stop-start, *tuple(tracklets.values())[0][Keys.keypoints].shape[1:]), dtype=float),
        Keys.prepaddings: np.zeros(shape=(len(tracklets),), dtype=int),
        Keys.postpaddings: np.zeros(shape=(len(tracklets),), dtype=int)
    }
    
    for tracklet_index, (tracklet_id, tracklet) in enumerate(tracklets.items()):
        stacked_tracklets[Keys.ids][tracklet_index] = tracklet_id
        stacked_tracklets[Keys.keypoints][tracklet_index] = tracklet_window(tracklet, start, stop-start)[Keys.keypoints]
        stacked_tracklets[Keys.prepaddings][tracklet_index] = tracklet[Keys.start]-stacked_tracklets[Keys.start]
        stacked_tracklets[Keys.postpaddings][tracklet_index] = stop-(tracklet[Keys.start]+tracklet[Keys.keypoints].shape[0])
    
    return stacked_tracklets