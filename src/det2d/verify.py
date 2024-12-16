import numpy as np
from .keys import Keys

def assert_tracklet_valid(tracklet: dict):
    '''
    Verify that a tracklet is stored correctly. Doesn't return anything, but throws an exception if the tracklet is invalid.
    
    Inputs:
    - tracklet: the tracklet to check
    '''
    assert Keys.start in tracklet.keys(), f"tracklet has no start frame"
    assert isinstance(tracklet[Keys.start], int), f"tracklet start frame must be an int"
    assert Keys.keypoints in tracklet.keys(), f"tracklet has no keypoints"
    assert isinstance(tracklet[Keys.keypoints], np.ndarray), f"tracklet keypoints must be an ndarray"
    assert len(tracklet[Keys.keypoints].shape)==3 and tracklet[Keys.keypoints].shape[-1]==3, f"tracklet keypoints must have shape [F,K,3]"
    assert Keys.prepadding in tracklet.keys() and Keys.postpadding in tracklet.keys(), f"tracklet has no padding information"
    assert isinstance(tracklet[Keys.prepadding], int) and isinstance(tracklet[Keys.postpadding], int), f"tracklet prepadding and postpadding must be int"
    assert tracklet[Keys.prepadding]>=0 and tracklet[Keys.postpadding]>=0, f"tracklet prepadding and postpadding must be at least 0"
    assert tracklet[Keys.prepadding]+tracklet[Keys.postpadding] <= tracklet[Keys.keypoints].shape[0], f"tracklet prepadding and postpadding must add up to at most the number of frames F"

def assert_tracklets_valid(tracklets: dict):
    '''
    Verify that a collection of tracklets is stored correctly. Doesn't return anything, but throws an exception if the stacked tracklets are invalid.
    
    Inputs:
    - tracklets: the tracklets to check
    '''
    for category, category_tracklets in tracklets.items():
        assert isinstance(category, int), f"tracklets categories must be int"
        for id, tracklet in category_tracklets.items():
            assert isinstance(id, int), f"tracklet ids must be int"
            assert_tracklet_velid(tracklet)

def assert_stacked_tracklets_valid(stacked_tracklets: dict):
    '''
    Verify that a collection of stacked tracklet is stored correctly. Doesn't return anything, but throws an exception if the stacked tracklets are invalid.
    
    Inputs:
    - stacked_tracklets: the stacked tracklets to check
    '''
    assert Keys.start in stacked_tracklets.keys(), f"stacked tracklets have no start frame"
    assert isinstance(stacked_tracklets[Keys.start], int), f"stacked tracklets start frame must be an int"
    assert Keys.keypoints in stacked_tracklets.keys(), f"stacked tracklets have no keypoints"
    assert isinstance(stacked_tracklets[Keys.keypoints], np.ndarray), f"stacked tracklets keypoints must be an ndarray"
    assert len(stacked_tracklets[Keys.keypoints].shape)==4 and stacked_tracklets[Keys.keypoints].shape[-1]==3, f"tracklet keypoints must have shape [D,F,K,3]"
    assert Keys.prepadding in stacked_tracklets.keys() and Keys.postpadding in stacked_tracklets.keys(), f"stacked tracklets have no padding information"
    assert isinstance(stacked_tracklets[Keys.prepadding], int) and isinstance(stacked_tracklets[Keys.postpadding], int), f"stacked tracklets prepadding and postpadding must be int"
    assert stacked_tracklets[Keys.prepadding]>=0 and stacked_tracklets[Keys.postpadding]>=0, f"stacked tracklet prepadding and postpadding must be at least 0"
    assert stacked_tracklets[Keys.prepadding]+stacked_tracklets[Keys.postpadding] <= stacked_tracklets[Keys.keypoints].shape[1], f"stacked tracklet prepadding and postpadding must add up to at most the number of frames F"
    
def assert_tracklets_comparable(tracklet1: dict, tracklet2: dict):
    '''
    Verify that two tracklets can be compared. Doesn't return anything, but throws an exception if the tracklets are invalid or can't be compared.
    
    Inputs:
    - tracklet1, tracklet2: the two tracklets to compare
    '''
    assert_tracklet_valid(tracklet1)
    assert_tracklet_valid(tracklet2)
    
    assert tracklet1[Keys.keypoints].shape==tracklet2[Keys.keypoints].shape, f"tracklet1 and tracklet2 must span equally many frames and have equally many keypoints to be comparable, but span {tracklet1[Keys.keypoints].shape[0]} and {tracklet2[Keys.keypoints].shape[0]} frames and have {tracklet1[Keys.keypoints].shape[1]} and {tracklet2[Keys.keypoints].shape[1]} keypoints"
    assert tracklet1[Keys.start]==tracklet2[Keys.start], f"tracklet1 and tracklet2 must start on the same frame to be comparable, but start on {min(tracklet1[Keys.start])} and {min(tracklet2[Keys.start])}"
    