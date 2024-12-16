import numpy as np
from .verify import assert_tracklet_valid
from .keys import Keys

def tracklet_window(tracklet: dict, window_start: int, window_length: int) -> dict:
    '''
    Apply windowing to a tracklet
    
    Inputs:
    - tracklet: tracklet to window
    - window_start: window start frame
    - window_length: window number of frames
    
    Outputs:
    - The windowed tracklet. The 'prepadding' and 'postpadding' keys show how many frames were zeropadded before- and after the original keypoints
    '''
    window_frames_before_tracklet, _, gap_frames_at_tracklet_start, window_frames_in_tracklet, gap_frames_at_tracklet_end, _, window_frames_after_tracklet = tracklet_window_overlap(tracklet, window_start, window_length)
    
    windowed_tracklet = {
        Keys.start: window_start,
        Keys.keypoints: np.zeros(shape=(window_length, *tracklet[Keys.keypoints].shape[1:]), dtype=float),
        Keys.prepadding:  max(0, tracklet[Keys.prepadding]+window_frames_before_tracklet-gap_frames_at_tracklet_start),
        Keys.postpadding: max(0, tracklet[Keys.postpadding]+window_frames_after_tracklet-gap_frames_at_tracklet_end)
    }
    
    if window_frames_in_tracklet==0: return windowed_tracklet
    
    windowed_tracklet[Keys.keypoints][window_frames_before_tracklet:window_length-window_frames_after_tracklet] = \
                tracklet[Keys.keypoints][gap_frames_at_tracklet_start:gap_frames_at_tracklet_start+window_frames_in_tracklet]
    
    return windowed_tracklet

def tracklets_window(tracklets: dict, window_start: int, window_length: int) -> dict:
    """
    Apply windowing to multiple tracklets
    
    Inputs:
    - tracklets: the tracklets
    - window_start: window start frame
    - window_length: window number of frames
    
    Outputs:
    - The windowed tracklets, each of which with added 'prepadding' and 'postpadding' keys which show how many frames were zeropadded before- and after the original keypoints
    """
    windowed_tracklets = {}
    for category in tracklets.keys():
        windowed_tracklets[category] = {}
        for id in tracklets[category].keys():
            windowed_tracklets[category][id] = tracklet_window(tracklets[category][id], window_start, window_length)
    
    return windowed_tracklets

def stacked_tracklets_window(stacked_tracklets: dict, window_start: int, window_length: int) -> dict:
    '''
    Apply windowing to multiple stacked tracklets
    
    Inputs:
    - stacked_tracklets: stacked tracklets to window, already indexed by category
    - window_start: window start frame
    - window_length: window number of frames
    
    Outputs:
    - The windowed stacked tracklets
    '''
    window_frames_before_tracklets, _, gap_frames_at_tracklets_start, window_frames_in_tracklets, gap_frames_at_tracklets_end, _, window_frames_after_tracklets = tracklet_window_overlap({
        Keys.start: stacked_tracklets[Keys.start],
        Keys.keypoints: np.zeros(shape=stacked_tracklets[Keys.keypoints].shape[1:], dtype=float),
        Keys.prepadding: int(min(stacked_tracklets[Keys.prepaddings])) if len(stacked_tracklets[Keys.prepaddings]) else 0,
        Keys.postpadding: int(min(stacked_tracklets[Keys.postpaddings])) if len(stacked_tracklets[Keys.postpaddings]) else 0
    }, window_start, window_length)
    
    windowed_stacked_tracklets = {
        Keys.start: window_start,
        Keys.keypoints: np.zeros(shape=(stacked_tracklets[Keys.keypoints].shape[0], window_length, *stacked_tracklets[Keys.keypoints].shape[2:]), dtype=float),
        Keys.prepaddings:  np.maximum(0, stacked_tracklets[Keys.prepaddings]+window_frames_before_tracklets-gap_frames_at_tracklets_start),
        Keys.postpaddings: np.maximum(0, stacked_tracklets[Keys.postpaddings]+window_frames_after_tracklets-gap_frames_at_tracklets_end)
    }
    
    if window_frames_in_tracklets==0: return windowed_stacked_tracklets
    
    windowed_stacked_tracklets[Keys.keypoints][:,window_frames_before_tracklets:window_length-window_frames_after_tracklets] = \
            stacked_tracklets[Keys.keypoints][:,gap_frames_at_tracklets_start:gap_frames_at_tracklets_start+window_frames_in_tracklets]
    
    return windowed_stacked_tracklets

def tracklet_window_overlap(tracklet: dict, window_start: int, window_length: int) -> tuple:
    '''
    Calculate several values detailing the overlap of a window with a tracklet
    
    Inputs:
    - tracklet: the tracklet
    - window_start: window start frame
    - window_length: window number of frames
    
    Outputs:
    - Number of frames in the window before the tracklet starts
    - Number of frames after the window, before the tracklet (0 if they overlap at all)
    - Start of the window overlap within the tracklet, value 0 corresponds to the start of the tracklet
    - Length of the window overlap within the tracklet
    - Number of frames after the tracklet before the window (0 if they overlap at all)
    - Number of frames in the window after the tracklet ends
    '''
    assert_tracklet_valid(tracklet)
    assert window_length>0, f"window_length must be greater than 0, but was {window_length}"
    
    window_stop = window_start+window_length
    
    tracklet_start = tracklet[Keys.start]
    tracklet_length = tracklet[Keys.keypoints].shape[0]
    tracklet_stop = tracklet_start+tracklet_length
    
    window_frames_before_tracklet     = min(window_length, max(0, tracklet_start-window_start))
    window_frames_after_tracklet      = min(window_length, max(0, window_stop-tracklet_stop))
    
    gap_frames_before_tracklet        = max(0, tracklet_start-window_stop)
    gap_frames_after_tracklet         = max(0, window_start-tracklet_stop)
    
    gap_frames_at_tracklet_start = min(tracklet_length, max(0, window_start-tracklet_start))
    window_frames_in_tracklet    = window_length-window_frames_before_tracklet-window_frames_after_tracklet
    gap_frames_at_tracklet_end   = min(tracklet_length, max(0, tracklet_stop-window_stop))
    
    return window_frames_before_tracklet, gap_frames_before_tracklet, gap_frames_at_tracklet_start, window_frames_in_tracklet, gap_frames_at_tracklet_end, gap_frames_after_tracklet, window_frames_after_tracklet