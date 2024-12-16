import sys
sys.path.append('src')
import datetime
import numpy as np

import det2d

categories_path = "./cats.json"
keypoints_path = "./testing/20240716-150900_20240716-163915_test.det2d.json"

def test_categories():
    categories = det2d.read_categories(categories_path)
    assert 'Human' in categories.__dir__(), "'Human' not present in categories"
    assert categories.Human==0, "Category indices incorrect"
    
    keypoints = det2d.read_category_keypoints(categories_path, categories.Human)
    assert keypoints.nose==0 and keypoints.rankle==16, "Keypoint indices incorrect"
    
    category_details = det2d.read_category_details(categories_path)

def test_read():
    categories = det2d.read_categories(categories_path)
    
    detections = det2d.read_detections(keypoints_path)
    tracklets  = det2d.read_tracklets(keypoints_path)
    
    assert set(tracklets.keys())=={categories.Human,}, f"Tracklet categories read incorrectly; should be ({categories.Human},), but was {tuple(tracklets.keys())}"
    assert set(tracklets[categories.Human].keys())=={0,1,2}, f"Tracklet ids read incorrectly; should be (0,1,2), but was {tuple(tracklets[categories.Human].keys())}"
    for tracklet in tracklets[categories.Human].values(): assert set(tracklet.keys())=={det2d.Keys.start, det2d.Keys.keypoints, det2d.Keys.prepadding, det2d.Keys.postpadding}, f"Tracklet keys read incorrectly; should be (det2d.Keys.start, det2d.Keys.keypoints, det2d.Keys.keyframes), but was {tuple(tracklet.keys())}"
    assert tracklets[categories.Human][0][det2d.Keys.keypoints].shape==(5,3,3), f"Tracklet with id 0 should have keypoints of shape (5,3,3), but this was {tracklets[categories.Human][0][det2d.Keys.keypoints].shape}"
    assert tracklets[categories.Human][1][det2d.Keys.keypoints].shape==(4,3,3), f"Tracklet with id 1 should have keypoints of shape (4,3,3), but this was {tracklets[categories.Human][0][det2d.Keys.keypoints].shape}"
    assert tracklets[categories.Human][2][det2d.Keys.keypoints].shape==(2,3,3), f"Tracklet with id 2 should have keypoints of shape (2,3,3), but this was {tracklets[categories.Human][0][det2d.Keys.keypoints].shape}"
    
    det2d.assert_tracklet_valid(tracklets[categories.Human][0])
    det2d.assert_tracklet_valid(tracklets[categories.Human][1])
    det2d.assert_tracklet_valid(tracklets[categories.Human][2])
    
    detections = det2d.read_detections(keypoints_path, range(11,14))
    assert set(detections.keys())==set(range(11,14)), f"Detections were read outside specified frame range"
    assert set([pose[det2d.Keys.id] for pose in detections[11][categories.Human]])=={0,1}, f"Wrong detections read on frame 11"
    assert set([pose[det2d.Keys.id] for pose in detections[12][categories.Human]])=={0,2}, f"Wrong detections read on frame 12"
    assert set([pose[det2d.Keys.id] for pose in detections[13][categories.Human]])=={0,1,2}, f"Wrong detections read on frame 13"

def test_convert():
    categories = det2d.read_categories(categories_path)
    
    detections = det2d.read_detections(keypoints_path)
    tracklets  = det2d.read_tracklets(keypoints_path)
    
    converted_tracklets = det2d.detections2tracklets(det2d.tracklets2detections(tracklets))
    for category_tracklets, category_converted_tracklets in zip(tracklets.values(), converted_tracklets.values()):
        for tracklet_id in category_tracklets.keys():
            assert tracklet_id in category_converted_tracklets.keys(), f"Tracklet ID {tracklet_id} disappeared during conversion"
            tracklet = category_tracklets[tracklet_id]
            converted_tracklet = category_converted_tracklets[tracklet_id]
            assert np.all(tracklet[det2d.Keys.keypoints]==converted_tracklet[det2d.Keys.keypoints]) and \
                tracklet[det2d.Keys.start]==converted_tracklet[det2d.Keys.start] and \
                tracklet[det2d.Keys.prepadding]==converted_tracklet[det2d.Keys.prepadding] and \
                tracklet[det2d.Keys.postpadding]==converted_tracklet[det2d.Keys.postpadding], "detections2tracklets and tracklets2detections are not each other's inverse"
    
    ranged_tracklets = det2d.detections2tracklets(detections, frame_range=range(12,14))
    for tracklet_id in tracklets[categories.Human].keys():
        tracklet = tracklets[categories.Human][tracklet_id]
        ranged_tracklet = ranged_tracklets[categories.Human][tracklet_id]
        assert ranged_tracklet[det2d.Keys.start] >= 12, f"Ranged tracklet {tracklet_id} starts before range start"
        assert ranged_tracklet[det2d.Keys.start] + ranged_tracklet[det2d.Keys.keypoints].shape[0] <= 14, f"Ranged tracklet {tracklet_id} stops after range stop"
        assert np.all(ranged_tracklet[det2d.Keys.keypoints]==tracklet[det2d.Keys.keypoints][ranged_tracklet[det2d.Keys.start]-tracklet[det2d.Keys.start]:ranged_tracklet[det2d.Keys.start]+ranged_tracklet[det2d.Keys.keypoints].shape[0]-tracklet[det2d.Keys.start]]), "Ranged tracklet keypoints don't match original tracklet keypoints"
    
def test_fill():
    categories = det2d.read_categories(categories_path)
    tracklets = det2d.read_tracklets(keypoints_path)
    
    interpolated_tracklets = det2d.interpolate_tracklets_gaps(tracklets, confidence_threshold=0.5)
    assert np.all(np.isclose(interpolated_tracklets[categories.Human][0][det2d.Keys.keypoints], [
        [[0,0,1.0],     [1,1,1.0], [2,2,1.0]],
        [[1.5,1.5,1.0], [1,1,1.0], [1.5,1.5,1.0]],
        [[3,3,1.0],     [1,1,1.0], [1,1,1.0]],
        [[0,0,1.0],     [1,1,1.0], [0.5,0.5,1.0]],
        [[0,0,0.0],     [2,2,1.0], [0,0,1.0]]
    ])), f"Tracklet with id 0 was not interpolated correctly (keypoints)"
    
    assert np.all(np.isclose(interpolated_tracklets[categories.Human][1][det2d.Keys.keypoints], [
        [[0,3,1],   [1,2,1],    [2,1,1]],
        [[1,2,0.7], [2,3,0.8],  [1,2,0.6]],
        [[2,1,0.8], [3,4,0.75], [1,2,0.6]],
        [[3,0,0.9], [4,5,0.7],  [1,2,0.6]]
    ])), f"Tracklet with id 1 was not interpolated correctly (keypoints)"
    
    assert np.all(np.isclose(interpolated_tracklets[categories.Human][2][det2d.Keys.keypoints], [
        [[5,5,1], [6,6,1], [7,7,1]],
        [[0,0,0], [0,0,0], [10,10,1]]
    ])), f"Tracklet with id 2 was not interpolated correctly (keypoints)"

def test_window():
    categories = det2d.read_categories(categories_path)
    tracklets = det2d.read_tracklets(keypoints_path)
    
    windowed_tracklet = det2d.tracklet_window(tracklets[categories.Human][2], window_start=10, window_length=5)
    assert windowed_tracklet[det2d.Keys.start]==10, f"Windowed tracklet start frame should be 10, but was {windowed_tracklet[det2d.Keys.start]}"
    assert windowed_tracklet[det2d.Keys.keypoints].shape==(5,3,3), f"Windowed tracklet keypoints should have shape (5,3,3), but this was {windowed_tracklet[det2d.Keys.keypoints].shape}"
    assert np.all(windowed_tracklet[det2d.Keys.keypoints][0:2,:,:]==0), f"Windowed tracklet start should be filled with zeroes, but wasn't"
    assert np.all(windowed_tracklet[det2d.Keys.keypoints][4,:,:]==0), f"Windowed tracklet end should be filled with zeroes, but wasn't"
    assert windowed_tracklet[det2d.Keys.prepadding]==2, f"Windowed tracklet should have prepadding value 2, but this was {windowed_tracklet[det2d.Keys.prepadding]}"
    assert windowed_tracklet[det2d.Keys.postpadding]==1, f"Windowed tracklet should have postpadding value 1, but this was {windowed_tracklet[det2d.Keys.postpadding]}"
    
    windowed_windowed_tracklet = det2d.tracklet_window(windowed_tracklet, window_start=9, window_length=7)
    assert windowed_windowed_tracklet[det2d.Keys.prepadding]==3, f"Windowed windowed tracklet should have prepadding value 3, but this was {windowed_windowed_tracklet[det2d.Keys.prepadding]}"
    assert windowed_windowed_tracklet[det2d.Keys.postpadding]==2, f"Windowed windowed tracklet should have postpadding value 2, but this was {windowed_windowed_tracklet[det2d.Keys.postpadding]}"
    
    windowed_tracklet2 = det2d.tracklet_window(tracklets[categories.Human][1], window_start=10, window_length=5)
    det2d.assert_tracklets_comparable(windowed_tracklet, windowed_tracklet2)
    
    windowed_tracklets = det2d.tracklets_window(tracklets, window_start=8, window_length=8)

def test_mask():
    categories = det2d.read_categories(categories_path)
    tracklets = det2d.read_tracklets(keypoints_path)
    
    tracklet = tracklets[categories.Human][0]
    interpolated_tracklet = det2d.interpolate_tracklet_gaps(tracklet, confidence_threshold=0.5)
    windowed_tracklet = det2d.tracklet_window(tracklet, window_start=8, window_length=8)
    
    assert np.all(det2d.tracklet_confidence_mask(tracklet, confidence_threshold=0.5)==[
        [True, True, True],
        [False, False, False],
        [True, False, False],
        [True, True, False],
        [False, True, True]
    ]), f"tracklet_confidence_mask returns wrong mask for tracklet with id 0"
    
    assert np.all(det2d.tracklet_confidence_mask(interpolated_tracklet, confidence_threshold=0.5)==[
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [False, True, True]
    ]), f"tracklet_confidence_mask returns wrong mask for interpolated tracklet with id 0"
    
    assert np.all(det2d.tracklet_confidence_mask(windowed_tracklet, confidence_threshold=0.5)==[
        [False, False, False],
        [False, False, False],
        [True, True, True],
        [False, False, False],
        [True, False, False],
        [True, True, False],
        [False, True, True],
        [False, False, False]
    ]), f"tracklet_confidence_mask returns wrong mask for windowed tracklet with id 0"
    
    assert np.all(det2d.tracklet_unpadded_mask(windowed_tracklet)==[False, False, True, True, True, True, True, False]), f"tracklet_unpadded_mask returns wrong mask for windowed tracklet with ID 0"
    assert np.all(det2d.tracklet_confidence_and_unpadded_mask(windowed_tracklet, confidence_threshold=0.5)==[
        [False, False, False],
        [False, False, False],
        [True, True, True],
        [False, False, False],
        [True, False, False],
        [True, True, False],
        [False, True, True],
        [False, False, False]
    ]), f"tracklet_confidence_unpadded_mask returns wrong mask for windowed tracklet with id 0"
    
def test_stack():
    categories = det2d.read_categories(categories_path)
    tracklets = det2d.read_tracklets(keypoints_path)
    human_tracklets = tracklets[categories.Human]
    
    try:
        det2d.stack_tracklets(human_tracklets)
    except:
        pass
    else:
        raise AssertionError("stack_tracklets should throw an error for unequal tracklet lengths, but didn't")
    
    stacked_human_tracklets = det2d.stack_tracklets(human_tracklets, window=True)
    assert set(stacked_human_tracklets[det2d.Keys.ids])=={0,1,2}, f"Expected stacked_human_tracklets['ids'] to be (0,1,2), but it was {stacked_human_tracklets[det2d.Keys.ids]}"
    assert stacked_human_tracklets[det2d.Keys.start]==10, f"Expected stacked_human_tracklets['start'] to be 10, but it was {stacked_human_tracklets[det2d.Keys.start]}"
    assert np.all(stacked_human_tracklets[det2d.Keys.prepaddings]==[0,0,2]), f"Expected stacked_human_tracklets['prepadding'] to be [0,0,2], but it was {stacked_human_tracklets[det2d.Keys.prepaddings]}"
    assert np.all(stacked_human_tracklets[det2d.Keys.postpaddings]==[0,1,1]), f"Expected stacked_human_tracklets['postpadding'] to be [0,1,1], but it was {stacked_human_tracklets[det2d.Keys.postpaddings]}"
    assert stacked_human_tracklets[det2d.Keys.keypoints].shape==(3,5,3,3), f"Expected stacked_human_tracklets['keypoints'] to have shape (n_tracklets,n_frames,n_keypoints,3)=(3,5,3,3), but it was {stacked_human_tracklets[det2d.Keys.keypoints].shape}"

def test_stacked_window():
    categories = det2d.read_categories(categories_path)
    tracklets = det2d.read_tracklets(keypoints_path)
    stacked_tracklets = det2d.stack_tracklets(tracklets[categories.Human], window=True)
    
    windowed_stacked_tracklets = det2d.stacked_tracklets_window(stacked_tracklets, window_start=8, window_length=8)
    assert windowed_stacked_tracklets[det2d.Keys.start]==8, f"Windowed stacked tracklets start frame should be 8, but was {windowed_stacked_tracklets[det2d.Keys.start]}"
    assert windowed_stacked_tracklets[det2d.Keys.keypoints].shape==(3,8,3,3), f"Windowed stacked tracklets keypoints should have shape (3,8,3,3), but this was {windowed_stacked_tracklets[det2d.Keys.keypoints].shape}"
    assert np.all(windowed_stacked_tracklets[det2d.Keys.keypoints][:,0:2,:,:]==0), f"Windowed stacked tracklets start should be filled with zeroes, but wasn't"
    assert np.all(windowed_stacked_tracklets[det2d.Keys.keypoints][:,7,:,:]==0), f"Windowed stacked tracklets end should be filled with zeroes, but wasn't"
    assert np.all(windowed_stacked_tracklets[det2d.Keys.prepaddings]==stacked_tracklets[det2d.Keys.prepaddings]+2), f"Windowed stacked tracklets should have prepadding values {stacked_tracklets[det2d.Keys.prepaddings]+2}, but this was {windowed_stacked_tracklets[det2d.Keys.prepaddings]}"
    assert np.all(windowed_stacked_tracklets[det2d.Keys.postpaddings]==stacked_tracklets[det2d.Keys.postpaddings]+1), f"Windowed stacked tracklets should have postpadding values {stacked_tracklets[det2d.Keys.postpaddings]+1}, but this was {windowed_stacked_tracklets[det2d.Keys.postpaddings]}"
    
    windowed_windowed_stacked_tracklets = det2d.stacked_tracklets_window(windowed_stacked_tracklets, window_start=7, window_length=10)
    assert np.all(windowed_windowed_stacked_tracklets[det2d.Keys.prepaddings]==windowed_stacked_tracklets[det2d.Keys.prepaddings]+1), f"Windowed windowed stacked tracklets should have prepadding values {windowed_stacked_tracklets[det2d.Keys.prepaddings]+1}, but this was {windowed_windowed_stacked_tracklets[det2d.Keys.prepaddings]}"
    assert np.all(windowed_windowed_stacked_tracklets[det2d.Keys.postpaddings]==windowed_stacked_tracklets[det2d.Keys.postpaddings]+1), f"Windowed windowed stacked tracklets should have postpadding values {windowed_stacked_tracklets[det2d.Keys.postpaddings]+1}, but this was {windowed_windowed_stacked_tracklets[det2d.Keys.postpaddings]}"
    
def test_stacked_mask():
    categories = det2d.read_categories(categories_path)
    tracklets = det2d.read_tracklets(keypoints_path)
    stacked_tracklets = det2d.stack_tracklets(tracklets[categories.Human], window=True)
    windowed_tracklets = det2d.tracklets_window(tracklets, window_start=8, window_length=8)[categories.Human]
    windowed_stacked_tracklets = det2d.stacked_tracklets_window(stacked_tracklets, window_start=8, window_length=8)
    
    windowed_stacked_tracklets_mask = det2d.stacked_tracklets_confidence_and_unpadded_mask(windowed_stacked_tracklets, confidence_threshold=0.5)
    for tracklet_index, (windowed_tracklet_id, windowed_tracklet) in enumerate(windowed_tracklets.items()):
        assert np.all(windowed_stacked_tracklets_mask[tracklet_index]==det2d.tracklet_confidence_and_unpadded_mask(windowed_tracklet, confidence_threshold=0.5)),\
            f"stacked_tracklets_confidence_and_unpadded_mask returns wrong mask for tracklet id {windowed_tracklet_id}"

def test_detection_loader():
    detection_loader = det2d.DetectionLoader(keypoints_path, window_length=3, window_interval=2)
    assert len(detection_loader)==3, f"detection_loader must represent 3 windows, but this was {len(detection_loader)}"
    
    detection_loader = iter(detection_loader)
    detections = next(detection_loader)
    assert tuple(detections.keys())==(10,11,12), f"detection_loader first window must contain keys (10,11,12), but this was {tuple(detections.keys())}"
    detections = next(detection_loader)
    assert tuple(detections.keys())==(12,13,14), f"detection_loader second window must contain keys (12,13,14), but this was {tuple(detections.keys())}"
    detections = next(detection_loader)
    assert tuple(detections.keys())==(14,), f"detection_loader third window must contain only key (14), but this was {tuple(detections.keys())}"
    detections = detections.copy()
    try:
        next(detection_loader)
    except StopIteration:
        pass
    else:
        raise AssertionError("detection_loader must raise a StopIteration on the fourth call, but didn't")
        
    detection_loader_2 = det2d.DetectionLoader(keypoints_path, window_length=3, window_interval=2, keypoint_indices={0: range(2)})
    detection_loader_2 = iter(detection_loader_2)
    detections_2 = next(detection_loader_2)
    detections_2 = next(detection_loader_2)
    detections_2 = next(detection_loader_2)
    for frame, frame_dict in detections_2.items():
        assert frame in detections_2.keys(), "Yielded frames with keypoint filter don't match full detection frames"
        for category, category_list in frame_dict.items():
            assert category in detections_2[frame].keys(), "Yielded categories with keypoint filter don't match full detection categories"
            for pose_index, pose_dict in enumerate(category_list):
                assert pose_dict[det2d.Keys.id] == detections_2[frame][category][pose_index][det2d.Keys.id], "Yielded pose IDs with keypoint filter don't match full detection IDs"
                assert np.all(pose_dict[det2d.Keys.keypoints][range(2)] == detections_2[frame][category][pose_index][det2d.Keys.keypoints]), "Yielded pose keypoints with keypoint filter don't match sampled full detections"
    
def test_tracklet_loader():
    tracklet_loader = det2d.TrackletLoader(keypoints_path, window_length=2, window_interval=2)
    assert len(tracklet_loader)==3, f"tracklet_loader must represent 3 windows, but this was {len(tracklet_loader)}"
    
    tracklet_loader = iter(tracklet_loader)
    tracklets = next(tracklet_loader)
    assert sorted(tuple(tracklets[0].keys()))==[0,1], f"tracklet_loader first window must contain keys (0,1) in Human category (0), but this was {tuple(tracklets[0].keys())}"
    tracklets = next(tracklet_loader)
    assert sorted(tuple(tracklets[0].keys()))==[0,1,2], f"tracklet_loader second window must contain keys (0,1,2) in Human category (0), but this was {tuple(tracklets[0].keys())}"
    tracklets = next(tracklet_loader)
    assert sorted(tuple(tracklets[0].keys()))==[0,], f"tracklet_loader third window must contain only key (0) in Human category (0), but this was {tuple(tracklets[0].keys())}"
    tracklets = tracklets.copy()
    try:
        next(tracklet_loader)
    except StopIteration:
        pass
    else:
        raise AssertionError("tracklet_loader must raise a StopIteration on the fourth call, but didn't")
        
    tracklet_loader_2 = det2d.TrackletLoader(keypoints_path, window_length=2, window_interval=2, keypoint_indices={0: range(2)})
    tracklet_loader_2 = iter(tracklet_loader_2)
    tracklets_2 = next(tracklet_loader_2)
    tracklets_2 = next(tracklet_loader_2)
    tracklets_2 = next(tracklet_loader_2)
    for category, category_dict in tracklets.items():
        assert category in tracklets_2.keys(), "Yielded categories with keypoint filter don't match full tracklet categories"
        for id, tracklet_dict in category_dict.items():
            assert id in tracklets_2[category].keys(), "Yielded IDs with keypoint filter don't match full tracklet IDs"
            assert tracklets_2[category][id][det2d.Keys.start] == tracklet_dict[det2d.Keys.start], "Yielded start frames with keypoint filter don't match full tracklet start frames"
            assert tracklets_2[category][id][det2d.Keys.prepadding] == tracklet_dict[det2d.Keys.prepadding], "Yielded prepadding with keypoint filter doesn't match full tracklet prepadding"
            assert tracklets_2[category][id][det2d.Keys.postpadding] == tracklet_dict[det2d.Keys.postpadding], "Yielded postpadding with keypoint filter doesn't match full tracklet postpadding"
            assert np.all(tracklets_2[category][id][det2d.Keys.keypoints] == tracklet_dict[det2d.Keys.keypoints][:,range(2)]), "Yielded keypoints with keypoint filter don't match sampled full tracklet keypoints"