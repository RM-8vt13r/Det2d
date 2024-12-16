Module that is used to read pose detections in .det2d.json format in various other MPE repositories  
  
src/det2d:
- read.py: read 2D pose detections or tracklets from .det2d.json files
- convert.py: convert from pose detection to pose tracklet format, and vice-versa
- fill.py: fill any missing keypoint detections in a tracklet using zeros or bilinear spline interpolation
- verify.py: verify that a tracklet is complete and usable for future processing
- window.py: apply time windowing to a tracklet using sampling and zero-padding
- categories.py: categories to index pose dictionaries
- keys.py: keys to index pose dictionaries
- mask.py: create tracklet masks for various purposes
- stack.py: stack keypoints of multiple tracklets for vectorized calculation

testing:
- test_read.py: tests to verify the correctness of everything in src/det2d
- test_keypoints.det2d.json: dummy poses used in test_read.py

Installation:
`pip install git+https://github.com/RM-8vt13r/Det2d.git`

Usage:
```python
from det2d import read_tracklets, read_detections, detections2tracklets, tracklets2detections, tracklet_confidence_mask, \
                    tracklet_window, tracklet_unpadded_mask, \
                    read_categories, Keys, \
                    DetectionLoader, TrackletLoader

# Reading pose files
tracklets  = read_tracklets(det2d_path)
detections = read_detections(det2d_path)
usable_tracklet_mask = tracklet_confidence_mask(tracklets, confidence_threshold=0.5)

# Converting between tracklet and detection format
tracklets  = detections2tracklets(detections)
detections = tracklets2detections(tracklets)

# Loading category details
categories = read_categories(categories_path)
keypoints  = read_category_keypoints(categories_path, categories.Human)

# Indexing tracklets
human_tracklets = tracklets[categories.Human]
human_tracklet  = human_tracklets[1] # Get the human tracklet with ID '1'

# Indexing keypoints
human_tracklet_keypoints = human_tracklet[Keys.keypoints]
human_tracklet_nose      = human_tracklet_keypoints[:,keypoints.nose,:]

# Windowing tracklets
windowed_tracklet = tracklet_window(human_tracklet, window_start=10, window_length=100)
original_tracklet_mask = tracklet_unpadded_mask(windowed_tracklet)

# Reading pose detections line by line
detection_loader = DetectionLoader(det2d_path)
n_detections = len(detection_loader)
for next_window in detection_loader:
    print(next_window)

tracklet_loader = TrackletLoader(det2d_path)
n_tracklets = len(tracklet_loader)
for next_window in tracklet_loader:
    print(next_window)
```

Remaining functions:
```python
stack_tracklets                       # Stack multiple tracklets for efficient vectorized calculation
interpolate_tracklets_gaps            # Interpolate gaps in multiple tracklets
interpolate_tracklet_gaps             # Interpolate gaps in a single tracklet
zero_tracklets_gaps                   # Fill gaps with zeroes in multiple tracklets
zero_tracklet_gaps                    # Fill gaps with zeroes in a single tracklet
tracklet_window_overlap               # Retrieve information relevant to windowing
tracklets_window                      # Apply a window to multiple tracklets
stacked_tracklets_window              # Apply a window to stacked tracklets
assert_tracklet_valid                 # Assert that a tracklet is in the correct format
assert_tracklets_valid                # Assert that multiple tracklets are in the correct format
assert_stacked_tracklets_valid        # Assert that stacked tracklets are in the correct format
assert_tracklets_comparable           # Assert if two tracklets cover the same frame range
read_categories                       # Obtain a namespace of all categories from a .json file
read_category_keypoints               # Read the keypoint definitions of a single category
read_category_details                 # Get all information on a single category
tracklet_confidence_mask              # Mask True where a tracklet satisfies a confidence threshold and False elsewhere
tracklet_unpadded_mask                # Mask False where a tracklet was padded and True elsewhere
tracklet_confidence_and_unpadded_mask # Combines tracklet_confidence_mask and tracklet_unpadded_mask
stacked_tracklets_confidence_mask     # Apply tracklet_confidence_mask to stacked tracklets
stacked_tracklets_unpadded_mask       # Apply tracklet_unpadded_mask to stacked tracklets
stacked_tracklets_confidence_and_unpadded_mask # Apply tracklet_confidence_and_unpadded_mask to stacked tracklets
```

Testing:
`pytest testing/`
