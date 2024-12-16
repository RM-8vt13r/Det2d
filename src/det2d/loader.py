from abc import ABC, abstractmethod
from typing import override
import os
import datetime
import json
from types import FunctionType
import numpy as np

from .read import _process_frame_detections_dict
from .keys import Keys
from .fill import zero_tracklet_gaps

class Loader(ABC):
    def __init__(self, path: str, window_length: int=1, window_interval: int=1, keypoint_indices: dict=None):
        """
        Create a (Detection- or Tracklet)Loader from the specified path, which loads a new window from a file on every call to __next__()
        
        Inputs:
        - path: the path to the det2d file (.det2d.json)
        - window_length: the number of subsequent poses to return on each iteration
        - window_interval: the distance in frames between two window start frames
        - keypoint_indices: dict of the keypoint indices to load per category; leave None to load all keypoints
        """
        assert os.path.isfile(path), f"Given path \"{path}\" does not point to a .det2d.json file"
        assert isinstance(window_length, int), f"window_length type must be int, but was {type(window_length)}"
        assert window_length > 0, f"window_length must be >0, but was {window_length}"
        assert isinstance(window_interval, int), f"window_interval type must be int, but was {type(window_interval)}"
        assert window_interval > 0, f"window_interval must be >0, but was {window_interval}"
        assert keypoint_indices is None or isinstance(keypoint_indices, dict), f"keypoint_indices must be None or a dict, but was a {type(keypoint_indices)}"
        self._path = path
        self._window_length = window_length
        self._window_interval = window_interval
        self._keypoint_indices = keypoint_indices
        self._init_len()
    
    def __iter__(self):
        """
        Make the Loader iterable
        """
        self._file_handle = open(self.path, 'r')
        first_line = self.file_handle.readline().strip()
        assert first_line=='{', f".det2d.json file must start with '{{', but the first line was '{first_line}'"
        
        self._current_window = {}
        self._current_window_start_frame = 0
        self._current_window_stop_frame = 0
        self._termination_frame = None
        self._det2d_start_frame = None
        return self
    
    def __next__(self):
        """
        Return the next window of detections
        
        Output:
        - The next det2d window (detections)
        - The window start timestamp in seconds
        """
        # Remove old frames
        self._del_first_window_frames(max(0, self.frames_in_current_window-self.window_length+self.window_interval)) # Delete first frames
        self._current_window_start_frame += max(0, self.frames_in_current_window-self.window_length+self.window_interval) # Update _window_start_frame, regardless of how much the window actually changes
        
        # Stop iteration if file has ended
        if self.termination_frame is not None and self.current_window_start_frame >= self.termination_frame:
            self.file_handle.close()
            raise StopIteration
            
        # Add new frames
        while self.frames_in_current_window < self.window_length:
            line = self.file_handle.readline().strip()
            self._current_window_stop_frame += 1 # Update _current_window_stop_frame, regardless of whether the window actually changes
            
            if line in ('', '}'): # End of file reached
                if self.termination_frame is None: self._termination_frame = self.current_window_stop_frame
                if len(self.current_window) == 0 and len(line) == 0:
                    self.file_handle.close()
                    raise StopIteration
                continue
            
            frame, detection_dicts = line.split(':', 1)
            frame = int(frame.strip()[1:-1])
            detection_dicts = json.loads(detection_dicts.strip().rstrip(','), object_hook=lambda dictionary: {int(key) if key.isdigit() else key: np.array(value) if isinstance(value, list | tuple) else value for key, value in dictionary.items()})
            _process_frame_detections_dict(frame, detection_dicts)
            if self.det2d_start_frame is None: self._det2d_start_frame = frame
            self._update_current_window(frame, detection_dicts) # Add current line to the window dict
        
        return self.current_window
    
    @abstractmethod
    def _del_first_window_frames(self, n_frames: int=1):
        """
        Delete a number of frames from the start of the current window
        
        Input:
        - n_frames: the number of frames to delete
        """
        pass
    
    @abstractmethod
    def _update_current_window(self, frame, detection_dicts):
        """
        Update the current window with a new line read from a file
        
        Input:
        - frame: the frame number of the new frame
        - detection_dicts: the detections on the new frame, already processed with _process_frame_detections_dict
        """
        pass
    
    def __len__(self):
        """
        Get the number of windows that can be extracted from this DetectionLoader
        """
        return self._n_windows
    
    def _init_len(self):
        """
        Retrieve the number of frames from the file.
        """
        self._file_handle = open(self.path, 'r')
        lines = self.file_handle.readlines()
        self.file_handle.close()
        
        while len(lines) and lines[0].strip()  in ('', '{'): lines = lines[1:]
        while len(lines) and lines[-1].strip() in ('', '}'): lines = lines[:-1]
        
        n_frames = len(lines)
        n_windows = int(np.ceil(n_frames/self.window_interval))
        
        self._n_frames = n_frames
        self._n_windows = n_windows
        
    @property
    def path(self):
        return self._path
    
    @property
    def window_length(self):
        return self._window_length
    
    @property
    def window_interval(self):
        return self._window_interval
    
    @property
    def current_window(self):
        return self._current_window
    
    @property
    def current_window_start_frame(self):
        return self._current_window_start_frame
        
    @property
    def current_window_stop_frame(self):
        return self._current_window_stop_frame
    
    @property
    def frames_in_current_window(self):
        return self.current_window_stop_frame-self.current_window_start_frame
    
    @property
    def termination_frame(self):
        return self._termination_frame
        
    @property
    def n_frames(self):
        return self._n_frames
        
    @property
    def n_windows(self):
        return self._n_windows
    
    @property
    def keypoint_indices(self):
        return self._keypoint_indices
        
    @property
    def file_handle(self):
        return self._file_handle
        
    @property
    def det2d_start_frame(self):
        return self._det2d_start_frame


class DetectionLoader(Loader):
    """
    Load detections from a file one by one
    """
    @override
    def _del_first_window_frames(self, n_frames: int=1):
        """
        See Loader
        """
        current_window_keys = list(self.current_window.keys())
        for _ in range(n_frames):
            if len(current_window_keys) == 0: break
            del self._current_window[current_window_keys.pop(0)]
    
    @override
    def _update_current_window(self, frame: int, detection_dicts: dict):
        """
        See Loader
        """
        self.current_window[frame] = {
            category: [
                pose_dict | {Keys.keypoints: pose_dict[Keys.keypoints][self.keypoint_indices[category] if self.keypoint_indices is not None else slice(None)]}
                for pose_dict in category_list
            ]
            for category, category_list in detection_dicts.items()
        }
        
        
class TrackletLoader(Loader):
    def __init__(self, path: str, window_length: int=1, window_interval: int=1, keypoint_indices: dict=None, fill_function: FunctionType=zero_tracklet_gaps, confidence_threshold: float=0):
        """
        Load tracklets from a file one by one
        
        Inputs:
        - fill_function: function from fill.py with which to fill tracklet gaps
        - confidence_threshold: keypoint confidence threshold below which to assume a keypoint to be undetected
        """
        super().__init__(path, window_length, window_interval, keypoint_indices)
        self._fill_function = fill_function
        self._confidence_threshold = confidence_threshold
        
    @override
    def _del_first_window_frames(self, n_frames: int=1):
        """
        See Loader
        """
        if self.det2d_start_frame is None: return
        tracklet_new_start_frame = self.det2d_start_frame + self.current_window_start_frame + n_frames
        categories = tuple(self.current_window.keys())
        for category in categories:
            category_dict = self.current_window[category]
            ids = tuple(category_dict.keys())
            for id in ids:
                tracklet_dict = category_dict[id]
                n_frames_to_delete = tracklet_new_start_frame - tracklet_dict[Keys.start]
                if n_frames_to_delete <= 0: continue
                
                if n_frames_to_delete >= tracklet_dict[Keys.keypoints].shape[0]:
                    del category_dict[id]
                    continue
                
                tracklet_dict[Keys.start] = tracklet_new_start
                tracklet_dict[Keys.keypoints] = tracklet_dict[Keys.keypoints][n_frames_to_delete:]
                tracklet_dict[Keys.prepadding] = max(0, tracklet_dict[Keys.prepadding]-n_frames_to_delete)
            
            if len(category_dict) == 0: del self.current_window[category]
    
    @override
    def _update_current_window(self, frame: int, detection_dicts: dict):
        """
        See Loader
        """
        for category, pose_list in detection_dicts.items():
            if category not in self.current_window.keys(): self.current_window[category] = {}
            category_dict = self.current_window[category]
            for pose_dict in pose_list:
                if pose_dict[Keys.id] not in category_dict.keys():
                    category_dict[pose_dict[Keys.id]] = {
                        Keys.start: frame,
                        Keys.keypoints: pose_dict[Keys.keypoints][None, self.keypoint_indices[category] if self.keypoint_indices is not None else slice(None)],
                        Keys.prepadding: 0,
                        Keys.postpadding: 0
                    }
                    continue
                
                tracklet_dict = category_dict[pose_dict[Keys.id]]
                tracklet_dict[Keys.keypoints] = np.append(tracklet_dict[Keys.keypoints], pose_dict[Keys.keypoints][None, self.keypoint_indices[category] if self.keypoint_indices is not None else slice(None)], axis=0)
                
                if frame == tracklet_dict[Keys.start] + len(tracklet_dict[Keys.keypoints]) - 1: continue
                
                tracklet_dict[Keys.frames] = np.append(np.arange(len(tracklet_dict[Keys.keypoints])-1)+tracklet_dict[Keys.start], frame)
                # import pdb
                # pdb.set_trace()
                category_dict[pose_dict[Keys.id]] = self._fill_function(tracklet_dict, self._confidence_threshold)