"""
EEG Data Processing Module
-------------------------
Handles EEG data loading, preprocessing, and epoching for real-time classification.
Adapted from the original eeg_motor_imagery.py script.
"""

import scipy.io
import numpy as np
import mne
import pandas as pd
from typing import List, Tuple

class EEGDataProcessor:
    """
    Processes EEG data from .mat files for motor imagery classification.
    """
    
    def __init__(self):
        self.fs = None
        self.ch_names = None
        self.event_id = {
            "left_hand": 1,
            "right_hand": 2,
            "neutral": 3,
            "left_leg": 4,
            "tongue": 5,
            "right_leg": 6,
        }
        
    def load_mat_file(self, file_path: str) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
        """Load and parse a single .mat EEG file."""
        mat = scipy.io.loadmat(file_path)
        content = mat['o'][0, 0]

        labels = content[4].flatten()
        signals = content[5]
        chan_names_raw = content[6]
        channels = [ch[0][0] for ch in chan_names_raw]
        fs = int(content[2][0, 0])

        return signals, labels, channels, fs
    
    def create_raw_object(self, signals: np.ndarray, channels: List[str], fs: int, 
                         drop_ground_electrodes: bool = True) -> mne.io.RawArray:
        """Create MNE Raw object from signal data."""
        df = pd.DataFrame(signals, columns=channels)
        
        if drop_ground_electrodes:
            # Drop auxiliary channels that should be excluded
            aux_exclude = ('X3', 'X5')
            columns_to_drop = [ch for ch in channels if ch in aux_exclude]
            
            df = df.drop(columns=columns_to_drop, errors="ignore")
            print(f"Dropped auxiliary channels {columns_to_drop}. Remaining channels: {len(df.columns)}")
        
        eeg = df.values.T
        ch_names = df.columns.tolist()
        
        self.ch_names = ch_names
        self.fs = fs

        info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types="eeg")
        raw = mne.io.RawArray(eeg, info)
        
        return raw
    
    def extract_events(self, labels: np.ndarray) -> np.ndarray:
        """Extract events from label array."""
        onsets = np.where((labels[1:] != 0) & (labels[:-1] == 0))[0] + 1
        event_codes = labels[onsets].astype(int)
        events = np.c_[onsets, np.zeros_like(onsets), event_codes]
        
        # Keep only relevant events
        mask = np.isin(events[:, 2], np.arange(1, 7))
        events = events[mask]
        
        return events
    
    def create_epochs(self, raw: mne.io.RawArray, events: np.ndarray, 
                     tmin: float = 0, tmax: float = 1.5, event_id=None) -> mne.Epochs:
        """Create epochs from raw data and events."""
        if event_id is None:
             event_id = self.event_id
        epochs = mne.Epochs(
            raw,
            events=events,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=None,
            preload=True,
        )
        return epochs
    
    def process_files(self, file_paths: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Process multiple EEG files and return combined data."""
        all_epochs = []
        allowed_labels = {1, 2, 4, 6}
        allowed_event_id = {k: v for k, v in self.event_id.items() if v in allowed_labels}

        for file_path in file_paths:
            signals, labels, channels, fs = self.load_mat_file(file_path)
            raw = self.create_raw_object(signals, channels, fs, drop_ground_electrodes=True)
            events = self.extract_events(labels)
            # only keep allowed labels
            events = events[np.isin(events[:, -1], list(allowed_labels))]
            # create epochs only for allowed labels
            epochs = self.create_epochs(raw, events, event_id=allowed_event_id)
            all_epochs.append((epochs, channels))
        
        if len(all_epochs) > 1:
            epochs_combined = mne.concatenate_epochs([ep for ep, _ in all_epochs])
            ch_names = all_epochs[0][1]  # Assume same channel order for all files
        else:
            epochs_combined = all_epochs[0][0]
            ch_names = all_epochs[0][1]
        # Convert to arrays for model input
        X = epochs_combined.get_data().astype("float32")
        y = (epochs_combined.events[:, -1] - 1).astype("int64")  # classes 0..5
        return X, y, ch_names
    
    def load_continuous_data(self, file_paths: List[str]) -> Tuple[np.ndarray, int]:
        """
        Load continuous raw EEG data without epoching.
        
        Args:
            file_paths: List of .mat file paths
            
        Returns:
            raw_data: Continuous EEG data [n_channels, n_timepoints]
            fs: Sampling frequency
        """
        all_raw_data = []
        
        for file_path in file_paths:
            signals, labels, channels, fs = self.load_mat_file(file_path)
            raw = self.create_raw_object(signals, channels, fs, drop_ground_electrodes=True)
            
            # Extract continuous data (no epoching)
            continuous_data = raw.get_data()  # [n_channels, n_timepoints]
            all_raw_data.append(continuous_data)
        
        # Concatenate all continuous data along time axis
        if len(all_raw_data) > 1:
            combined_raw = np.concatenate(all_raw_data, axis=1)
        else:
            combined_raw = all_raw_data[0]
            
        return combined_raw, fs
    
    def prepare_loso_split(self, file_paths: List[str], test_session_idx: int = 0) -> Tuple:
        """
        Prepare Leave-One-Session-Out (LOSO) split for EEG data.
        
        Args:
            file_paths: List of .mat file paths (one per subject)
            test_subject_idx: Index of subject to use for testing
            
        Returns:
            X_train, y_train, X_test, y_test, subject_info
        """
        all_sessions_data = []
        session_info = []
        
        # Load each subject separately
        for i, file_path in enumerate(file_paths):
            signals, labels, channels, fs = self.load_mat_file(file_path)
            raw = self.create_raw_object(signals, channels, fs, drop_ground_electrodes=True)
            events = self.extract_events(labels)
            epochs = self.create_epochs(raw, events)
            
            # Convert to arrays
            X_subject = epochs.get_data().astype("float32")
            y_subject = (epochs.events[:, -1] - 1).astype("int64")
            all_sessions_data.append((X_subject, y_subject))
            session_info.append({
                'file_path': file_path,
                'subject_id': f"Subject_{i+1}",
                'n_epochs': len(X_subject),
                'channels': channels,
                'fs': fs
            })
        
        # LOSO split: one session for test, others for train
        test_sessions = all_sessions_data[test_session_idx]
        train_sessions = [all_sessions_data[i] for i in range(len(all_sessions_data)) if i != test_session_idx]

        # Combine training sessions
        if len(train_sessions) > 1:
            X_train = np.concatenate([sess[0] for sess in train_sessions], axis=0)
            y_train = np.concatenate([sess[1] for sess in train_sessions], axis=0)
        else:
            X_train, y_train = train_sessions[0]

        X_test, y_test = test_sessions

        print("LOSO Split:")
        print(f"  Test Subject: {session_info[test_session_idx]['subject_id']} ({len(X_test)} epochs)")
        print(f"  Train Subjects: {len(train_sessions)} subjects ({len(X_train)} epochs)")

        return X_train, y_train, X_test, y_test, session_info

    def simulate_real_time_data(self, X: np.ndarray, y: np.ndarray, mode: str = "random") -> Tuple[np.ndarray, int]:
        """
        Simulate real-time EEG data for demo purposes.
        
        Args:
            X: EEG data array (currently epoched data)
            y: Labels array  
            mode: "random", "sequential", or "class_balanced"
            
        Returns:
            Single epoch and its true label
        """
        if mode == "random":
            idx = np.random.randint(0, len(X))
        elif mode == "sequential":
            # Use a counter for sequential sampling (would need to store state)
            idx = np.random.randint(0, len(X))  # Simplified for now
        elif mode == "class_balanced":
            # Sample ensuring we get different classes
            available_classes = np.unique(y)
            target_class = np.random.choice(available_classes)
            class_indices = np.where(y == target_class)[0]
            idx = np.random.choice(class_indices)
        else:
            idx = np.random.randint(0, len(X))
            
        return X[idx], y[idx]
    

