import pandas as pd  
import numpy as np   
import matplotlib.pyplot as plt 
import mne  # EEG processing toolkit
from mne.preprocessing import ICA  # Artifact removal
import pywt  # Wavelet transforms

class EEGSpindleAnalyzer:

    def __init__(self, file_path, sheet_name, sfreq=512):

        self.file_path = file_path
        self.sheet_name = sheet_name
        self.sfreq = sfreq  # Store sampling frequency
        self.raw = None  # MNE Raw object container
        self.ica = None  # Independent Component Analysis results
        self.filtered_data = None  # Alpha-filtered EEG data

        # Execute processing pipeline
        self.load_data()      # Load data
        self.apply_ica()      # Apply ICA
        self.filter_alpha()   # Bandpass filter

    def load_data(self):
        df = pd.read_excel(self.file_path, sheet_name=self.sheet_name)
        
        # Remove time column if present (non-EEG channel)
        if 'time' in df.columns:
            df = df.drop(columns=['time'])
            
        # Convert data to MNE-compatible format
        # Transpose because MNE expects channels × time
        data = df.values.T * 1e-6  # Convert µV to volts
        ch_names = df.columns.tolist()  # Extract channel names
        
        # Create MNE Info object with EEG metadata
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=self.sfreq,
            ch_types='eeg'  # All channels are EEG type
        )
        
        # Create RawArray container for continuous EEG data
        self.raw = mne.io.RawArray(data, info)

    def apply_ica(self):
        # Initialize ICA with 20 components (typical for EEG)
        ica = ICA(
            n_components=20,
            method='infomax',  # Using Infomax instead of FastICA
            random_state=97  # Seed for reproducibility
        )
        
        # Fit ICA to the raw EEG data
        ica.fit(self.raw)
        self.ica = ica  # Store ICA model for later inspection

    def filter_alpha(self):
        # IIR filter is computationally efficient for this range
        self.raw.filter(
            l_freq=8,   # Lower cutoff frequency
            h_freq=13,  # Upper cutoff frequency
            method='iir' # Infinite Impulse Response filter
        )
        self.filtered_data = self.raw.get_data()

    def detect_spindles_electrode(self, electrode, threshold=1.5, min_duration=0.5):

        # Get electrode index and filtered signal
        idx = self.raw.ch_names.index(electrode)
        signal = self.filtered_data[idx, :]
        
        # Compute analytic signal using Hilbert transform
        analytic_signal = hilbert(signal)
        amplitude_envelope = np.abs(analytic_signal)
        
        # Threshold detection logic
        thresholded = amplitude_envelope > threshold
        state_changes = np.diff(thresholded.astype(int))
        
        # Find spindle start and end indices
        starts = np.where(state_changes == 1)[0]  # Rising edges
        ends = np.where(state_changes == -1)[0]   # Falling edges

        # Handle edge cases
        # Case 1: Signal starts above threshold
        if len(ends) == 0 and len(starts) > 0:
            ends = np.array([len(thresholded) - 1])
            
        # Case 2: No spindles detected
        if len(starts) == 0:
            return []
            
        # Case 3: First end before first start
        if ends[0] < starts[0]:
            starts = np.insert(starts, 0, 0)
            
        # Case 4: Last start without matching end
        if starts[-1] > ends[-1]:
            ends = np.append(ends, len(thresholded)-1)

        # Convert indices to time durations
        spindles = []
        for start_idx, end_idx in zip(starts, ends):
            duration = (end_idx - start_idx) / self.sfreq
            if duration >= min_duration:
                start_time = start_idx / self.sfreq
                end_time = end_idx / self.sfreq
                spindles.append((start_time, end_time))
                
        return spindles