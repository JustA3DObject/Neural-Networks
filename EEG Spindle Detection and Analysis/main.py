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