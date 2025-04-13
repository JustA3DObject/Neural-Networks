import pandas as pd  
import numpy as np   
import matplotlib.pyplot as plt 
import mne  # EEG processing toolkit
from mne.preprocessing import ICA  # Artifact removal
import pywt  # Wavelet transforms
from scipy.signal import hilbert  # Signal envelope detection

class EEGSpindleAnalyzer:

    def __init__(self, file_path, sheet_name, sfreq=512):

        self.file_path = file_path
        self.sheet_name = sheet_name
        self.sfreq = sfreq  # Store sampling frequency
        self.raw = None  # MNE Raw object container
        self.ica = None  # Independent Component Analysis results
        self.filtered_data = None  # Alpha-filtered EEG data
        self.spindle_counts = {}  # Dictionary: {electrode: spindle_count}
        self.spindle_events = {}  # Dictionary: {electrode: [(start, end)]}

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

    def detect_all_spindles(self, threshold=1.5, min_duration=0.5):
        for channel in self.raw.ch_names:
            spindles = self.detect_spindles_electrode(
                channel, 
                threshold=threshold,
                min_duration=min_duration
            )
            self.spindle_counts[channel] = len(spindles)
            self.spindle_events[channel] = spindles

    def get_total_spindles(self):
        return sum(self.spindle_counts.values())

    def get_electrode_counts(self):
        return self.spindle_counts.copy()
    
    def compute_scaleogram(self, electrode=None):
        # Compute wavelet scaleogram using Continuous Wavelet Transform (CWT)
       
        # Select data source
        if electrode:
            idx = self.raw.ch_names.index(electrode)
            data = self.filtered_data[idx, :]
        else:
            data = self.filtered_data.mean(axis=0)  # Global average
            
        # Continuous Wavelet Transform parameters
        scales = np.arange(1, 128)  # Range of wavelet scales
        coefficients, _ = pywt.cwt(
            data, 
            scales, 
            wavelet='morl',  # Morlet wavelet
            sampling_period=1/self.sfreq  # Time between samples
        )
        return coefficients

    def visualize_scaleogram(self, coefficients, electrode=None):
        # Visualize scaleogram with spindle overlays
  
        # Create time axis
        time_points = coefficients.shape[1]
        time_axis = np.arange(time_points) / self.sfreq
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Plot wavelet magnitude
        plt.imshow(
            np.abs(coefficients),  # Magnitude of complex coefficients
            extent=[time_axis[0], time_axis[-1], 1, 128],  # Axis ranges
            cmap='PRGn',  # Purple-Green colormap
            aspect='auto',  # Automatic aspect ratio
            interpolation='nearest'  # Pixel-based interpolation
        )
        
        # Add spindle overlays if specified
        if electrode:
            spindles = self.spindle_events.get(electrode, [])
            for start, end in spindles:
                plt.axvspan(
                    start, end,
                    color='red',
                    alpha=0.3,  # Semi-transparent
                    label='Spindle' if start == spindles[0][0] else ''
                )
                
        # Add plot decorations
        plt.colorbar(label='Coefficient Magnitude')
        plt.ylabel('Wavelet Scale')
        plt.xlabel('Time (seconds)')
        title = f'Scaleogram - {electrode}' if electrode else 'Global Scaleogram'
        plt.title(title)
        plt.show()

    def epoch_analysis(self, epoch_duration=10):
 
        # Create epochs using MNE's built-in function
        epochs = mne.make_fixed_length_epochs(
            self.raw,
            duration=epoch_duration,
            preload=True  # Load data into memory
        )
        
        # Visualize each epoch
        for epoch_num in range(len(epochs)):
            # Get epoch data (shape: 1 × channels × time)
            epoch_data = epochs[epoch_num].get_data()[0]
            
            # Create plot
            plt.figure(figsize=(12, 6))
            plt.plot(epochs.times, epoch_data.T)  # Plot all channels
            plt.title(f'Epoch {epoch_num+1} ({epoch_duration}s)')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Amplitude (μV)')
            plt.show()

# Problem 1: Total Spindle Count Comparison
print("\nProblem 1: Total Spindle Count Comparison")
ecbl = EEGSpindleAnalyzer('Original_Data.xlsx', 'ECBL')
ecbl.detect_all_spindles()
eobl = EEGSpindleAnalyzer('Original_Data.xlsx', 'EOBL')
eobl.detect_all_spindles()

print(f"ECBL Total Spindles: {ecbl.get_total_spindles()}")
print(f"EOBL Total Spindles: {eobl.get_total_spindles()}")
print(f"Absolute Difference: {abs(ecbl.get_total_spindles() - eobl.get_total_spindles())}")

# Problem 2: Electrode-wise Comparison
print("\nProblem 2: Electrode-wise Comparison")
ecbl_counts = ecbl.get_electrode_counts()
eobl_counts = eobl.get_electrode_counts()

# Print formatted comparison table
print(f"{'Electrode':<8} | {'ECBL':<5} | {'EOBL':<5} | Difference")
print("-" * 35)
for electrode in ecbl_counts:
    diff = abs(ecbl_counts[electrode] - eobl_counts[electrode])
    print(f"{electrode:<8} | {ecbl_counts[electrode]:<5} | {eobl_counts[electrode]:<5} | {diff}")