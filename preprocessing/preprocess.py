import os
import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy import signal

def preprocess_light_curve(fits_file, window_size=1001, polyorder=2):
    # Open the FITS file
    with fits.open(fits_file) as hdul:
        # Assume the data is in the first extension, adjust if needed
        data = Table(hdul[1].data)

    time = data['TIME']
    flux = data['PDCSAP_FLUX']  # Using PDCSAP flux, change if needed

    # Remove NaN values
    mask = np.isfinite(time) & np.isfinite(flux)
    time = time[mask]
    flux = flux[mask]

    # Normalize the flux
    flux_norm = flux / np.median(flux)

    # Detrend the light curve using Savitzky-Golay filter
    flux_detrended = signal.savgol_filter(flux_norm, window_size, polyorder)

    # Calculate residuals (this will highlight transit signals)
    residuals = flux_norm - flux_detrended

    # Interpolate to a regular time grid (TCNs typically require uniform sampling)
    time_regular = np.linspace(time.min(), time.max(), len(time))
    flux_interpolated = np.interp(time_regular, time, residuals)

    return time_regular, flux_interpolated

def prepare_tcn_input(time, flux, segment_length=1000, stride=500):
    # Segment the light curve into overlapping windows
    segments = []
    for i in range(0, len(flux) - segment_length + 1, stride):
        segment = flux[i:i + segment_length]
        segments.append(segment)

    # Convert to numpy array and reshape for TCN input
    segments = np.array(segments)
    segments = segments.reshape(segments.shape[0], segments.shape[1], 1)

    return segments

def process_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.fits'):
                fits_file = os.path.join(root, file)
                print(f"Processing: {fits_file}")
                try:
                    time, flux = preprocess_light_curve(fits_file)
                    tcn_input = prepare_tcn_input(time, flux)
                    print(f"TCN input shape: {tcn_input.shape}")
                    print(f"Number of segments: {tcn_input.shape[0]}")
                    print(f"Segment length: {tcn_input.shape[1]}")
                    print("---")
                except Exception as e:
                    print(f"Error processing {fits_file}: {str(e)}")
                    print("---")

# Directories to process
directories = [
    'data/kepler',
    'data/k2',
    'data/tess/mastdownload/TESS'
]

# Process each directory
for directory in directories:
    print(f"Processing directory: {directory}")
    process_directory(directory)