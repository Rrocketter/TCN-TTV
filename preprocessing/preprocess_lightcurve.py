import os
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from astropy.stats import sigma_clip
from scipy import interpolate, signal
from lightkurve import search_lightcurve


def load_data(file_path):
    """Load light curve data from K2, Kepler, or TESS."""
    if 'tess' in file_path.lower():
        lc = search_lightcurve(file_path).download()
        time, flux = lc.time.value, lc.flux.value
    else:  # K2 or Kepler
        with fits.open(file_path) as hdul:
            time = hdul[1].data['TIME']
            flux = hdul[1].data['PDCSAP_FLUX']
    return time, flux


def remove_outliers(flux, sigma=5, maxiters=5):
    """Remove outliers using sigma clipping."""
    return sigma_clip(flux, sigma=sigma, maxiters=maxiters).filled(np.nan)


def handle_missing_data(time, flux):
    """Interpolate missing data."""
    mask = np.isfinite(flux)
    f = interpolate.interp1d(time[mask], flux[mask], bounds_error=False, fill_value="extrapolate")
    return f(time)


def detrend_lightcurve(time, flux, window_length=101):
    """Detrend the light curve using a median filter."""
    trend = signal.medfilt(flux, kernel_size=window_length)
    return flux / trend


def extract_transit_windows(time, flux, transit_times, duration):
    """Extract windows around transit events."""
    windows = []
    for t in transit_times:
        mask = (time >= t - duration / 2) & (time <= t + duration / 2)
        windows.append((time[mask], flux[mask]))
    return windows


def normalize_flux(flux):
    """Normalize flux values."""
    return (flux - np.median(flux)) / np.median(flux)


def normalize_time(time, t0):
    """Normalize time values."""
    t = Time(time, format='jd', scale='tdb')
    return (t.jd - t0) / (24 * 3600)  # Convert to days


def preprocess_lightcurve(file_path, transit_times, transit_duration):
    """Complete preprocessing pipeline for a single light curve."""
    # Load data
    time, flux = load_data(file_path)

    # Remove outliers
    flux_cleaned = remove_outliers(flux)

    # Handle missing data
    flux_filled = handle_missing_data(time, flux_cleaned)

    # Detrend light curve
    flux_detrended = detrend_lightcurve(time, flux_filled)

    # Extract transit windows
    transit_windows = extract_transit_windows(time, flux_detrended, transit_times, transit_duration)

    # Normalize each transit window
    normalized_windows = []
    for t, f in transit_windows:
        t_norm = normalize_time(t, t[0])
        f_norm = normalize_flux(f)
        normalized_windows.append((t_norm, f_norm))

    return normalized_windows


def process_all_lightcurves(data_dir, output_dir, transit_info):
    """Process all light curves in the given directory."""
    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.fits'):
                file_path = os.path.join(root, file)
                star_id = os.path.splitext(file)[0]

                if star_id in transit_info:
                    transit_times = transit_info[star_id]['transit_times']
                    transit_duration = transit_info[star_id]['transit_duration']

                    try:
                        normalized_windows = preprocess_lightcurve(file_path, transit_times, transit_duration)

                        # Save processed data
                        output_file = os.path.join(output_dir, f"{star_id}_processed.npz")
                        np.savez(output_file, *normalized_windows)

                        print(f"Processed {file_path}")
                    except Exception as e:
                        print(f"Error processing {file_path}: {str(e)}")
                else:
                    print(f"No transit info for {star_id}, skipping")


if __name__ == "__main__":
    # Example usage
    data_dir = "path/to/your/data"
    output_dir = "path/to/output"

    # You should replace this with actual transit information
    transit_info = {
        "star_id_1": {"transit_times": [2458000.1, 2458010.2], "transit_duration": 0.5},
        "star_id_2": {"transit_times": [2458005.3, 2458015.4], "transit_duration": 0.6},
        # Add more stars and their transit information
    }

    process_all_lightcurves(data_dir, output_dir, transit_info)