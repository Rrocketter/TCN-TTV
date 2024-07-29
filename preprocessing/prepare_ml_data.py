import numpy as np
import os
import logging
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.stats import sigmaclip
import warnings

# Set up logging
logging.basicConfig(filename='ttv_detection.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def load_processed_data(file_path):
    """Load processed data from .npz file."""
    with np.load(file_path) as data:
        return data['time'], data['flux']


def pad_or_truncate(array, target_len):
    """Pad or truncate array to target length."""
    if len(array) > target_len:
        return array[:target_len]
    else:
        return np.pad(array, (0, target_len - len(array)), mode='constant')


def detect_transit_time(time, flux):
    """Detect the time of transit minimum."""
    if len(flux) == 0:
        return None
    # Use a window to find the local minimum
    window = max(min(len(flux) // 10, 100),
                 1)  # Use 10% of the data or 100 points, whichever is smaller, but at least 1
    smoothed_flux = np.convolve(flux, np.ones(window) / window, mode='same')
    return time[np.argmin(smoothed_flux)]


def linear_ephemeris(epoch, t0, period):
    """Linear ephemeris model."""
    return t0 + epoch * period


def fit_linear_ephemeris(transit_times):
    """Fit a linear ephemeris to the observed transit times."""
    epochs = np.arange(len(transit_times))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, pcov = curve_fit(linear_ephemeris, epochs, transit_times)
        perr = np.sqrt(np.diag(pcov))
        if np.any(np.isnan(perr)) or np.any(np.isinf(perr)):
            raise ValueError("Covariance estimation failed")
        return popt, True
    except (RuntimeError, ValueError) as e:
        logging.warning(f"Curve fitting failed: {str(e)}. Using simple linear regression.")
        # Fallback to simple linear regression
        A = np.vstack([epochs, np.ones(len(epochs))]).T
        period, t0 = np.linalg.lstsq(A, transit_times, rcond=None)[0]
        return [t0, period], False


def calculate_ttv(observed_time, expected_time, period):
    """Calculate TTV in units of minutes."""
    return (observed_time - expected_time) * 24 * 60  # Convert days to minutes


def prepare_data_for_ttv_detection(processed_data_dir, window_size=1000, ttv_threshold=30):
    """Prepare data for TTV detection model."""
    all_transits = []
    all_labels = []
    all_ttvs = []

    # Loop through each mission folder
    for mission in ['kepler', 'k2', 'tess']:
        mission_dir = os.path.join(processed_data_dir, mission)
        if not os.path.exists(mission_dir):
            logging.warning(f"Directory not found: {mission_dir}")
            continue

        logging.info(f"Processing {mission} data...")

        # Load and format all processed data for the current mission
        for file in os.listdir(mission_dir):
            if file.endswith('.npz'):
                file_path = os.path.join(mission_dir, file)
                logging.info(f"Processing file: {file_path}")

                time, flux = load_processed_data(file_path)
                logging.info(f"Loaded data from {file_path}. Time shape: {time.shape}, Flux shape: {flux.shape}")

                # Detect transit times
                observed_transit_times = [detect_transit_time(time[i:i + window_size], flux[i:i + window_size])
                                          for i in range(0, len(flux), window_size)]
                observed_transit_times = [t for t in observed_transit_times if t is not None]
                logging.info(f"Detected {len(observed_transit_times)} transits in {file_path}")

                # Fit linear ephemeris
                if len(observed_transit_times) < 2:
                    logging.warning(f"Not enough transits in {file_path}. Skipping...")
                    continue  # Skip if not enough transits

                [t0, period], fit_successful = fit_linear_ephemeris(observed_transit_times)
                logging.info(f"Fitted linear ephemeris for {file_path}. t0: {t0}, period: {period}")

                ttvs = []
                for i, transit_time in enumerate(observed_transit_times):
                    start_idx = i * window_size
                    end_idx = start_idx + window_size
                    transit_flux = pad_or_truncate(flux[start_idx:end_idx], window_size)
                    all_transits.append(transit_flux)

                    # Calculate expected transit time
                    expected_transit_time = linear_ephemeris(i, t0, period)

                    # Calculate TTV
                    ttv = calculate_ttv(transit_time, expected_transit_time, period)
                    ttvs.append(ttv)

                # Use sigma clipping to remove outliers
                ttvs_clipped, low, high = sigmaclip(ttvs, low=3, high=3)
                for i, ttv in enumerate(ttvs):
                    if low <= ttv <= high:
                        all_ttvs.append(ttv)
                        # Label based on whether TTV exceeds threshold
                        label = 1 if abs(ttv) > ttv_threshold else 0
                        all_labels.append(label)
                        logging.info(f"Transit {i} in {file_path}: TTV = {ttv:.2f} minutes, Label = {label}")
                    else:
                        logging.warning(
                            f"Outlier TTV detected in {file_path}: {ttv:.2f} minutes. Skipping this transit.")

    # Convert to numpy arrays
    X = np.array(all_transits)
    y = np.array(all_labels)
    ttvs = np.array(all_ttvs)

    logging.info(f"Total transits processed: {len(X)}")
    logging.info(f"Positive labels (TTVs): {np.sum(y)}")

    return X, y, ttvs


if __name__ == "__main__":
    processed_data_dir = "../processed_data/"
    logging.info("Starting TTV detection data preparation")

    X, y, ttvs = prepare_data_for_ttv_detection(processed_data_dir)

    # Save the processed data
    output_dir = "../ml_data/"
    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(os.path.join(output_dir, "ttv_detection_data.npz"),
                        X=X, y=y, ttvs=ttvs)

    logging.info(f"Processed data saved to {output_dir}ttv_detection_data.npz")
    print(f"Processed data saved to {output_dir}ttv_detection_data.npz")
    print(f"Total samples: {len(X)}")
    print(f"Positive labels (TTVs): {np.sum(y)}")
    print(f"Percentage of positive labels: {np.sum(y) / len(y) * 100:.2f}%")

    logging.info("TTV detection data preparation completed")

# TODO (if needed):
# - Implement more sophisticated transit detection methods.
# - Handle missing transits or partial light curves.
# - Account for measurement uncertainties.
# - Deal with multi-planet systems where TTVs might be more complex.