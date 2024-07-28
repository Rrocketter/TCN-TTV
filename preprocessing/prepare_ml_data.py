import numpy as np
import os
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit

def load_processed_data(file_path):
    """Load processed data from .npz file."""
    with np.load(file_path) as data:
        return [data[arr] for arr in data.files]

def pad_or_truncate(array, target_len):
    """Pad or truncate array to target length."""
    if len(array) > target_len:
        return array[:target_len]
    else:
        return np.pad(array, (0, target_len - len(array)), mode='constant')

def detect_transit_time(time, flux):
    """Detect the time of transit minimum."""
    return time[np.argmin(flux)]

def linear_ephemeris(epoch, t0, period):
    """Linear ephemeris model."""
    return t0 + epoch * period

def fit_linear_ephemeris(transit_times):
    """Fit a linear ephemeris to the observed transit times."""
    epochs = np.arange(len(transit_times))
    popt, _ = curve_fit(linear_ephemeris, epochs, transit_times)
    return popt

def calculate_ttv(observed_time, expected_time, period):
    """Calculate TTV in units of minutes."""
    return (observed_time - expected_time) * 24 * 60  # Convert days to minutes

def prepare_data_for_ttv_detection(processed_data_dir, window_size=1000, test_size=0.2, batch_size=32, ttv_threshold=5):
    """Prepare data for TTV detection model."""
    all_transits = []
    all_labels = []
    all_ttvs = []

    # Load and format all processed data
    for file in os.listdir(processed_data_dir):
        if file.endswith('.npz'):
            file_path = os.path.join(processed_data_dir, file)
            transit_windows = load_processed_data(file_path)

            # Detect transit times
            observed_transit_times = [detect_transit_time(time, flux) for time, flux in transit_windows]

            # Fit linear ephemeris
            t0, period = fit_linear_ephemeris(observed_transit_times)

            for i, (time, flux) in enumerate(transit_windows):
                formatted_flux = pad_or_truncate(flux, window_size)
                all_transits.append(formatted_flux)

                # Calculate expected transit time
                expected_transit_time = linear_ephemeris(i, t0, period)

                # Calculate TTV
                observed_transit_time = observed_transit_times[i]
                ttv = calculate_ttv(observed_transit_time, expected_transit_time, period)
                all_ttvs.append(ttv)

                # Label based on whether TTV exceeds threshold
                all_labels.append(1 if abs(ttv) > ttv_threshold else 0)

    # Convert to numpy arrays
    X = np.array(all_transits)
    y = np.array(all_labels)
    ttvs = np.array(all_ttvs)

    # Split into train and test sets
    X_train, X_test, y_train, y_test, ttvs_train, ttvs_test = train_test_split(X, y, ttvs, test_size=test_size, random_state=42)

    # Create batches
    def batch_generator(X, y, ttvs, batch_size):
        while True:
            for i in range(0, len(X), batch_size):
                yield X[i:i + batch_size], y[i:i + batch_size], ttvs[i:i + batch_size]

    train_generator = batch_generator(X_train, y_train, ttvs_train, batch_size)
    test_generator = batch_generator(X_test, y_test, ttvs_test, batch_size)

    return train_generator, test_generator, X_train.shape[0], X_test.shape[0]

if __name__ == "__main__":
    processed_data_dir = "path/to/processed/data"
    train_gen, test_gen, train_steps, test_steps = prepare_data_for_ttv_detection(processed_data_dir)

    # Now you can use train_gen and test_gen with your TCN model
    # For example, with a Keras model:
    # model.fit(train_gen, steps_per_epoch=train_steps//batch_size, ...)




#IF NEEDED:
# TODO
# More sophisticated transit detection methods.
# Handling of missing transits or partial light curves.
# Accounting for measurement uncertainties.
# Dealing with multi-planet systems where TTVs might be more complex.