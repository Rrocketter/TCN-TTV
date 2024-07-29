import os
import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy import signal
from astropy.timeseries import LombScargle
import logging

logging.basicConfig(filename='processing.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def inspect_fits_file(fits_file):
    with fits.open(fits_file) as hdul:
        logging.info(f"Inspecting file: {fits_file}")
        if 'LIGHTCURVE' in hdul:
            data = Table(hdul['LIGHTCURVE'].data)
            logging.info(f"Kepler LIGHTCURVE columns: {data.colnames}")
        else:
            data = Table(hdul[1].data)
            logging.info(f"Other mission columns: {data.colnames}")


def preprocess_light_curve(fits_file, window_size=1001, polyorder=2):
    try:
        with fits.open(fits_file) as hdul:
            # Check  structure
            if 'LIGHTCURVE' in hdul:
                # Kepler
                data = Table(hdul['LIGHTCURVE'].data)
                time = data['TIME']
                flux = None
                if 'SAP_FLUX' in data.colnames:
                    flux = data['SAP_FLUX']
                elif 'PDCSAP_FLUX' in data.colnames:
                    flux = data['PDCSAP_FLUX']
                elif 'FLUX' in data.colnames:
                    flux = data['FLUX']
                    logging.info(f"Using alternative flux column: FLUX")
                else:
                    raise KeyError("No suitable flux column found in Kepler data.")
            else:
                # other missions
                data = Table(hdul[1].data)
                time = data['TIME']
                flux = None
                if 'PDCSAP_FLUX' in data.colnames:
                    flux = data['PDCSAP_FLUX']
                elif 'SAP_FLUX' in data.colnames:
                    flux = data['SAP_FLUX']
                elif 'FLUX' in data.colnames:
                    flux = data['FLUX']
                    logging.info(f"Using alternative flux column: FLUX")
                else:
                    raise KeyError("No suitable flux column found in the data.")

        # Remove NaN values
        mask = np.isfinite(time) & np.isfinite(flux)
        time = time[mask]
        flux = flux[mask]

        # Normalize flux
        flux_norm = flux / np.median(flux)

        # Detrend the light curve using Savitzky-Golay filter
        flux_detrended = signal.savgol_filter(flux_norm, window_size, polyorder)

        # Calculate residuals (highlight transit signals)
        residuals = flux_norm - flux_detrended

        # Interpolate to regular time grid (uniform sampling)
        time_regular = np.linspace(time.min(), time.max(), len(time))
        flux_interpolated = np.interp(time_regular, time, residuals)

        return time_regular, flux_interpolated

    except Exception as e:
        logging.error(f"Error in preprocess_light_curve: {e}")
        raise


def remove_stellar_variability(time, flux, min_period=0.1, max_period=20):
    frequency, power = LombScargle(time, flux).autopower(minimum_frequency=1 / max_period,
                                                         maximum_frequency=1 / min_period)
    best_frequency = frequency[np.argmax(power)]
    best_period = 1 / best_frequency

    model = LombScargle(time, flux).model(time, best_period)
    flux_detrended = flux - model + np.median(flux)

    return flux_detrended


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


def process_file(fits_file, output_dir):
    output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(fits_file))[0]}.npz")
    logging.info(f"Processing: {fits_file}")
    try:
        time, flux = preprocess_light_curve(fits_file)
        flux_detrended = remove_stellar_variability(time, flux)
        tcn_input = prepare_tcn_input(time, flux_detrended)
        np.savez(output_file, time=time, flux=flux_detrended, tcn_input=tcn_input)
        logging.info(f"Saved processed data to: {output_file}")
        logging.info(f"TCN input shape: {tcn_input.shape}")
        logging.info("---")
    except KeyError as e:
        logging.error(f"Column error in {fits_file}: {str(e)}")
        logging.info("---")
    except Exception as e:
        logging.error(f"Error processing {fits_file}: {str(e)}")
        logging.info("---")


def process_directory(directory, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    if 'tess' in directory.lower():
        # TESS
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('_lc.fits'):
                    fits_file = os.path.join(root, file)
                    process_file(fits_file, output_dir)
    else:
        # Kepler and K2
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.fits'):
                    fits_file = os.path.join(root, file)
                    process_file(fits_file, output_dir)


if __name__ == "__main__":
    base_dir = '../data'

    directories = [
        # os.path.join(base_dir, 'kepler'),
        # os.path.join(base_dir, 'k2'),
        os.path.join(base_dir, 'tess', 'mastDownload', 'TESS')
    ]

    output_base_dir = '../processed_data/'

    # Process directory
    for directory in directories:
        logging.info(f"Processing directory: {directory}")
        output_dir = os.path.join(output_base_dir, os.path.basename(directory))
        process_directory(directory, output_dir)

logging.info("Preprocessing complete. Data is ready for ML-specific preprocessing.")
