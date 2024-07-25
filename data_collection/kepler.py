import pandas as pd
import logging
import os
import time
from lightkurve import search_lightcurve
import warnings
from astropy.utils.exceptions import AstropyWarning
from tqdm import tqdm
import shutil

# Clear cache to avoid potential issues
cache_dir = os.path.expanduser("~/.lightkurve/cache/")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)

# Set up logging
logging.basicConfig(filename='kepler_download.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load the list of Kepler IDs
kepler_ids = pd.read_csv('Kepler Objects of Interest 2024-07-23.csv')['kepid']

# Ignore specific warnings
warnings.simplefilter('ignore', category=AstropyWarning)
warnings.simplefilter('ignore', category=UserWarning)

# Function to download Kepler light curves
def download_kepler_data(kepler_id, output_dir):
    target_name = f"KIC {kepler_id}"
    logging.info(f"Processing {target_name}")

    try:
        # Search for light curve files using lightkurve
        search_result = search_lightcurve(target_name, mission='Kepler')
        logging.info(f"Found {len(search_result)} light curves for {target_name}")

        if len(search_result) > 0:
            # Download the light curve files
            lightcurve = search_result.download_all()
            if lightcurve:
                for i, lc in enumerate(lightcurve):
                    filename = os.path.join(output_dir, f"{kepler_id}_light_curve_{i + 1}.fits")
                    lc.to_fits(filename, overwrite=True)
                    logging.info(f"Saved light curve data for {target_name} to file {filename}")
                return True
            else:
                logging.warning(f"No light curve downloaded for {target_name}")
        else:
            logging.warning(f"No light curves found for {target_name}")
    except Exception as e:
        logging.error(f"An error occurred while processing {target_name}: {str(e)}")

    return False

# Function to process all Kepler IDs with progress tracking and error handling
def process_all_kepler_ids(kepler_ids, output_directory, resume_from=0):
    os.makedirs(output_directory, exist_ok=True)

    successful_downloads = 0
    for i, kepler_id in enumerate(tqdm(kepler_ids[resume_from:], initial=resume_from, total=len(kepler_ids))):
        if download_kepler_data(kepler_id, output_directory):
            successful_downloads += 1

        # Add a small delay to avoid overwhelming the server
        time.sleep(1)

        # Periodically save progress
        if (i + 1) % 100 == 0:
            logging.info(f"Processed {i + 1} Kepler IDs. Successful downloads: {successful_downloads}")

    logging.info(f"Finished processing all Kepler IDs. Total successful downloads: {successful_downloads}")

# Example usage
output_directory = '../data/kepler/'

# Process all Kepler IDs
# process_all_kepler_ids(kepler_ids, output_directory)

# If the script was interrupted, you can resume from a specific index
process_all_kepler_ids(kepler_ids, output_directory, resume_from=885)
