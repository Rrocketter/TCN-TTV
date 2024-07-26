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
logging.basicConfig(filename='k2_download.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load the list of EPIC IDs from the 'epic_hostname' column
epic_ids = pd.read_csv('K2 Planets July 23.csv')['epic_hostname']

# Remove the 'EPIC ' prefix from the EPIC IDs
epic_ids = epic_ids.str.replace('EPIC ', '')

# Ignore specific warnings
warnings.simplefilter('ignore', category=AstropyWarning)
warnings.simplefilter('ignore', category=UserWarning)

# Function to download K2 light curves
def download_k2_data(epic_id, output_dir):
    target_name = f"EPIC {epic_id}"
    logging.info(f"Processing {target_name}")

    try:
        # Search for light curve files using lightkurve
        search_result = search_lightcurve(target_name, mission='K2')
        logging.info(f"Found {len(search_result)} light curves for {target_name}")

        if len(search_result) > 0:
            # Filter out unsupported types
            valid_products = search_result[(search_result.table['productSubGroupDescription'] != 'K2SC')]

            if len(valid_products) > 0:
                # Download the light curve files
                lightcurves = valid_products.download_all()
                if lightcurves:
                    for i, lc in enumerate(lightcurves):
                        filename = os.path.join(output_dir, f"{epic_id}_light_curve_{i + 1}.fits")
                        lc.to_fits(filename, overwrite=True)
                        logging.info(f"Saved light curve data for {target_name} to file {filename}")
                    return True
                else:
                    logging.warning(f"No valid light curve downloaded for {target_name}")
            else:
                logging.warning(f"No valid light curves found for {target_name}")
        else:
            logging.warning(f"No light curves found for {target_name}")
    except Exception as e:
        logging.error(f"An error occurred while processing {target_name}: {str(e)}")

    return False

# Function to process all EPIC IDs with progress tracking and error handling
def process_all_epic_ids(epic_ids, output_directory, resume_from=0):
    os.makedirs(output_directory, exist_ok=True)

    successful_downloads = 0
    for i, epic_id in enumerate(tqdm(epic_ids[resume_from:], initial=resume_from, total=len(epic_ids))):
        if download_k2_data(epic_id, output_directory):
            successful_downloads += 1

        # Add a small delay to avoid overwhelming the server
        time.sleep(1)

        # Periodically save progress
        if (i + 1) % 100 == 0:
            logging.info(f"Processed {i + 1} EPIC IDs. Successful downloads: {successful_downloads}")

    logging.info(f"Finished processing all EPIC IDs. Total successful downloads: {successful_downloads}")

# Example usage
output_directory = '../data/k2/'

# Process all EPIC IDs
# process_all_epic_ids(epic_ids, output_directory)

# If the script was interrupted, you can resume from a specific index
process_all_epic_ids(epic_ids, output_directory, resume_from=0)
