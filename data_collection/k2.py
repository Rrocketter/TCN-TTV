import pandas as pd
import logging
import os
import time
from lightkurve import search_lightcurve
import warnings
from astropy.utils.exceptions import AstropyWarning
from tqdm import tqdm
import shutil

# clear cache
cache_dir = os.path.expanduser("~/.lightkurve/cache/")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)

# logging
logging.basicConfig(filename='k2_download.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load the list of EPIC IDs from the 'epic_hostname' column
epic_ids = pd.read_csv('K2 Planets July 23.csv')['epic_hostname']

# Remove  'EPIC ' from EPIC IDs
epic_ids = epic_ids.str.replace('EPIC ', '')

# Ignore warnings
warnings.simplefilter('ignore', category=AstropyWarning)
warnings.simplefilter('ignore', category=UserWarning)

# download K2 light curves
def download_k2_data(epic_id, output_dir):
    target_name = f"EPIC {epic_id}"
    logging.info(f"Processing {target_name}")

    try:
        # search for light curve files 
        search_result = search_lightcurve(target_name, mission='K2')
        logging.info(f"Found {len(search_result)} light curves for {target_name}")

        if len(search_result) > 0:
            # take out out unsupported types
            valid_products = search_result[(search_result.table['productSubGroupDescription'] != 'K2SC')]

            if len(valid_products) > 0:
                # download light curve files
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

# process EPIC IDs
def process_all_epic_ids(epic_ids, output_directory, resume_from=0):
    os.makedirs(output_directory, exist_ok=True)

    successful_downloads = 0
    for i, epic_id in enumerate(tqdm(epic_ids[resume_from:], initial=resume_from, total=len(epic_ids))):
        if download_k2_data(epic_id, output_directory):
            successful_downloads += 1

        #  delay to avoid overwhelming the server
        time.sleep(1)

        # save progress
        if (i + 1) % 100 == 0:
            logging.info(f"Processed {i + 1} EPIC IDs. Successful downloads: {successful_downloads}")

    logging.info(f"Finished processing all EPIC IDs. Total successful downloads: {successful_downloads}")

output_directory = '../data/k2/'

# process_all_epic_ids(epic_ids, output_directory)

process_all_epic_ids(epic_ids, output_directory, resume_from=0)
