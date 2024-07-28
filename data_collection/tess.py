from astroquery.mast import Observations, Catalogs
import pandas as pd
import os
import time
from astropy import units as u
from astropy.coordinates import SkyCoord
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(filename='tess_download.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load the list of TIC IDs
tic_ids = pd.read_csv('TESS Project Candidates 2024-07-23.csv')['tid']


# Function to download TESS light curves
def download_tess_data(tic_id, output_dir):
    target_name = f"TIC {tic_id}"
    logging.info(f"Processing {target_name}")

    try:
        # Query the TIC catalog for the target's coordinates
        tic_result = Catalogs.query_criteria(catalog="TIC", ID=tic_id)
        if len(tic_result) == 0:
            logging.warning(f"Could not find TIC entry for {target_name}")
            return None

        # Create a SkyCoord object
        target_coord = SkyCoord(ra=tic_result['ra'][0], dec=tic_result['dec'][0], unit=(u.degree, u.degree))

        # Query observations
        obs_table = Observations.query_criteria(coordinates=target_coord, radius=0.02 * u.deg,
                                                project='TESS', dataproduct_type='timeseries')
        logging.info(f"Found {len(obs_table)} observations for {target_name}")

        if len(obs_table) > 0:
            # Filter for 2-minute cadence data
            two_min_cadence_obs = obs_table[obs_table['dataproduct_type'] == 'timeseries']

            if len(two_min_cadence_obs) == 0:
                logging.warning(f"No 2-minute cadence data found for {target_name}")
                return None

            # Download the light curves
            data_products = Observations.get_product_list(two_min_cadence_obs)
            if len(data_products) > 0:
                # Filter for SPOC products (2-minute cadence)
                spoc_products = data_products[data_products['productSubGroupDescription'] == 'LC']
                if len(spoc_products) > 0:
                    manifest = Observations.download_products(spoc_products, download_dir=output_dir)
                    logging.info(f"Downloaded products for {target_name}")
                    return manifest
                else:
                    logging.warning(f"No SPOC light curve products found for {target_name}")
            else:
                logging.warning(f"No data products found for 2-minute cadence observations of {target_name}")
        else:
            logging.warning(f"No observations found for {target_name}")
    except Exception as e:
        logging.error(f"An error occurred while processing {target_name}: {str(e)}")

    return None


# Function to process all TIC IDs with progress tracking and error handling
def process_all_tics(tic_ids, output_directory, resume_from=0):
    os.makedirs(output_directory, exist_ok=True)

    successful_downloads = 0
    for i, tic_id in enumerate(tqdm(tic_ids[resume_from:], initial=resume_from, total=len(tic_ids))):
        if download_tess_data(tic_id, output_directory) is not None:
            successful_downloads += 1

        # Add a small delay to avoid overwhelming the server
        time.sleep(1)

        # Periodically save progress
        if (i + 1) % 100 == 0:
            logging.info(f"Processed {i + 1} TIC IDs. Successful downloads: {successful_downloads}")

    logging.info(f"Finished processing all TIC IDs. Total successful downloads: {successful_downloads}")


# Example usage
output_directory = '../data/tess/'

# Process all TIC IDs
# process_all_tics(tic_ids, output_directory)

# If the script was interrupted, you can resume from a specific index
process_all_tics(tic_ids, output_directory, resume_from=1765)