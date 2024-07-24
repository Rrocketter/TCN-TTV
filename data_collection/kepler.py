from astroquery.mast import Observations, Catalogs
import pandas as pd
import os
import time
from astropy import units as u
from astropy.coordinates import SkyCoord
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(filename='kepler_download.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load the list of Kepler IDs (KIC)
ids = pd.read_csv('Kepler_targets.csv')['kic_id']


def download_kepler_data(kic_id, output_dir):
    target_name = f"KIC {kic_id}"
    logging.info(f"Processing {target_name}")
    try:
        # Query the Kepler catalog
        cat_result = Catalogs.query_criteria(catalog="Kepler", kic_kepler_id=kic_id)
        if len(cat_result) == 0:
            logging.warning(f"Could not find catalog entry for {target_name}")
            return None

        # Create a SkyCoord object
        target_coord = SkyCoord(ra=cat_result['ra'][0], dec=cat_result['dec'][0], unit=(u.degree, u.degree))

        # Query observations
        obs_table = Observations.query_criteria(coordinates=target_coord, radius=0.02 * u.deg, project=['Kepler'])
        logging.info(f"Found {len(obs_table)} observations for {target_name}")

        if len(obs_table) > 0:
            # Filter for long cadence data
            lc_obs = obs_table[obs_table['dataproduct_type'] == 'timeseries']
            if len(lc_obs) == 0:
                logging.warning(f"No long cadence data found for {target_name}")
                return None

            # Download the light curves
            data_products = Observations.get_product_list(lc_obs)
            if len(data_products) > 0:
                # Filter for light curve products
                lc_products = data_products[data_products['productSubGroupDescription'] == 'LC']
                if len(lc_products) > 0:
                    manifest = Observations.download_products(lc_products, download_dir=output_dir)
                    logging.info(f"Downloaded products for {target_name}")
                    return manifest
                else:
                    logging.warning(f"No light curve products found for {target_name}")
            else:
                logging.warning(f"No data products found for observations of {target_name}")
        else:
            logging.warning(f"No observations found for {target_name}")
    except Exception as e:
        logging.error(f"An error occurred while processing {target_name}: {str(e)}")
    return None


def process_all_kepler_ids(target_ids, output_directory, resume_from=0):
    os.makedirs(output_directory, exist_ok=True)
    successful_downloads = 0
    for i, target_id in enumerate(tqdm(target_ids[resume_from:], initial=resume_from, total=len(target_ids))):
        if download_kepler_data(target_id, output_directory) is not None:
            successful_downloads += 1
        # Add a small delay to avoid overwhelming the server
        time.sleep(1)
        # Periodically save progress
        if (i + 1) % 100 == 0:
            logging.info(f"Processed {i + 1} target IDs. Successful downloads: {successful_downloads}")
    logging.info(f"Finished processing all Kepler target IDs. Total successful downloads: {successful_downloads}")


# Example usage
output_directory = '../data/kepler/'
# Process all target IDs
process_all_kepler_ids(ids, output_directory)
# If the script was interrupted, you can resume from a specific index
# process_all_kepler_ids(ids, output_directory, resume_from=1000)