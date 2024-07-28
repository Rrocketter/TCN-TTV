from astropy.io import fits
from astropy.table import Table

def inspect_fits_file(fits_file):
    with fits.open(fits_file) as hdul:
        print(f"Inspecting file: {fits_file}")
        if 'LIGHTCURVE' in hdul:
            data = Table(hdul['LIGHTCURVE'].data)
            print(f"Kepler LIGHTCURVE columns: {data.colnames}")
        else:
            data = Table(hdul[1].data)
            print(f"Other mission columns: {data.colnames}")

# Example of how to use the inspect_fits_file function
fits_file = '../data/kepler/4263293_light_curve_23.fits'
inspect_fits_file(fits_file)
