# Temporal Convolutional Networks for Exoplanet Transit Timing Variations

### Objective
Develop a machine learning approach using Temporal Convolutional Networks (TCNs) to detect and analyze transit timing variations (TTVs) in multi-planet systems. This method aims to reveal hidden planets or provide deeper insights into planetary dynamics.

## Table of Contents
1. [Installation](#installation)
    - [Clone the Repository](#clone-the-repository)
    - [Create a Virtual Environment](#create-a-virtual-environment)
    - [Install Dependencies](#install-dependencies)
2. [Data Collection](#data-collection)
    - [Kepler](#Kepler)
    - [K2](#K2)
    - [TESS](#TESS)
3. [Data Preprocessing](#data-preprocessing)
    - [Step 1](#Step 1)
    - [Step 2](#Step 2)
    - [Download Link](#Download-link)
4. [Running the Model](#running-the-model)

## Installation

### Clone the Repository
To get started, clone the repository to your local machine using the following command:
```bash
git clone https://github.com/your-username/your-repository.git
```
Navigate into the project directory:
```bash
cd your-repository
```

### Create a Virtual Environment
Create a virtual environment to manage dependencies:
```bash
python -m venv env
```
Activate the virtual environment:

- On Windows:
    ```bash
    .\env\Scripts\activate
    ```

- On macOS and Linux:
    ```bash
    source env/bin/activate
    ```

### Install Dependencies
Install the required dependencies using pip:
```bash
pip install -r requirements.txt
```

## Data Collection
Use the already provided csv files from the missions and run the respective python scripts to download the data.
```bash
cd data_collection
```

### Kepler
Use the [Kepler Objects of Interest 2024-07-23.csv](data_collection%2FKepler%20Objects%20of%20Interest%202024-07-23.csv) to download the data from the Kepler mission.

To download the Kepler data, run the following command:
```bash
python3  kepler.py 
```

### K2
Use the [K2 Planets July 23.csv](data_collection%2FK2%20Planets%20July%2023.csv) to download the data from the K2 mission.

To download the Kepler data, run the following command:
```bash
python3  k2.py 
```

### TESS
Use the [TESS Project Candidates 2024-07-23.csv](data_collection%2FTESS%20Project%20Candidates%202024-07-23.csv) to download the data from the K2 mission.

To download the Kepler data, run the following command:
```bash
python3  tess.py 
```


## Data Preprocessing
Cleaning the light curve and taking only information that we need.
This includes cleaning, transforming, and splitting the data.

```bash
cd preprocessing
```

### Step 1
Run the general data preprocessing script:
```bash
python preprocess.py
```

### Step 2
Run the script to prepare the data to be feed into the ml model:
```bash
python preprocess_ml.py
```

### Download Link
[Download the preprocessed data](https://drive.google.com/file/d/1xAbxV0cSpKsVz79PRKnj6BMrJsiDoW5M/view?usp=sharing)
(Around 5.5 GB)

## Running the Model
Run the custom TCN model to detect TTVs in the light curves.

```bash
cd ml
```

### Train + Evaluate the Model

Change the location of the data in the actual_model.py file to where the data is stored.
``` python 
data = np.load("../ml_data/ttv-dataset/ttv_detection_data.npz") 
```

Run the TCN model
```bash
python actual_model.py
```

[//]: # (### Evaluate the Model)

[//]: # (```bash)

[//]: # (python scripts/evaluate_model.py --model model/model.pkl --data data/test_data.csv)

[//]: # (```)


### Research Papers
1. **"Transit Timing Variations as a Method for Detecting Exoplanets"**:
   - Holman, M. J., & Murray, N. W. (2005). *Science*, 307(5713), 1288-1291.
   - This foundational paper discusses the TTV method and its potential for detecting exoplanets.

2. **"Application of Convolutional Neural Networks to Exoplanet Detection in Light Curves"**:
   - Shallue, C. J., & Vanderburg, A. (2018). *The Astronomical Journal*, 155(2), 94.
   - Explores the use of convolutional neural networks for detecting exoplanets, providing a basis for adapting CNN techniques to TCNs for TTV analysis.

3. **"Neural Network Approaches to TTV Detection"**:
   - Pearson, K. A., Palafox, L., & Griffith, C. A. (2018). *Monthly Notices of the Royal Astronomical Society*, 474(4), 4782-4796.
   - Investigates the application of neural networks for transit timing variation detection.

4. **"Machine Learning for Exoplanet Detection and Characterization"**:
   - Armstrong, D. J., Pollacco, D., & Santerne, A. (2017). *Monthly Notices of the Royal Astronomical Society*, 465(3), 2634-2654.
   - Reviews various machine learning methods used in exoplanet detection, including TTV analysis.