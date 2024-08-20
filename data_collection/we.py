import pandas as pd

# Load the CSV file
file_path = 'TESS Project Candidates 2024-07-23.csv'  # Replace with the path to your CSV file
df = pd.read_csv(file_path)

# Count the number of unique planetary systems
unique_systems = df['tid'].nunique()

# Count the number of individual planets
total_planets = len(df)

print(f"Number of unique planetary systems: {unique_systems}")
print(f"Number of individual planets: {total_planets}")