# Data Import and Preparation

## Overview

This stage focuses on importing and preparing the dataset for your bakery sales forecasting project. Efficient and accurate data preparation is fundamental for successful forecasting models. The task includes importing the dataset into Python, cleaning it, and constructing new variables that are pertinent to forecasting sales.

## Guidelines

Focus on the following key issues:

### Importing/Exporting Data
- Put all the csv with their original name in data directory. If you are using shared codespace they should already be present. `data` directory is ignored and the .csv files are not pushed to git
- We need `processed` folder to exists under `data` i.e `data\processed`

### Merging Data from different sources

### Data Cleaning

### Handling Missing Values

### Constructing New Variables
- (Develop and integrate new variables that could significantly influence sales predictions. This may include date-related features (like day of the week, holidays), weather conditions, or special events.)

### Data Transformation
- (e.g., converting metric data to categorical data)

## Notes
- `save_dataframe_to_csv(df_merged_extended_weather, "df_merged_extended_weather")` requires `processed` directory inside data to exist.
- `plot_missing_heatmap` function can be used for plotting our dataframes so we can visualize them.

## Task
Create a Jupyter Notebook or Python-script to conduct the data import and preparation process. Ensure each step is well-documented.
