# Data Preparation

## Overview

This stage focuses on importing, merging, cleaning, and preparing the bakery sales dataset for forecasting models. The data preparation process integrates multiple data sources, handles missing values, creates engineered features, and splits the data into train, validation, and test sets.

## Implementation

### Notebooks
- [**`yousra_descriptive_stats.ipynb`**](yousra_descriptive_stats.ipynb): Main notebook for data merging, cleaning, feature engineering, and dataset splitting
- [**`process_holidays.ipynb`**](process_holidays.ipynb): Processes public and school holiday data for Schleswig-Holstein (2013-2019)
- [**`utils.py`**](utils.py): Utility functions including `plot_missing_heatmap()` for visualization

## Data Importing

### Source Data Files
Located in `data/` directory:
- **`umsatzdaten_gekuerzt.csv`**: Sales data (Umsatz) by product group (Warengruppe)
- **`wetter.csv`**: Weather data including temperature, cloud coverage, wind speed, and weather codes
- **`kiwo.csv`**: Kieler Woche event indicator data
- **`official_holidays.csv`**: Public holidays in Schleswig-Holstein
- **`school_holidays.csv`**: School holidays in Schleswig-Holstein
- **`test.csv`**: Test dataset for final predictions

All source CSV files are loaded using pandas and dates are normalized to `datetime` format.

## Merging Data from Different Sources

### Merge Strategy
- Full outer join on `Datum` (date) field to preserve all available data
- Sequential merging: Sales → Weather → Kiwo → Holidays
- Extended weather data fetched from **Open-Meteo API** to fill missing weather information
- Holiday data merged using date ranges converted to binary indicators

### Holiday Data Processing
- Date ranges from `official_holidays.csv` and `school_holidays.csv` expanded to daily records
- Binary columns created: `public_holiday`, `school_holiday`
- Additional feature: `next_day_holiday` (shifted public holiday indicator)

## Data Cleaning

### Date Standardization
- All date columns converted to `datetime` format using `pd.to_datetime()` with error coercion
- Invalid date rows dropped
- Duplicate date columns removed

### Missing Value Handling
- **Weather codes**: High percentage missing (2,822 out of ~11,607 rows) - column dropped or filled with category `-1`
- **Extended weather data**: Missing weather information supplemented using Open-Meteo API with daily aggregated data
- **Sales data**: Rows with missing `Umsatz_umsatz` and `Warengruppe_umsatz` removed from training/validation sets
- Missing values visualized using `plot_missing_heatmap()` function

## Constructing New Variables

### Date-Related Features
- **`day_of_week`**: Day of week (0=Monday, 6=Sunday)
- **`is_saturday`**: Binary indicator for Saturday
- **`is_sunday`**: Binary indicator for Sunday
- **`month`**: Month of the year (1-12)

### Sales Features
- **`umsatz_rolling7`**: 7-day rolling average of sales (Umsatz_umsatz)

### Weather Features (Extended)
- **`sunshine_duration`**: Total sunshine duration (seconds)
- **`temperature_2m_mean`**: Mean temperature at 2 meters
- **`sunshine_hours`**: Sunshine hours (converted from sunshine_duration)

### Holiday Features
- **`public_holiday`**: Binary indicator (0/1)
- **`school_holiday`**: Binary indicator (0/1)
- **`next_day_holiday`**: Binary indicator for day before public holiday

## Data Transformation

### Data Splitting
Data split into three temporal sets:
- **Training Set**: 2013-07-01 to 2017-07-31 (7,487 rows after cleaning)
- **Validation Set**: 2017-08-01 to 2018-07-31 (1,841 rows after cleaning)
- **Test Set**: 2018-08-01 to 2019-07-31 (1,830 rows after cleaning)

### Output Files
Processed datasets saved to `data/processed/`:
- `df_train_data_cleaned.csv`: Cleaned training data
- `df_validation_data_cleaned.csv`: Cleaned validation data
- `df_test_data_cleaned.csv`: Cleaned test data
- `df_train_data_raw.csv`: Raw (unfiltered) training data
- `df_validation_data_raw.csv`: Raw validation data
- `df_test_data_raw.csv`: Raw test data
- `df_holidays.csv`: Processed holiday indicators
- `df_merged_extended_weather.csv`: Merged data with extended weather
- `df_extended_weather_holidays.csv`: Final merged dataset with all features
- `df_extended_extras_with_test.csv`: Complete dataset including test data

## Key Insights

- **Final Feature Set**: 18-21 features including sales, weather, holidays, temporal features
- **No Missing Values**: After cleaning, training and validation sets have no missing values
- **Temporal Integrity**: Data split preserves temporal order for time series forecasting
- **Feature Engineering**: Rich set of engineered features to capture seasonal, weekly, and holiday patterns

## Usage

To reproduce the data preparation:
1. Ensure all source CSV files are in `data/` directory
2. Create `data/processed/` directory if not exists
3. Run `process_holidays.ipynb` to generate holiday features
4. Run `yousra_descriptive_stats.ipynb` for complete data pipeline

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn (for visualizations)
- Open-Meteo API (for extended weather data)
