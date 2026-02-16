# Dataset Characteristics

## Overview

This stage explores and analyzes the prepared dataset to understand its key characteristics, distributions, correlations, and patterns. Statistical analysis and visualizations help identify important features for forecasting bakery sales.

## Implementation

### Notebooks
- [**`DataCharacter.ipynb`**](DataCharacter.ipynb): Comprehensive exploratory data analysis on merged dataset with extended weather features
- [**`train_data_characteristics.ipynb`**](train_data_characteristics.ipynb): Focused analysis on cleaned training data

## Dataset Overview

### Dataset Dimensions
- **Training Set**: 7,487 rows × 17 features (2013-07-01 to 2017-07-31)
- **Validation Set**: 1,841 rows × 18 features (2017-08-01 to 2018-07-31)
- **Test Set**: 1,830 rows × 15 features (2018-08-01 to 2019-07-31)

### Features
- **Target Variable**: `Umsatz_umsatz` (sales amount)
- **Product Groups**: `Warengruppe_umsatz` (1.0 to 6.0)
- **Weather Features**: Temperature, cloud coverage, wind speed, sunshine hours, precipitation
- **Temporal Features**: Day of week, month, weekend indicators
- **Event Features**: Kieler Woche, public holidays, school holidays
- **Engineered Features**: Rolling 7-day average, next day holiday indicator

## Missing Values

### Analysis Results
- **After Cleaning**: No missing values in training and validation datasets
- **Weather Code**: Removed due to high missingness (2,822 out of 11,607 rows)
- **Extended Weather Data**: Missing values filled using Open-Meteo API
- **Test Data**: Contains NaN in sales columns (by design for prediction)

**Visualization**: Missing value heatmaps created using `plot_missing_heatmap()` function to identify patterns

## Feature Distributions

### Sales Distribution (`Umsatz_umsatz`)
- Mean sales per product group varies significantly
- **Product Group Analysis**: Different product categories show distinct sales patterns
- **Temporal Patterns**: 
  - Higher sales on certain weekdays
  - Seasonal variations observed
  - School holidays show notable impact

### Weather Features
- **Temperature**: Range from cold to warm with seasonal patterns
- **Sunshine Hours**: Strong seasonal variation, correlated with sales
- **Cloud Coverage**: Inversely related to sales
- **Precipitation**: Mixed patterns with sales

### Confidence Intervals
- 95% confidence intervals calculated for numeric features
- Bar charts with error bars show variability in feature distributions
- Key features display distinct statistical properties

## Correlations

### Correlation with Sales (`Umsatz_umsatz`)
Sorted by correlation strength (descending):

**Positive Correlations**:
- `temperature_2m_mean`: **0.224** (strongest weather predictor)
- `Temperatur_weather`: **0.223**
- `school_holiday`: **0.176** (significant holiday effect)
- `sunshine_hours`: **0.172**
- `sunshine_duration`: **0.172**
- `day_of_week`: **0.136** (weekly patterns)
- `KielerWoche_kiwo`: **0.058** (local event impact)
- `public_holiday`: **0.044**
- `Windgeschwindigkeit_weather`: **0.015**

**Negative Correlations**:
- `Bewoelkung_weather`: **-0.089** (cloud coverage reduces sales)
- `precipitation_hours`: **-0.036**
- `rain_sum`: **-0.006**

### Statistical Significance Testing
- **T-tests performed** for holiday effects:
  - Public holidays: Statistically significant impact
  - School holidays: Statistically significant positive impact
  - Next day holiday: Effect analyzed for pre-holiday sales patterns

### Correlation Matrix
- Full correlation heatmaps generated
- Multicollinearity identified between some weather features (e.g., sunshine_hours and sunshine_duration)
- Feature selection informed by correlation analysis

## Key Insights

1. **Weather Impact**: Temperature and sunshine hours are strongest weather predictors of sales
2. **Holiday Effects**: School holidays show stronger positive correlation than public holidays
3. **Weekly Patterns**: Day of week shows moderate correlation, indicating weekly seasonality
4. **Product Variation**: Different product groups (Warengruppe) have distinct sales patterns
5. **Feature Redundancy**: Some weather features are highly correlated (e.g., `temperature_2m_mean` and `Temperatur_weather`)
6. **Seasonal Trends**: Sales vary significantly across months and seasons
7. **Data Quality**: No missing values after cleaning ensures reliable model training

## Visualizations

- **Correlation heatmaps**: Feature-to-feature and feature-to-target relationships
- **Scatter plots**: Bivariate relationships (e.g., temperature vs. sales)
- **Distribution plots**: Histograms and box plots for feature distributions
- **Missing value heatmaps**: Verification of data completeness
- **Time series plots**: Sales patterns over time
- **Bar charts with confidence intervals**: Statistical summaries of features

## Usage

To reproduce the analysis:
1. Ensure processed data files exist in `data/processed/` directory
2. Run `DataCharacter.ipynb` for comprehensive exploratory analysis
3. Run `train_data_characteristics.ipynb` for training data-specific analysis

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scipy (for statistical tests)
- sklearn (for preprocessing utilities)