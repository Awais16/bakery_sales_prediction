# Baseline Models

## Overview

This stage establishes baseline models for bakery sales forecasting. Multiple regression approaches are implemented and evaluated to serve as reference points for more complex models. The baseline models include Linear Regression, Polynomial Regression, Cosinor Regression, Random Forest, and XGBoost.

## Implementation

### Notebooks
- [**`LinearRegression_2.ipynb`**](LinearRegression_2.ipynb): Linear and Polynomial Regression models with feature selection experiments
- [**`CosinorRegression.ipynb`**](CosinorRegression.ipynb): Cosinor (rhythmic) regression for capturing periodic sales patterns
- [**`RandomForest.ipynb`**](RandomForest.ipynb): Random Forest ensemble model
- [**`XGBRegressor.ipynb`**](XGBRegressor.ipynb): XGBoost gradient boosting model
- [**`LinearRegression.ipynb`**](LinearRegression.ipynb): Initial linear regression experiments

## Models Implemented

### 1. Linear Regression
**Implementation**: Scikit-learn `LinearRegression`

**Feature Selection**:
- Product group (`Warengruppe_umsatz`) - categorical, one-hot encoded
- Temporal features: `day_of_week`, `month`, `is_saturday`, `is_sunday`
- Weather features: `Temperatur_weather`, `Bewoelkung_weather`, `sunshine_hours`
- Holiday features: `public_holiday`, `school_holiday`, `next_day_holiday`
- Event features: `KielerWoche_kiwo`
- Engineered features: `last_day_of_year`, `last_day_of_year_w5` (Warengruppe 5 specific)

**Performance**:
- **R² Train**: 0.8945 - 0.8965
- **R² Validation**: 0.8465 - 0.8495
- **MAPE Validation**: 0.2258 - 0.2262

**Experiment Tracking**: Multiple feature combinations tested and results logged in `experiment_results_LR` DataFrame

### 2. Polynomial Regression
**Implementation**: Scikit-learn `PolynomialFeatures` + `LinearRegression`

**Approach**:
- Degree 2 polynomial features generated
- Applied to selected numerical features
- Combined train and validation data for final model

**Performance**:
- **R² Train**: ~0.8945
- **R² Validation**: ~0.8475
- **MAPE Validation**: ~0.2262
- **R² Combined**: ~0.8944

**Experiment Tracking**: Results logged in `experiment_results_PR` DataFrame

### 3. Cosinor Regression
**Implementation**: Linear Regression with trigonometric transformations

**Approach**:
- Models periodic/rhythmic patterns using cosine and sine terms
- Captures circadian or seasonal rhythms in sales
- Features: `cos_term` and `sin_term` based on day of week

**Parameters Extracted**:
- **MESOR** (midline): 209.35
- **Amplitude**: 4.16
- **Acrophase**: 1.55 radians

**Performance**:
- **R² Train**: 0.0004 ⚠️ (very poor)
- **R² Validation**: -0.0110 ⚠️ (worse than baseline)
- **MAPE Validation**: 0.8336

**Conclusion**: Cosinor model alone is insufficient; sales patterns are not purely periodic

### 4. Random Forest
**Implementation**: Scikit-learn `RandomForestRegressor`

**Configuration**:
- `n_estimators=200`
- `random_state=42`

**Features Used**:
- Dropped: `Datum`, `Umsatz_umsatz`, `umsatz_rolling7`, `temperature_2m_mean`
- One-hot encoding for categorical features
- All remaining numerical and binary features

**Feature Importance** (Top 5):
1. `Warengruppe_umsatz`: Dominant predictor
2. `day_of_week`: 0.136
3. `sunshine_hours`: High importance
4. `Temperatur_weather`: Significant
5. `school_holiday`: Notable impact

**Performance**:
- **R² Train**: 0.9825 ✅ (excellent fit)
- **R² Validation**: 0.7537 (moderate generalization)
- **MAPE Validation**: 0.2137

**Analysis**: High training accuracy suggests some overfitting; validation performance acceptable

### 5. XGBoost Regressor
**Implementation**: `XGBRegressor` from xgboost library

**Configuration**:
- `n_estimators`: Variable (100-200 in experiments)
- `learning_rate`: 0.001 - 0.01
- `random_state=42`

**Features Used**:
- One-hot encoded product groups
- One-hot encoded days of week
- Weather features
- Holiday indicators

**Feature Importance** (Top features):
- `Warengruppe_umsatz_2.0`: 0.641 (most important)
- `Warengruppe_umsatz_5.0`: 0.192
- `Warengruppe_umsatz_3.0`: 0.021
- Other product groups, temporal, and weather features

**Performance**:
- **R² Train**: 0.891 - 0.920
- **R² Validation**: 0.867 - 0.870
- **MAPE Validation**: 0.214

**Experiment Tracking**: Hyperparameters and results logged in `experiment_results` DataFrame

## Evaluation

### Metrics Used
- **R² Score**: Coefficient of determination (higher is better, 1.0 is perfect)
- **MAPE** (Mean Absolute Percentage Error): Average percentage error (lower is better)
- **MAE** (Mean Absolute Error): Average absolute error in sales units

### Model Comparison

| Model | R² Train | R² Validation | MAPE Validation | Notes |
|-------|----------|---------------|-----------------|-------|
| **Linear Regression** | 0.8945 | 0.8495 | 0.2258 | Good baseline |
| **Polynomial Regression** | 0.8945 | 0.8475 | 0.2262 | Similar to linear |
| **Cosinor Regression** | 0.0004 | -0.0110 | 0.8336 | ❌ Not suitable alone |
| **Random Forest** | 0.9825 | 0.7537 | 0.2137 | Overfitting evident |
| **XGBoost** | 0.920 | 0.870 | 0.214 | **Best balance** ✅ |

### Best Baseline Model: XGBoost
- Highest validation R² (0.870)
- Competitive MAPE (0.214)
- Good balance between bias and variance
- Feature importance analysis provides interpretability

## Feature Selection

### Key Features Across Models
1. **Product Group** (`Warengruppe_umsatz`): Most important predictor in all models
2. **Day of Week**: Captures weekly sales patterns
3. **Weather**: Temperature and sunshine positively correlated with sales
4. **Holidays**: School holidays and public holidays affect sales
5. **Month/Season**: Monthly patterns captured through month features
6. **is_last_day_of_year**: Handles last day year exceptional sales

### Dropped Features
- `Datum`: Used for splitting but not as feature
- `umsatz_rolling7`: Avoided to prevent data leakage
- `temperature_2m_mean`: Redundant with `Temperatur_weather`
- `Wettercode_weather`: Too many missing values

## Key Insights

1. **Product Group Dominance**: `Warengruppe_umsatz` is by far the most important feature (especially groups 2 and 5)
2. **Temporal Patterns**: Day of week and month significantly influence sales
3. **Weather Impact**: Temperature and sunshine positively affect sales; cloud coverage negatively affects
4. **Holiday Effects**: School holidays have stronger impact than public holidays
5. **Model Complexity**: XGBoost provides best validation performance; simple cosinor regression insufficient
6. **Overfitting Risk**: Random Forest shows signs of overfitting (R² train >> R² validation)
7. **Generalization**: Linear models generalize well but have lower ceiling; tree-based models achieve higher accuracy

## Visualizations

- **Predicted vs. Actual Plots**: Scatter plots comparing predictions to ground truth
- **Feature Importance Plots**: Bar charts showing relative importance of features
- **Residual Analysis**: Distribution of prediction errors
- **Experiment Tracking Tables**: Comparison of different feature combinations and hyperparameters

## Usage

To reproduce the baseline models:
1. Ensure cleaned datasets exist in `data/processed/` directory
2. Run notebooks in any order (they are independent)
3. Review `experiment_results` DataFrames for performance comparisons
4. Selected best model (XGBoost) for comparison with advanced models

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn (LinearRegression, PolynomialFeatures, RandomForestRegressor, metrics)
- xgboost (XGBRegressor)

## Next Steps

- Use XGBoost performance (R² ~0.87, MAPE ~0.214) as benchmark
- Implement neural network models to potentially improve upon baseline
- Explore more advanced feature engineering
- Consider ensemble methods combining multiple baseline models