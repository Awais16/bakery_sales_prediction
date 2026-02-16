# Model Definition and Evaluation

## Overview

This stage implements and evaluates a Neural Network model to improve upon baseline models for bakery sales forecasting. The deep learning approach uses feed-forward neural networks with careful feature engineering and preprocessing to achieve competitive performance on the validation set.

## Implementation

### Notebooks
- [**`01_NN_baseline_best.ipynb`**](01_NN_baseline_best.ipynb): Initial neural network baseline with MSE loss
- [**`01_NN_best_clean.ipynb`**](01_NN_best_clean.ipynb): Final optimized neural network model with MAPE loss (production version)

## Model Selection

### Chosen Architecture: Feed-Forward Neural Network

**Rationale**:
- Captures non-linear relationships between features and sales
- Handles complex interactions between product groups, weather, and temporal patterns
- Outperforms linear baselines while avoiding overfitting
- Suitable for tabular regression tasks with moderate feature count

**Framework**: TensorFlow/Keras
- Industry-standard deep learning library
- Easy to prototype and iterate
- Good support for preprocessing pipelines
- Efficient training on CPU/GPU

## Model Architecture

### Network Structure
```
Input Layer (n_features)
    ↓
Dense(64 units, activation='relu')
    ↓
Dense(32 units, activation='relu')
    ↓  
Dense(1 unit, activation='linear')  # Output layer
```

### Layer Details
- **Input Shape**: Variable (depends on one-hot encoding)
  - Approximately 40-50 features after encoding
- **Hidden Layer 1**: 64 neurons, ReLU activation
  - Captures complex feature interactions
- **Hidden Layer 2**: 32 neurons, ReLU activation
  - Further abstraction and dimensionality reduction
- **Output Layer**: 1 neuron, linear activation
  - Regression output (predicted sales)

### Why This Architecture?
- **Shallow network** (2 hidden layers): Sufficient for tabular data, reduces overfitting risk
- **Decreasing width** (64→32): Funnel architecture for feature compression
- **ReLU activation**: Prevents vanishing gradients, computationally efficient
- **No dropout**: Small network size doesn't require aggressive regularization
- **Linear output**: Standard for regression tasks

## Feature Engineering

### Features Used
**Numerical Features** (StandardScaler applied):
- `Temperatur_weather`: Temperature in °C
- `Bewoelkung_weather`: Cloud coverage
- `Windgeschwindigkeit_weather`: Wind speed
- `sunshine_hours`: Sunshine duration
- `KielerWoche_kiwo`: Binary indicator for local event

**Categorical Features** (OneHotEncoder applied):
- `Warengruppe_umsatz`: Product group (1-6) → 6 binary features
- `day_of_week`: Day of week (0-6) → 7 binary features
- `month`: Month of year (1-12) → 12 binary features

**Binary Features** (no scaling):
- `is_saturday`: Weekend indicator
- `is_sunday`: Weekend indicator
- `public_holiday`: Public holiday indicator
- `school_holiday`: School holiday indicator

### Features Dropped
- **`Datum`**: Date column (used for splitting, not as feature)
- **`Umsatz_umsatz`**: Target variable
- **`umsatz_rolling7`**: Avoided to prevent data leakage
- **`temperature_2m_mean`**: Redundant with `Temperatur_weather`

### Preprocessing Pipeline
**ColumnTransformer** with two parallel transformations:
1. **StandardScaler**: Numerical features normalized (mean=0, std=1)
2. **OneHotEncoder**: Categorical features converted to binary vectors

**Reason**: Neural networks perform better with normalized inputs and explicit categorical encoding

## Hyperparameter Tuning

### Optimizer Configuration
- **Optimizer**: Adam (Adaptive Moment Estimation)
  - Learning rate: 0.001 (baseline) or 0.0005 (final model)
  - Default β1=0.9, β2=0.999
- **Why Adam?**: Adaptive learning rates, robust to hyperparameter choices, fast convergence

### Loss Functions
- **Baseline Model**: MSE (Mean Squared Error)
  - Standard regression loss
  - Penalizes large errors heavily
- **Final Model**: MAPE (Mean Absolute Percentage Error)
  - Direct optimization of evaluation metric
  - Treats all errors proportionally

### Training Configuration
- **Batch Size**: 32
  - Balance between training speed and gradient stability
- **Epochs**: 200 (maximum)
  - Early stopping typically halts around 20-40 epochs
- **Validation Split**: Separate validation set (2017-08-01 to 2018-07-31)

### Regularization Techniques
- **Early Stopping**:
  - Monitor: `val_loss`
  - Patience: 10 epochs
  - Restore best weights: ✅
  - Prevents overfitting by stopping when validation loss stops improving

## Implementation Details

### Training Process
1. **Data Loading**: Read cleaned CSV files from `data/processed/`
2. **Feature Engineering**: Add `month` column from date
3. **Train-Validation Split**: Use pre-split temporal datasets
4. **Preprocessing**: Apply ColumnTransformer (scaling + encoding)
5. **Model Building**: Instantiate Sequential model with 3 layers
6. **Compilation**: Set optimizer, loss, and metrics
7. **Training**: Fit with early stopping callback
8. **Evaluation**: Predict on validation set and calculate metrics
9. **Prediction**: Generate test set predictions for submission

### Code Structure
```python
# Preprocessing
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(), categorical_features)
])

# Model
model = keras.Sequential([
    keras.layers.Input(shape=(X_train_preprocessed.shape[1],)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1)
])

# Compilation
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
    loss="mape",
    metrics=["mae"]
)

# Training with early stopping
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)
```

## Evaluation Metrics

### Metrics Used
- **R² Score**: Proportion of variance explained (higher is better, max=1.0)
- **RMSE** (Root Mean Squared Error): Standard deviation of residuals (lower is better)
- **MAPE** (Mean Absolute Percentage Error): Average percentage error (lower is better)
- **MAE** (Mean Absolute Error): Average absolute error in sales units

### Model Performance

#### Baseline Neural Network Model (`01_NN_baseline_best.ipynb`)
- **Loss Function**: MSE
- **Learning Rate**: 0.001
- **Performance**:
  - **Validation RMSE**: 42.83
  - **Validation R²**: 0.8916 ✅
  - **Training**: Converged with early stopping

**Analysis**: Strong performance, comparable to XGBoost baseline (R²=0.87)

#### Final Neural Network Model (`01_NN_best_clean.ipynb`)
- **Loss Function**: MAPE (direct optimization of evaluation metric)
- **Learning Rate**: 0.0005
- **Performance**:
  - **Validation MAPE**: 17.52% ✅
  - **Validation RMSE**: 49.82
  - **Validation R²**: 0.8533
  - **Predictions**: Mean=193.81, Min=27.20, Max=679.02

**Analysis**: 
- MAPE of 17.52% is competitive with baseline models
- Slightly lower R² than baseline NN, but better calibrated for percentage errors
- Production model used for final test predictions

### Comparison with Baseline Models

| Model | R² Val | MAPE Val | RMSE Val | Notes |
|-------|--------|----------|----------|-------|
| Linear Regression | 0.8495 | 0.2258 | N/A | Simple baseline |
| XGBoost | 0.8700 | 0.214 | N/A | **Best baseline** |
| NN (MSE loss) | 0.8916 | N/A | 42.83 | Strong performance |
| **NN (MAPE loss)** | **0.8533** | **0.1752** | **49.82** | **Production model** ✅ |

**Key Findings**:
- Neural network with MSE loss achieves highest R² (0.8916)
- MAPE-optimized NN achieves lowest percentage error (17.52%)
- Trade-off between R² and MAPE optimization
- Neural networks competitive with gradient boosting

## Key Insights

1. **Deep Learning Viability**: Neural networks achieve competitive performance on tabular sales data
2. **Architecture Simplicity**: Shallow networks (2 hidden layers) sufficient; deeper networks not necessary
3. **Loss Function Impact**: Direct MAPE optimization improves percentage error at slight R² cost
4. **Feature Importance**: Product group (Warengruppe) remains dominant, similar to baseline models
5. **Preprocessing Critical**: StandardScaler and OneHotEncoder essential for NN performance
6. **Early Stopping**: Prevents overfitting; models converge in 20-40 epochs typically
7. **Generalization**: NNs generalize well; no significant overfitting observed
8. **Production Readiness**: Final model produces reasonable predictions (no extreme outliers)

## Visualizations

- **Loss Curves**: Training vs. validation loss over epochs
  - Monitors overfitting and convergence
- **Predicted vs. Actual**: Scatter plot comparing predictions to ground truth
  - Diagonal line indicates perfect predictions
  - Most points cluster near diagonal
- **Residual Distributions**: Histogram of prediction errors
  - Analysis of error patterns

## Predictions

### Test Set Predictions
- **Output File**: `data/processed/submission_nn_best.csv`
- **Format**: `id`, `umsatz` (predicted sales)
- **Rows**: 1,830 predictions (one per test sample)
- **Statistics**:
  - Mean prediction: 193.81
  - Min prediction: 27.20
  - Max prediction: 679.02
- **Validation**: Predictions in reasonable range, no extreme outliers

## Usage

To reproduce the neural network model:
1. Ensure cleaned datasets exist in `data/processed/` directory
2. Run `01_NN_best_clean.ipynb` for final production model
3. Or run `01_NN_baseline_best.ipynb` for MSE-optimized model
4. Check `data/processed/submission_nn_best.csv` for test predictions

## Dependencies

- pandas
- numpy
- tensorflow (Keras API)
- scikit-learn (preprocessing, metrics)
- matplotlib (visualizations)

## Next Steps & Potential Improvements

1. **Hyperparameter Tuning**: Grid search over learning rates, batch sizes, network widths
2. **Deeper Networks**: Experiment with 3-4 hidden layers
3. **Regularization**: Add dropout layers to reduce overfitting risk
4. **Ensemble Methods**: Combine NN with XGBoost predictions
5. **Feature Engineering**: Interaction terms, lag features (careful with leakage)
6. **Cross-Validation**: Time series cross-validation for robust evaluation
7. **Advanced Architectures**: LSTM/GRU for sequence modeling, Transformer-based models