# House Price Prediction

## Project Overview

This project aims to predict house prices using machine learning techniques. The model is trained on historical house sale data and predicts the sales price of houses based on various features such as the size, location, year built, and more.

The dataset used in this project comes from a **Kaggle competition** and includes a variety of house-related attributes like square footage, neighborhood, and garage type.

## Files in the Project

- **train.csv**: This file contains the historical house sale data, which includes the house features and the actual sale prices (used to train the model).
- **test.csv**: This file contains similar features as the training data but without the sale prices. The task is to predict these missing sale prices.
- **submission.csv**: After predictions are made, the results are saved in this file and formatted according to Kaggle's submission guidelines.

## Key Features in the Dataset

Some of the important features in the dataset include:

- **LotArea**: Lot size in square feet.
- **OverallQual**: Overall material and finish quality.
- **YearBuilt**: Original construction date.
- **TotalBsmtSF**: Total square feet of basement area.
- **GrLivArea**: Above-ground living area square feet.
- **GarageCars**: Size of garage in car capacity.
- **FullBath**: Full bathrooms above grade.

## Model and Techniques Used

1. **Data Preprocessing**:
   - Handling missing values using strategies like mean imputation for numerical features and 'None' for categorical features.
   - One-hot encoding for categorical variables.
   - Logarithmic transformation on the `SalePrice` target to handle skewed distributions.

2. **Machine Learning Models**:
   - **Linear Regression**: A basic regression model used for initial experimentation.
   - **Random Forest**: An ensemble model that handles non-linearity better.
   - **Stacking Regressor**: A more advanced ensemble technique that combines multiple models to improve performance.
   
3. **Stacking Ensemble**:
   - A combination of different models (base estimators) was used to improve prediction accuracy.
   - Logarithmic transformation was applied to `SalePrice`, and after prediction, the exponent was taken to get back to the original scale.

4. **Model Evaluation**:
   - Models were evaluated using metrics like **Mean Absolute Error (MAE)**, **Mean Squared Error (MSE)**, and **R2 Score**.
   - The final model achieved a **R2 score** of 0.91 on the test data, indicating strong predictive performance.

## Pipeline Used

A pipeline was created to preprocess the data and apply the model in one step:
```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline(steps=[
    ('preprocessing', preprocessing_transformer),  # Preprocessing steps (handling missing data, scaling, etc.)
    ('model', model)  # Final prediction model (e.g., RandomForest, Stacking Regressor)
])
