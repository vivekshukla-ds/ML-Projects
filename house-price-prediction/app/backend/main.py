import os
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict

# Base path
BASE_PATH = r"C:\Users\shukl\Vivek Study DS\Git Project\house-price-prediction"
MODEL_PATH = os.path.join(BASE_PATH, "models")

# Load preprocessors and model
try:
    num_imputer = joblib.load(os.path.join(MODEL_PATH, "num_imputer.joblib"))
    cat_imputer = joblib.load(os.path.join(MODEL_PATH, "cat_imputer.joblib"))
    encoder = joblib.load(os.path.join(MODEL_PATH, "encoder.joblib"))
    scaler = joblib.load(os.path.join(MODEL_PATH, "scaler.joblib"))
    outlier_limits = joblib.load(os.path.join(MODEL_PATH, "outlier_limits.joblib"))
    model = joblib.load(os.path.join(MODEL_PATH, "best_model.joblib"))
except Exception as e:
    raise Exception(f"Error loading model/preprocessors: {e}")

app = FastAPI()

# Define input schema
class HouseInput(BaseModel):
    data: Dict[str, float | str | int]

@app.post("/predict")
async def predict(input_data: HouseInput):
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([input_data.data])

        # Identify features
        numeric_features = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
        categorical_features = [col for col in df.columns if df[col].dtype == 'object']

        # Handle skewness
        high_skew_features = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
                              '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 
                              'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']
        for col in high_skew_features:
            if col in df.columns:
                df[col] = np.log1p(df[col])

        # Outlier capping
        for feature, (lower, upper) in outlier_limits.items():
            if feature in df.columns:
                df[feature] = np.clip(df[feature], lower, upper)

        # Impute missing
        if numeric_features:
            df[numeric_features] = num_imputer.transform(df[numeric_features])
        if categorical_features:
            df[categorical_features] = cat_imputer.transform(df[categorical_features])

        # Rare labels
        rare_thresh = 0.01
        for col in categorical_features:
            freq = df[col].value_counts(normalize=True)
            rare_labels = freq[freq < rare_thresh].index
            df[col] = df[col].replace(rare_labels, 'Other')

        # Ordinal mapping
        qual_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
        qual_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
                     'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
        for col in qual_cols:
            if col in df.columns:
                df[col] = df[col].map(qual_map).fillna(0)
                if col in categorical_features:
                    categorical_features.remove(col)
                    numeric_features.append(col)

        # Encoding
        if categorical_features:
            encoded_data = encoder.transform(df[categorical_features])
            encoded_cols = encoder.get_feature_names_out(categorical_features)
            encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=df.index)
            df = df.drop(columns=categorical_features)
            df = pd.concat([df, encoded_df], axis=1)

        # Feature Engineering
        df['HouseAge'] = df['YrSold'] - df['YearBuilt']
        df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
        df['SoldMonth'] = df['MoSold']
        df['TotalBath'] = (df['BsmtFullBath'] + 0.5*df['BsmtHalfBath'] +
                           df['FullBath'] + 0.5*df['HalfBath'])
        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
        df['TotalPorch'] = (df['OpenPorchSF'] + df['EnclosedPorch'] +
                            df['3SsnPorch'] + df['ScreenPorch'])
        df['GrLivArea_per_Room'] = df['GrLivArea'] / (df['TotRmsAbvGrd'] + 1)
        df['GarageCars_per_Area'] = df['GarageCars'] / (df['GarageArea'] + 1)

        new_features = ['HouseAge', 'RemodAge', 'SoldMonth', 'TotalSF', 'TotalBath', 'TotalPorch',
                        'GrLivArea_per_Room', 'GarageCars_per_Area']
        numeric_features += new_features

        # Scaling
        df[numeric_features] = scaler.transform(df[numeric_features])

        # Align columns with X_train
        X_train = joblib.load(os.path.join(BASE_PATH, "data", "processed", "X_train.pkl"))
        for col in X_train.columns:
            if col not in df.columns:
                df[col] = 0
        df = df[X_train.columns]

        # Predict
        prediction = model.predict(df)
        return {"SalePrice": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")