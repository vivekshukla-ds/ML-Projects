# House Prices Prediction Project

## Overview
This project predicts house prices using the Kaggle "House Prices: Advanced Regression Techniques" dataset. It includes a machine learning pipeline (data preprocessing, model training, prediction) and a Next.js web app for interactive predictions.

## Folder Structure
- `data/raw/`: `train.csv`, `test.csv`, `sample_submission.csv`.
- `data/processed/`: Processed data (`X_train.pkl`, `y_train.pkl`, `X_test.pkl`, `id_test.pkl`, `submission.csv`).
- `models/`: Saved models and preprocessors (`best_model.joblib`, `num_imputer.joblib`, `cat_imputer.joblib`, `encoder.joblib`, `scaler.joblib`, `outlier_limits.joblib`).
- `src/`: Python scripts for ML pipeline (`data_preprocessing.py`, `model_training.py`, `prediction.py`).
- `tests/`: Test preprocessing script (`test_preprocessing.py`).
- `app/`: Next.js app with FastAPI backend (`app/backend/main.py`).
- `notebooks/`: Jupyter notebook for EDA and modeling (`01_house_price_eda_and_modeling.ipynb`).

## ML Pipeline Setup
1. Place `train.csv`, `test.csv`, `sample_submission.csv` in `data/raw/`.
2. Install dependencies: `pip install -r requirements.txt`
3. Run scripts:
   - `python src/data_preprocessing.py` (processes data, saves to `data/processed/`)
   - `python src/model_training.py` (trains models, saves best to `models/`)
   - `python src/prediction.py` (generates `data/processed/submission.csv`)
4. For new test data: Update `new_test_path` in `tests/test_preprocessing.py`, run `python tests/test_preprocessing.py`, then `python src/prediction.py`.

## Web App Setup
1. Navigate to `app/`:
   ```bash
   cd app
   ```
2. Install Node.js dependencies:
   ```bash
   npm install
   ```
3. Run the Next.js app:
   ```bash
   npm run dev
   ```
4. Start the FastAPI backend:
   ```bash
   cd backend
   pip install -r requirements.txt
   uvicorn main:app --reload
   ```
5. Access the app at `http://localhost:3000`. Use the form (`PredictionForm.tsx`) to input house features and view predictions (`PredictionChart.tsx`).

## Notes
- **EDA**: Visualizations in `src/data_preprocessing.py` and `src/model_training.py`. Comment out to speed up execution.
- **Model**: Best model selected based on Mean CV RMSE.
- **Paths**: Scripts use `BASE_PATH = r"C:\Users\shukl\Vivek Study DS\Git Project\house-price-prediction"`. Update if your path differs.
- **Submission**: `submission.csv` must have 1459 rows with `Id` and `SalePrice` for Kaggle.
- **Web App**: Ensure backend (`app/backend/main.py`) is running for predictions. Frontend expects API at `http://localhost:8000/predict`.

## Dependencies
See `requirements.txt` for ML pipeline and `app/backend/requirements.txt` for backend.

## Kaggle Submission
1. Run ML pipeline scripts to generate `data/processed/submission.csv`.
2. Visit [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).
3. Upload `submission.csv`, add a description (e.g., “Stacking Ensemble”), and submit.
4. Verify format: 1459 rows, columns `Id`, `SalePrice`.

## Git Submission
1. Initialize Git: `git init`
2. Add files: `git add .` (`.gitignore` excludes large files)
3. Commit: `git commit -m "House Prices Prediction Project"`
4. Create GitHub repo, link it: `git remote add origin <repo-url>`
5. Push: `git push -u origin main`