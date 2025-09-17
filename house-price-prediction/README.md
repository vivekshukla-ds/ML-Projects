# House Prices Prediction Project

This project predicts house prices using Kaggle's [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) dataset. It includes an ML pipeline and a Next.js web app for predictions.

## Setup
1. Clone or download the `ML projects` repository.
2. Navigate to `house-price-prediction/`.
3. Place `train.csv` and `test.csv` in `house-price-prediction/data/raw/` (download from Kaggle).
4. Install Python dependencies: `pip install -r house-price-prediction/requirements.txt`.
5. For the web app, install backend dependencies: `pip install -r house-price-prediction/app/backend/requirements.txt`.
6. For the frontend, navigate to `house-price-prediction/app/` and run `npm install`.

## Running the ML Pipeline
1. Preprocess data: `python house-price-prediction/src/data_preprocessing.py`
2. Train model: `python house-price-prediction/src/model_training.py`
3. Generate predictions: `python house-price-prediction/src/prediction.py`

## Running the Web App
1. Start the backend: Navigate to `house-price-prediction/app/backend/` and run `uvicorn main:app --reload`.
2. Start the frontend: Navigate to `house-price-prediction/app/` and run `npm run dev`.
3. Access at `http://localhost:3000`.

## Kaggle Submission
1. Run the ML pipeline to generate `house-price-prediction/data/processed/submission.csv`:
   - `python house-price-prediction/src/data_preprocessing.py`
   - `python house-price-prediction/src/model_training.py`
   - `python house-price-prediction/src/prediction.py`
2. Visit [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).
3. Upload `house-price-prediction/data/processed/submission.csv`, add a description (e.g., “Stacking Ensemble Submission”).
4. Verify format: 1459 rows with columns `Id`, `SalePrice`.

## Uploading to GitHub (Web Interface)
1. Go to your `ML projects` repository (e.g., `https://github.com/yourusername/ML-projects`).
2. If `house-price-prediction/` doesn’t exist:
   - Click “Add file” > “Create new file”.
   - Name: `house-price-prediction/dummy.txt`, content: `Temporary file`, commit with “Created house-price-prediction folder”.
3. Upload files:
   - Navigate to `house-price-prediction/`.
   - Click “Add file” > “Upload files”.
   - Select files from your computer (e.g., `requirements.txt`, `src/data_preprocessing.py`).
   - Prepend paths (e.g., `house-price-prediction/src/data_preprocessing.py`).
   - Commit with messages like “Uploaded src files”.
4. Exclude files listed in `house-price-prediction/.gitignore` (`data/raw/*`, `models/*`, etc.).
5. Optionally, delete `dummy.txt` after uploading (click trash icon, commit).

## Notes
- Large files (`train.csv`, `test.csv`, models) are excluded by `.gitignore`. Download them from Kaggle and place in `house-price-prediction/data/raw/`.
- Link this project in Kaggle submissions: `https://github.com/yourusername/ML-projects/tree/main/house-price-prediction`.
