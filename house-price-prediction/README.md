# House Prices Prediction Project

This project predicts house prices using Kaggle's [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) dataset. It includes an ML pipeline (`src/`) and a Next.js web app (`app/`) for interactive predictions.

## Project Structure
```
house-price-prediction/
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── prediction.py
├── tests/
│   └── test_preprocessing.py
├── app/
│   ├── backend/
│   │   ├── main.py
│   │   └── requirements.txt
│   ├── src/
│   │   ├── app/
│   │   │   ├── globals.css
│   │   │   ├── layout.tsx
│   │   │   └── page.tsx
│   │   ├── components/
│   │   │   ├── PredictionForm.tsx
│   │   │   ├── PredictionChart.tsx
│   │   │   ├── WalletConnect.tsx
│   │   │   └── ui/
│   │   ├── lib/
│   │   │   ├── api.ts
│   │   │   └── web3-config.ts
│   │   ├── store/
│   │   │   └── usePredictionStore.ts
│   │   ├── types/
│   │   │   └── index.ts
│   ├── package.json
│   ├── next.config.js
│   ├── tailwind.config.js
│   ├── tsconfig.json
│   └── public/
├── notebooks/
│   └── 01_house_price_eda_and_modeling.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup
1. Clone or download the `ML projects` repository from GitHub: `https://github.com/yourusername/ML-projects`.
2. Navigate to `house-price-prediction/`.
3. Place `train.csv` and `test.csv` in `house-price-prediction/data/raw/` (download from Kaggle).
4. Install Python dependencies:
   ```bash
   pip install -r house-price-prediction/requirements.txt
   ```
5. For the web app backend, install dependencies:
   ```bash
   pip install -r house-price-prediction/app/backend/requirements.txt
   ```
6. For the web app frontend, navigate to `house-price-prediction/app/` and install Node.js dependencies:
   ```bash
   npm install
   ```

## Running the ML Pipeline
1. Preprocess data:
   ```bash
   python house-price-prediction/src/data_preprocessing.py
   ```
2. Train model:
   ```bash
   python house-price-prediction/src/model_training.py
   ```
3. Generate predictions:
   ```bash
   python house-price-prediction/src/prediction.py
   ```

## Running the Web App
1. Start the backend:
   ```bash
   cd house-price-prediction/app/backend
   uvicorn main:app --reload
   ```
2. Start the frontend:
   ```bash
   cd house-price-prediction/app
   npm run dev
   ```
3. Access the app at `http://localhost:3000`.

## Uploading to GitHub (Web Interface)
1. Go to your `ML projects` repository (e.g., `https://github.com/yourusername/ML-projects`).
2. If `house-price-prediction/` doesn’t exist:
   - Click **“Add file”** > **“Create new file”**.
   - Name: `house-price-prediction/dummy.txt`.
   - Content: `Temporary file for folder creation`.
   - Commit message: `Created house-price-prediction folder`.
   - Click **“Commit new file”**.
3. Upload project files:
   - Navigate to `house-price-prediction/`.
   - Click **“Add file”** > **“Upload files”**.
   - From your computer (`C:\Users\shukl\Vivek Study DS\Git Project\house-price-prediction`), select files in batches:
     - Root: `requirements.txt`, `.gitignore`, `README.md`.
     - `src/`: `data_preprocessing.py`, `model_training.py`, `prediction.py`.
     - `tests/`: `test_preprocessing.py`.
     - `notebooks/`: `01_house_price_eda_and_modeling.ipynb`.
     - `app/`: `backend/main.py`, `backend/requirements.txt`, `src/app/globals.css`, `src/app/layout.tsx`, `src/app/page.tsx`, `src/components/PredictionForm.tsx`, `src/components/PredictionChart.tsx`, `src/components/WalletConnect.tsx`, `src/lib/api.ts`, `src/lib/web3-config.ts`, `src/store/usePredictionStore.ts`, `src/types/index.ts`, `package.json`, `next.config.js`, `tailwind.config.js`, `tsconfig.json`.
   - Prepend paths (e.g., `house-price-prediction/src/data_preprocessing.py`, `house-price-prediction/app/backend/main.py`).
   - Commit with messages like `Uploaded src files`, `Uploaded app files`.
   - Click **“Commit changes”** for each batch.
4. Exclude files listed in `house-price-prediction/.gitignore`:
   - `data/raw/*`, `models/*`, `data/processed/*.pkl`, `app/node_modules/`, `app/.next/`, `app/.env*`, `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`, `*.log`, `*.tmp`, `.DS_Store`.
5. Optionally, delete `dummy.txt`:
   - Go to `house-price-prediction/dummy.txt`.
   - Click the trash icon.
   - Commit message: `Removed placeholder file`.
   - Click **“Commit changes”**.

## Notes
- **Excluded Files**: Large files (`train.csv`, `test.csv`, models) are excluded by `.gitignore`. Download them from Kaggle and place in `house-price-prediction/data/raw/`.
- **Repository Link**: Share this project via `https://github.com/yourusername/ML-projects/tree/main/house-price-prediction`.
- **Testing**: Download the repo as a ZIP (click “Code” > “Download ZIP”), place data files, and run scripts to verify functionality.
- **Updates**: To update files, navigate to `house-price-prediction/`, click “Add file” > “Upload files”, or edit existing files, and click “Commit changes”.