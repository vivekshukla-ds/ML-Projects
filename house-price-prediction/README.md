House Prices Prediction Project
This project predicts house prices using Kaggle's House Prices - Advanced Regression Techniques dataset. It includes an ML pipeline (src/) and a Next.js web app (app/) for predictions.
Setup

Clone or download the ML projects repository from GitHub.
Navigate to house-price-prediction/.
Place train.csv and test.csv in house-price-prediction/data/raw/ (download from Kaggle).
Install Python dependencies:pip install -r house-price-prediction/requirements.txt


For the web app backend, install dependencies:pip install -r house-price-prediction/app/backend/requirements.txt


For the web app frontend, navigate to house-price-prediction/app/ and install Node.js dependencies:npm install



Running the ML Pipeline

Preprocess data:python house-price-prediction/src/data_preprocessing.py


Train model:python house-price-prediction/src/model_training.py


Generate predictions:python house-price-prediction/src/prediction.py



Running the Web App

Start the backend:cd house-price-prediction/app/backend
uvicorn main:app --reload


Start the frontend:cd house-price-prediction/app
npm run dev


Access the app at http://localhost:3000.

Uploading to GitHub (Web Interface)

Go to your ML projects repository (e.g., https://github.com/yourusername/ML-projects).
If house-price-prediction/ doesn’t exist:
Click “Add file” > “Create new file”.
Name: house-price-prediction/dummy.txt.
Content: Temporary file for folder creation.
Commit message: Created house-price-prediction folder.
Click “Commit new file”.


Upload project files:
Navigate to house-price-prediction/.
Click “Add file” > “Upload files”.
From your computer (C:\Users\shukl\Vivek Study DS\Git Project\house-price-prediction), select files:
Root: requirements.txt, .gitignore, README.md.
src/: data_preprocessing.py, model_training.py, prediction.py.
tests/: test_preprocessing.py.
notebooks/: 01_house_price_eda_and_modeling.ipynb.
app/: backend/main.py, backend/requirements.txt, src/app/globals.css, src/app/layout.tsx, src/app/page.tsx, src/components/PredictionForm.tsx, src/components/PredictionChart.tsx, src/components/WalletConnect.tsx, src/lib/api.ts, src/lib/web3-config.ts, src/store/usePredictionStore.ts, src/types/index.ts, package.json, next.config.js, tailwind.config.js, tsconfig.json.


Prepend paths (e.g., house-price-prediction/src/data_preprocessing.py, house-price-prediction/app/backend/main.py).
Commit with messages like Uploaded src files, Uploaded app files.
Click “Commit changes” for each batch.


Exclude files listed in house-price-prediction/.gitignore:
data/raw/*, models/*, data/processed/*.pkl, app/node_modules/, app/.next/, app/.env*, etc.


Optionally, delete dummy.txt:
Go to house-price-prediction/dummy.txt.
Click the trash icon, commit with message: Removed placeholder file.
Click “Commit changes”.



Notes

Excluded Files: Large files (train.csv, test.csv, models) are excluded by .gitignore. Download them from Kaggle and place in house-price-prediction/data/raw/.
Repository Link: Share this project via https://github.com/yourusername/ML-projects/tree/main/house-price-prediction.
Testing: Download the repo as a ZIP, place data files, and run scripts to verify functionality.
