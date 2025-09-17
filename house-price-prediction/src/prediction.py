import os
import pandas as pd
import joblib

# Base path
BASE_PATH = r"C:\Users\shukl\Vivek Study DS\Git Project\house-price-prediction"
OUTPUT_PATH = os.path.join(BASE_PATH, "data", "processed")
MODEL_PATH = os.path.join(BASE_PATH, "models")

# Load processed test data and model
X_test = joblib.load(os.path.join(OUTPUT_PATH, "X_test.pkl"))
id_test = joblib.load(os.path.join(OUTPUT_PATH, "id_test.pkl"))
best_model = joblib.load(os.path.join(MODEL_PATH, "best_model.joblib"))

# Predict
y_pred = best_model.predict(X_test)

# Prepare Submission
submission = pd.DataFrame({"Id": id_test, "SalePrice": y_pred})
submission_file = os.path.join(OUTPUT_PATH, "submission.csv")
submission.to_csv(submission_file, index=False)
print(f"Submission saved to {submission_file}")