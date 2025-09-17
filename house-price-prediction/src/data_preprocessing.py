import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib
import warnings
warnings.filterwarnings('ignore')

# Plot styling
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (8, 6)

# Base project path
BASE_PATH = r"C:\Users\shukl\Vivek Study DS\Git Project\house-price-prediction"
DATA_PATH = os.path.join(BASE_PATH, "data", "raw")
OUTPUT_PATH = os.path.join(BASE_PATH, "data", "processed")
MODEL_PATH = os.path.join(BASE_PATH, "models")

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# Load Train and Test Data
train = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
test = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))
pd.set_option('display.max_columns', None)

print("Train Data Shape:", train.shape)
print('-' * 30)
display(train.head(3))

print("Test Data Shape:", test.shape)
print('-' * 29)
display(test.head(3))

# Combine for preprocessing
train['dataset'] = 'train'
test['dataset'] = 'test'
full_data = pd.concat([train, test], sort=False).reset_index(drop=True)
print('Combining test and train data done')
print('')

full_data.head(3)

# Basic info
print("\nFull Data Info:")
full_data.info()

# Statistical summary
full_data.describe()

# SalePrice Distribution (Train Data)
plt.figure(figsize=(12, 4))
sns.histplot(train['SalePrice'], kde=True, bins=50, color=sns.color_palette("plasma", 1)[0])
plt.title("SalePrice Distribution", fontsize=16)
plt.xlabel("SalePrice")
plt.ylabel("Count")
plt.show()

# Missing Values
missing_percent = full_data.isnull().sum() / len(full_data) * 100
missing_percent = missing_percent[missing_percent > 0].sort_values(ascending=False)

plt.figure(figsize=(14, 5))
sns.barplot(x=missing_percent.index, y=missing_percent.values, palette="magma")
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.ylabel("Missing Values (%)", fontsize=14)
plt.title("Percentage of Missing Values by Column", fontsize=16)
plt.show()

missing_cols = full_data.columns[full_data.isnull().sum() > 0]
plt.figure(figsize=(14, 4))
sns.heatmap(full_data[missing_cols].isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap", fontsize=16)
plt.show()

# Numeric Features
numeric_features = full_data.select_dtypes(include=np.number).columns.tolist()
print('Total Number of Numeric features:', len(numeric_features))
print('--' * 17)
print(f'List of Numeric features:\n--------------------------\n{numeric_features}')

# Boxplots for Numeric Features
numeric_features = [col for col in numeric_features if col not in ['Id', 'SalePrice']]
sample_numeric = numeric_features[:12]
cols = 5
rows = (len(sample_numeric) + cols - 1) // cols
fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 3*rows))
axs = np.ravel(axs)

palette_colors = sns.color_palette("pastel", n_colors=len(sample_numeric))
for i, ax in enumerate(axs):
    if i >= len(sample_numeric):
        fig.delaxes(ax)
    else:
        col = sample_numeric[i]
        sns.boxplot(y=full_data[col], ax=ax, color=palette_colors[i], fliersize=6, linewidth=1.2)
        ax.set_facecolor("#f9f9f9")
        ax.set_title(f"{col} Distribution", fontsize=14, color="#333333")
        ax.set_xlabel("")
        ax.set_ylabel("")
plt.tight_layout()
plt.show()

# Correlation
train_rows = full_data['SalePrice'].notnull()
numeric_features = full_data.select_dtypes(include=np.number).columns.tolist()
numeric_features.remove('SalePrice')

corr_with_target = full_data.loc[train_rows, numeric_features + ['SalePrice']].corr()['SalePrice'].sort_values(ascending=False)
top_features = corr_with_target.drop('SalePrice').head(20).index.tolist()
print("Top correlated features:", top_features)

# Scatter Plots
cols = 4
rows = (len(top_features) + cols - 1) // cols
fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(18, 5*rows))
axs = axs.flatten()
palette = sns.color_palette("Set2", len(top_features))

for i, feature in enumerate(top_features):
    sns.scatterplot(
        x=full_data.loc[train_rows, feature],
        y=full_data.loc[train_rows, 'SalePrice'], ax=axs[i],
        color=palette[i], alpha=0.7, edgecolor='k'
    )
    axs[i].set_title(f"{feature} vs SalePrice", fontsize=14)
    axs[i].set_xlabel(feature, fontsize=12)
    axs[i].set_ylabel("SalePrice", fontsize=12)
    axs[i].grid(True, linestyle='--', alpha=0.5)

for j in range(i+1, len(axs)):
    fig.delaxes(axs[j])
plt.tight_layout()
plt.show()

# Plotly Correlation Heatmap
corr_matrix = full_data.loc[train_rows, numeric_features + ['SalePrice']].corr()
top_features = corr_matrix['SalePrice'].abs().sort_values(ascending=False).drop('SalePrice').head(20).index.tolist()
top_corr_matrix = corr_matrix.loc[top_features, top_features]

fig = px.imshow(
    top_corr_matrix,
    text_auto=".2f",
    color_continuous_scale='RdBu_r',
    origin='upper',
    aspect="auto",
    title="Top 20 Feature Correlation Heatmap (Train Data)",
    labels=dict(x="Features", y="Features", color="Correlation")
)
fig.update_layout(width=1000, height=800, title=dict(font=dict(size=20)), xaxis_tickangle=-45)
fig.show()

# Skewness
skew_values = full_data.loc[train_rows, numeric_features].skew().sort_values(ascending=False)
print("Skewness of numeric features (train data):")
print(skew_values)

high_skew = skew_values[abs(skew_values) > 0.75]
print("\nHighly skewed features:")
print(high_skew)

palette = sns.color_palette("Set3", len(high_skew))
cols = 4
rows = (len(high_skew) + cols - 1) // cols
plt.figure(figsize=(6*cols, 4*rows))

for i, feature in enumerate(high_skew.index):
    plt.subplot(rows, cols, i+1)
    sns.histplot(
        full_data.loc[train_rows, feature],
        kde=True, color=palette[i], bins=30
    )
    plt.title(f"{feature} (Skew: {high_skew[feature]:.2f})", fontsize=12, color=palette[i])
    plt.xlabel(feature)
    plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Categorical Features
categorical_features = full_data.select_dtypes(include=['object', 'category']).columns.tolist()
if 'dataset' in categorical_features:
    categorical_features.remove('dataset')
print(f"Total Categorical Features: {len(categorical_features)}")
print("Categorical Features:", categorical_features)

# Bar Plots for Categorical
categorical_features = [col for col in full_data.select_dtypes(include='object').columns if col != 'dataset']
cols = 3
rows = (len(categorical_features) + cols - 1) // cols
plt.figure(figsize=(6*cols, 4*rows))
palette = sns.color_palette("Set2", len(categorical_features))

for i, feature in enumerate(categorical_features):
    plt.subplot(rows, cols, i+1)
    sns.barplot(
        x=feature, y='SalePrice', data=full_data.loc[train_rows],
        ci=None, palette="pastel"
    )
    plt.title(f"{feature} vs SalePrice", fontsize=12)
    plt.xlabel(feature)
    plt.ylabel("Median SalePrice")
    plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Outlier Detection
numeric_features = full_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_features = [col for col in numeric_features if col not in ['Id', 'SalePrice']]

outlier_limits = {}
outlier_summary = []
palette = sns.color_palette("husl", len(numeric_features))

cols = 4
rows = (len(numeric_features) + cols - 1) // cols
plt.figure(figsize=(5 * cols, 3 * rows))

for i, feature in enumerate(numeric_features):
    Q1 = full_data.loc[train_rows, feature].quantile(0.25)
    Q3 = full_data.loc[train_rows, feature].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    outlier_limits[feature] = (lower, upper)
    outlier_count = full_data.loc[train_rows, feature].apply(lambda x: x < lower or x > upper).sum()
    outlier_summary.append([feature, outlier_count, lower, upper])
    
    plt.subplot(rows, cols, i+1)
    sns.boxplot(x=full_data.loc[train_rows, feature], color=palette[i])
    plt.title(f"{feature}\nOutliers: {outlier_count}", fontsize=10)
plt.tight_layout()
plt.show()

outlier_summary_df = pd.DataFrame(outlier_summary, columns=['Feature', 'Outlier Count', 'Lower Bound', 'Upper Bound'])
outlier_summary_df = outlier_summary_df.sort_values(by='Outlier Count', ascending=False)
print("\n         --- Outlier Detection Summary ---")
display(outlier_summary_df)

# Save outlier limits
joblib.dump(outlier_limits, os.path.join(MODEL_PATH, "outlier_limits.joblib"))

# Preprocessing
train_rows = full_data['SalePrice'].notnull()
test_rows = ~train_rows

# Save Ids for submission
id_train = full_data.loc[train_rows, 'Id']
id_test = full_data.loc[test_rows, 'Id']
full_data = full_data.drop(columns=['Id'])

numeric_features = full_data.select_dtypes(include=np.number).columns.tolist()
numeric_features.remove('SalePrice')
categorical_features = full_data.select_dtypes(include='object').columns.tolist()

# Handle skewness
skew_values = full_data.loc[train_rows, numeric_features].skew()
high_skew = skew_values[abs(skew_values) > 0.75]
full_data[high_skew.index] = np.log1p(full_data[high_skew.index])
skew_after = full_data.loc[train_rows, high_skew.index].skew()
print("\nSkewness after log1p transformation (train data):\n", skew_after)

# Outlier capping
for feature, (lower, upper) in outlier_limits.items():
    full_data[feature] = np.clip(full_data[feature], lower, upper)
print("Outlier capping complete for all numeric features!")

# Missing values
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='constant', fill_value='None')
num_imputer.fit(full_data.loc[train_rows, numeric_features])
cat_imputer.fit(full_data.loc[train_rows, categorical_features])
full_data[numeric_features] = num_imputer.transform(full_data[numeric_features])
full_data[categorical_features] = cat_imputer.transform(full_data[categorical_features])

joblib.dump(num_imputer, os.path.join(MODEL_PATH, "num_imputer.joblib"))
joblib.dump(cat_imputer, os.path.join(MODEL_PATH, "cat_imputer.joblib"))

# Rare label handling
rare_thresh = 0.01
for col in categorical_features:
    freq = full_data.loc[train_rows, col].value_counts(normalize=True)
    rare_labels = freq[freq < rare_thresh].index
    full_data[col] = full_data[col].replace(rare_labels, 'Other')

# Ordinal mapping
qual_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
qual_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
             'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
for col in qual_cols:
    if col in full_data.columns:
        full_data[col] = full_data[col].map(qual_map).fillna(0)
        if col in categorical_features:
            categorical_features.remove(col)
            numeric_features.append(col)

# One-hot encoding
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder.fit(full_data.loc[train_rows, categorical_features])
encoded_all = encoder.transform(full_data[categorical_features])
encoded_cols = encoder.get_feature_names_out(categorical_features)
encoded_df = pd.DataFrame(encoded_all, columns=encoded_cols, index=full_data.index)

joblib.dump(encoder, os.path.join(MODEL_PATH, "encoder.joblib"))

full_data = full_data.drop(columns=categorical_features)
full_data = pd.concat([full_data, encoded_df], axis=1)
print("Categorical features encoded. Shape:", full_data.shape)

# Feature Engineering
full_data['HouseAge'] = full_data['YrSold'] - full_data['YearBuilt']
full_data['RemodAge'] = full_data['YrSold'] - full_data['YearRemodAdd']
full_data['SoldMonth'] = full_data['MoSold']
full_data['TotalBath'] = (full_data['BsmtFullBath'] + 0.5*full_data['BsmtHalfBath'] +
                          full_data['FullBath'] + 0.5*full_data['HalfBath'])
full_data['TotalSF'] = full_data['TotalBsmtSF'] + full_data['1stFlrSF'] + full_data['2ndFlrSF']
full_data['TotalPorch'] = (full_data['OpenPorchSF'] + full_data['EnclosedPorch'] +
                           full_data['3SsnPorch'] + full_data['ScreenPorch'])
full_data['GrLivArea_per_Room'] = full_data['GrLivArea'] / (full_data['TotRmsAbvGrd'] + 1)
full_data['GarageCars_per_Area'] = full_data['GarageCars'] / (full_data['GarageArea'] + 1)

new_features = ['HouseAge', 'RemodAge', 'SoldMonth', 'TotalSF', 'TotalBath', 'TotalPorch',
                'GrLivArea_per_Room', 'GarageCars_per_Area']
print("List of features created in Feature Engineering:")
for f in new_features:
    print("-", f)

numeric_features += new_features

# Feature Scaling
scaler = StandardScaler()
scaler.fit(full_data.loc[train_rows, numeric_features])
full_data[numeric_features] = scaler.transform(full_data[numeric_features])

joblib.dump(scaler, os.path.join(MODEL_PATH, "scaler.joblib"))

# Separate train/test
X_train = full_data.loc[train_rows].drop(columns=['SalePrice', 'dataset'], errors='ignore')
y_train = full_data.loc[train_rows, 'SalePrice']
X_test = full_data.loc[test_rows].drop(columns=['SalePrice', 'dataset'], errors='ignore')

print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
print("Preprocessing Done!")

# Save processed data and ids
joblib.dump(X_train, os.path.join(OUTPUT_PATH, "X_train.pkl"))
joblib.dump(y_train, os.path.join(OUTPUT_PATH, "y_train.pkl"))
joblib.dump(X_test, os.path.join(OUTPUT_PATH, "X_test.pkl"))
joblib.dump(id_test, os.path.join(OUTPUT_PATH, "id_test.pkl"))

print("Processed data saved!")