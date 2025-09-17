import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from scipy.stats import randint, uniform
import joblib
import shap
from plotly.subplots import make_subplots
import sys

# Base path
BASE_PATH = r"C:\Users\shukl\Vivek Study DS\Git Project\house-price-prediction"
OUTPUT_PATH = os.path.join(BASE_PATH, "data", "processed")
MODEL_PATH = os.path.join(BASE_PATH, "models")

# Load processed data
X_train = joblib.load(os.path.join(OUTPUT_PATH, "X_train.pkl"))
y_train = joblib.load(os.path.join(OUTPUT_PATH, "y_train.pkl"))

# Model Definition
rf_model = RandomForestRegressor(n_estimators=500, max_depth=12, min_samples_split=5, random_state=42, n_jobs=-1)
xgb_model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, subsample=0.8,
                         colsample_bytree=0.8, random_state=42, n_jobs=-1)
lgb_model = LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=-1, subsample=0.8,
                          colsample_bytree=0.8, random_state=42, verbose=-1)
gbr_model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, random_state=42)
cat_model = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=4, random_state=42, verbose=0)
stack_model = StackingRegressor(
    estimators=[
        ('rf', rf_model),
        ('xgb', xgb_model),
        ('gbr', gbr_model),
        ('lgb', lgb_model),
        ('catb', cat_model)
    ],
    final_estimator=LinearRegression(), n_jobs=-1)

# Training and Evaluation
def cv_rmse(model, X, y, cv=5):
    scores = np.sqrt(-cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error'))
    return scores

models = {
    'RandomForest': rf_model,
    'XGBoost': xgb_model,
    'LightGBM': lgb_model,
    'GradientBoosting': gbr_model,
    'CatBoost' : cat_model,    
    'Stacking': stack_model}

results = []
best_model = None
best_model_name = None
best_rmse = float("inf")

for name, model in models.items():
    print(f"{name} Model Training ...", end='', flush=True)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    
    rmse = np.sqrt(mean_squared_error(y_train, y_pred))
    mae = mean_absolute_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)
    
    std = cv_rmse(model, X_train, y_train).std()
    mean_cv_rmse = cv_rmse(model, X_train, y_train).mean()
    
    results.append({
        'Model': name,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Std RMSE': std,
        'Mean CV RMSE': mean_cv_rmse})
    
    if mean_cv_rmse < best_rmse:
        best_rmse = mean_cv_rmse
        best_model = model
        best_model_name = name
    
    sys.stdout.write('\r' + ' ' * 120 + '\r')
    print(f"{name} Model Trained.\n{name}_training Result with metrics: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.2f}, Std={std:.2f}, Mean CV RMSE={mean_cv_rmse:.2f}\n")

results_df = pd.DataFrame(results)

def highlight_best_row(row):
    if row['Model'] == best_model_name:
        return ['background-color: #004d40; color:white; font-weight:bold']*len(row)
    else:
        return ['']*len(row)

def highlight_best_metric(row):
    style = ['']*len(row)
    if row['Model'] == best_model_name:
        col_index = row.index.get_loc('Mean CV RMSE')
        style[col_index] = 'background-color: darkred; color:white; font-weight:bold'
    return style

styled_table = results_df.style.apply(highlight_best_row, axis=1)\
                               .apply(highlight_best_metric, axis=1)

print(' ')
print("                   Model Comparison Table with Best Highlight ")
print("="*80)
display(styled_table)

# Plot comparison
best_color = '#00BFC4'
other_color = '#B0BEC5'

colors = [best_color if m == best_model_name else other_color for m in results_df['Model']]
results_df['Color'] = colors

fig = px.bar(results_df, x='Model', y='Mean CV RMSE',
             text='Mean CV RMSE',
             color='Color',
             hover_data=['RMSE', 'MAE', 'R2', 'Std RMSE', 'Mean CV RMSE'],
             color_discrete_map="identity",
             title="Model Comparison (CV RMSE)",
             width=1100, height=500)

fig.update_traces(texttemplate='%{text:.2f}', textposition='outside', marker_line_width=0)
fig.update_layout(hoverlabel=dict(font_size=12, font_color='white', bgcolor='#333333', font_family="Arial"),
                  yaxis_title="CV RMSE (Lower is Better)", xaxis_title="Models", showlegend=False,
                  template="plotly_white", uniformtext_minsize=12, uniformtext_mode='hide', bargap=0.3)
fig.show()

print(f"\n Best model: {best_model_name} with CV RMSE = {best_rmse:.2f}")
print('-'*47)
print(' ')

# Train and save best model
if best_model is not None and best_model_name is not None:
    print(f"\nBest model selected: {best_model_name} with Mean CV RMSE={best_rmse:.2f}")
    best_model.fit(X_train, y_train)
    model_file = os.path.join(MODEL_PATH, "best_model.joblib")
    joblib.dump(best_model, model_file)
    print(f"{best_model_name} model saved to {model_file}")
    loaded_model = joblib.load(model_file)
    print(f"Verified: Loaded model type is {type(loaded_model).__name__}")

# Feature Importance
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    feat_names = X_train.columns
    fi_df = pd.DataFrame({"Feature": feat_names, "Importance": importances}).sort_values(by="Importance", ascending=False).head(20)
    min_imp = fi_df['Importance'].min()
    max_imp = fi_df['Importance'].max()
    fi_df['Importance_norm'] = (fi_df['Importance'] - min_imp) / (max_imp - min_imp + 1e-6)
    fi_df['Color'] = fi_df['Importance_norm'].apply(lambda x: f'rgba(0,191,196,{0.3 + 0.7*x:.2f})')
    
    fig = px.bar(
        fi_df[::-1], x='Importance', y='Feature', orientation='h', text='Importance',
        color='Color', color_discrete_map="identity",
        title=f'Top 20 Feature Importances ({best_model_name})', width=1100, height=500)
    
    fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig.update_layout(xaxis_title="Importance", yaxis_title="Feature", template="plotly_white")
    fig.show()
else:
    print(f"{best_model_name} does not support feature importances.")

# Hyperparameter Tuning
rf_params = {'n_estimators': randint(100, 500),'max_depth': randint(3, 15),
    'min_samples_split': randint(2, 10),'min_samples_leaf': randint(1, 5)}

xgb_params = {'n_estimators': randint(100, 500),'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.1),'subsample': uniform(0.6, 0.4)}

lgb_params = {'n_estimators': randint(100, 500),'num_leaves': randint(20, 50),'learning_rate': uniform(0.01, 0.1),
              'subsample': uniform(0.6, 0.4)}

gbr_params = {'n_estimators': randint(100, 500),'max_depth': randint(3, 10),'learning_rate': uniform(0.01, 0.1),
              'subsample': uniform(0.6, 0.4)}

cat_params = {'iterations': randint(200, 500),'depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.1),'l2_leaf_reg': randint(1, 10)}

n_iter_search = 20
cv = 5
scoring_metric = 'neg_root_mean_squared_error'

models_random = {
    'RandomForest': (RandomForestRegressor(random_state=42), rf_params),
    'XGBoost': (XGBRegressor(random_state=42, verbosity=0), xgb_params),
    'LightGBM': (LGBMRegressor(random_state=42), lgb_params),
    'GradientBoosting': (GradientBoostingRegressor(random_state=42), gbr_params),
    'CatBoost': (CatBoostRegressor(random_state=42, verbose=0), cat_params)}

best_models_random = {}
for name, (model, param) in models_random.items():
    print(f"\rRunning RandomizedSearchCV for {name} ...")
    rand_search = RandomizedSearchCV(
        estimator=model, param_distributions=param, n_iter=n_iter_search,
        cv=cv, scoring=scoring_metric, verbose=0, n_jobs=-1, random_state=42)
    rand_search.fit(X_train, y_train)
    best_models_random[name] = rand_search.best_estimator_
    print(f"Completed RandomizedSearch for: {name} | Best Params: {rand_search.best_params_}\n")

# SHAP Plots
shap.initjs()

for name, model in best_models_random.items():
    print(f"\nTraining & Explaining Model: {name}")
    model.fit(X_train, y_train)
    try:
        if "XGBoost" in name or "LightGBM" in name:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_train)
    except Exception as e:
        print(f"Could not generate SHAP values for {name}: {e}")
        continue
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    if hasattr(model, "feature_importances_"):
        feature_importance = model.feature_importances_
    elif name == "CatBoost":
        feature_importance = model.get_feature_importance()
    else:
        feature_importance = None

    if feature_importance is not None:
        importance_df = pd.DataFrame({"Feature": X_train.columns, 
            "Importance": feature_importance}).sort_values(by="Importance", ascending=False).head(15)
        
        axes[0].barh(importance_df["Feature"], importance_df["Importance"], color='teal')
        axes[0].set_title(f"{name} - Top 15 Feature Importances")
        axes[0].set_xlabel("Feature Importance")
        axes[0].set_ylabel("Feature")
        axes[0].invert_yaxis()

    shap_importance = np.abs(shap_values.values).mean(axis=0)
    shap_df = pd.DataFrame({"Feature": X_train.columns, 
        "SHAP": shap_importance}).sort_values(by="SHAP", ascending=False).head(15)

    axes[1].barh(shap_df["Feature"], shap_df["SHAP"], color='purple')
    axes[1].set_title(f"{name} - Top 15 SHAP Importances")
    axes[1].set_xlabel("Mean Absolute SHAP Value")
    axes[1].set_ylabel("Feature")
    axes[1].invert_yaxis()

    top_features = np.argsort(np.sum(np.abs(shap_values.values), axis=0))[-15:]
    feature_names = X_train.columns[top_features]

    shap_values_top = shap_values.values[:, top_features]
    X_train_top = X_train.iloc[:, top_features]

    for i in range(len(feature_names)):
        feature_data = X_train_top.iloc[:, i]
        shap_data = shap_values_top[:, i]
        colors = feature_data
        axes[2].scatter(shap_data, [i] * len(shap_data), c=colors, cmap='viridis', alpha=0.5, s=20, edgecolor='k', linewidth=0.5)

    axes[2].set_yticks(np.arange(len(feature_names)))
    axes[2].set_yticklabels(feature_names)
    axes[2].set_title(f"{name} - SHAP Beeswarm Plot")
    axes[2].set_xlabel("SHAP Value")
    axes[2].set_ylabel("Feature")
    axes[2].set_ylim(-1, len(feature_names))

    sm = plt.cm.ScalarMappable(cmap='viridis')
    sm.set_array(X_train_top.values)
    cbar = fig.colorbar(sm, ax=axes[2])
    cbar.set_label('Feature Value', rotation=270, labelpad=15)

    plt.tight_layout()
    plt.show()

# Stacking SHAP
try:
    stack_model.fit(X_train, y_train)
    base_model_preds = np.hstack([est.fit(X_train, y_train).predict(X_train).reshape(-1, 1) for est in stack_model.estimators_])
    feature_names = [f"Pred_{est[0]}" for est in stack_model.estimators]
    explainer_stack = shap.Explainer(stack_model.final_estimator_, base_model_preds)
    shap_values_stack = explainer_stack(base_model_preds)

    fig_stack = make_subplots(rows=1, cols=1, subplot_titles=("Stacking Meta-Model - Top SHAP Features",))

    shap_importance_stack = np.abs(shap_values_stack.values).mean(axis=0)
    shap_stack_df = pd.DataFrame({"Feature": feature_names, "SHAP": shap_importance_stack}).sort_values(
        by="SHAP", ascending=False)
    
    fig_stack.add_trace(go.Bar(x=shap_stack_df["SHAP"], y=shap_stack_df["Feature"], orientation="h",
                               marker=dict(color=shap_stack_df["SHAP"], colorscale="tealrose")),
                        row=1, col=1)
    
    fig_stack.update_yaxes(autorange="reversed", row=1, col=1)
    fig_stack.update_layout(height=300, width=1000, font=dict(size=14), margin=dict(l=100, r=20, t=50, b=50))
    fig_stack.show()
except Exception as e:
    print(f"Could not generate SHAP for Stacking Regressor: {e}")