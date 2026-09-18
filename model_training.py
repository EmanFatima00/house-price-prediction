"""
Pakistan House Price Analysis & Prediction — Advanced ML Pipeline
Author: Eman Fatima | BS-AI @ PAF-IAST
Dataset: Zameen.com Pakistani Real Estate Data (168K+ listings)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings, os, json, joblib

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "zameen_clean.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("  PAKISTAN HOUSE PRICE PREDICTION — ADVANCED ML PIPELINE")
print("=" * 60)

# ── 1. Load ───────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"\n✅ Dataset loaded: {len(df):,} listings")
print(f"   Cities: {df['city'].nunique()} | Property types: {df['property_type'].nunique()}")

# ── 2. Feature Engineering ────────────────────────────────────
df['date_added']    = pd.to_datetime(df['date_added'], errors='coerce')
df['house_age']     = datetime.now().year - df['date_added'].dt.year
df['house_age']     = df['house_age'].fillna(df['house_age'].median()).clip(0, 50)
df['baths_per_bed'] = df['baths'] / (df['bedrooms'] + 1)
df['total_rooms']   = df['bedrooms'] + df['baths']
df['is_for_sale']   = (df['purpose'] == 'For Sale').astype(int)
df['log_area']      = np.log1p(df['Area Size'])
df['log_price']     = np.log1p(df['price'])

le_city  = LabelEncoder()
le_ptype = LabelEncoder()
le_prov  = LabelEncoder()
df['city_enc']     = le_city.fit_transform(df['city'].astype(str))
df['ptype_enc']    = le_ptype.fit_transform(df['property_type'].astype(str))
df['province_enc'] = le_prov.fit_transform(df['province_name'].astype(str))
print("✅ Feature engineering complete")

# ── 3. Prepare ────────────────────────────────────────────────
features = ['Area Size','log_area','bedrooms','baths','total_rooms',
            'baths_per_bed','house_age','is_for_sale',
            'latitude','longitude','city_enc','ptype_enc','province_enc']

X = df[features].fillna(0)
y = df['log_price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print(f"✅ Train: {len(X_train):,} | Test: {len(X_test):,}")

# ── 4. Train Models ───────────────────────────────────────────
models = {
    'Linear Regression': (LinearRegression(),     True),
    'Ridge Regression':  (Ridge(alpha=10),         True),
    'Random Forest':     (RandomForestRegressor(
                            n_estimators=100, max_depth=10,
                            random_state=42, n_jobs=-1),  False),
    'XGBoost':           (XGBRegressor(
                            n_estimators=100, max_depth=6,
                            learning_rate=0.1, random_state=42,
                            verbosity=0),                 False),
}

print("\n" + "=" * 60)
print("  MODEL COMPARISON")
print("=" * 60)

results = {}
for name, (model, use_sc) in models.items():
    Xtr = X_train_sc if use_sc else X_train
    Xte = X_test_sc  if use_sc else X_test
    print(f"\n⏳ Training {name}...")
    model.fit(Xtr, y_train)
    preds_log    = model.predict(Xte)
    preds_actual = np.expm1(preds_log)
    y_actual     = np.expm1(y_test)
    r2   = r2_score(y_actual, preds_actual)
    mae  = mean_absolute_error(y_actual, preds_actual)
    rmse = np.sqrt(mean_squared_error(y_actual, preds_actual))
    mape = np.mean(np.abs((y_actual - preds_actual) / (y_actual + 1))) * 100
    results[name] = dict(model=model, r2=r2, mae=mae, rmse=rmse, mape=mape,
                         preds=preds_actual, actuals=y_actual, use_sc=use_sc)
    print(f"   R²   : {r2:.4f}")
    print(f"   MAE  : PKR {mae:,.0f}")
    print(f"   MAPE : {mape:.2f}%")

# ── 5. Best ───────────────────────────────────────────────────
best_name = max(results, key=lambda x: results[x]['r2'])
best = results[best_name]
print(f"\n🏆 Best Model: {best_name} | R²: {best['r2']:.4f} | MAE: PKR {best['mae']:,.0f}")

# ── 6. Save ───────────────────────────────────────────────────
joblib.dump(best['model'],  os.path.join(OUTPUT_DIR, "house_model.pkl"))
joblib.dump(scaler,         os.path.join(OUTPUT_DIR, "scaler.pkl"))
joblib.dump(le_city,        os.path.join(OUTPUT_DIR, "le_city.pkl"))
joblib.dump(le_ptype,       os.path.join(OUTPUT_DIR, "le_ptype.pkl"))
joblib.dump(le_prov,        os.path.join(OUTPUT_DIR, "le_prov.pkl"))

meta = {
    "best_model":      best_name,
    "r2":              round(best['r2'], 4),
    "mae":             round(best['mae'], 0),
    "mape":            round(best['mape'], 2),
    "rmse":            round(best['rmse'], 0),
    "total_listings":  len(df),
    "cities":          sorted(df['city'].unique().tolist()),
    "property_types":  sorted(df['property_type'].unique().tolist()),
    "provinces":       sorted(df['province_name'].unique().tolist()),
    "use_scaler":      best['use_sc'],
}
json.dump(meta, open(os.path.join(OUTPUT_DIR, "meta.json"), "w"))
print("✅ All files saved to outputs/")

# ── 7. Visualizations ─────────────────────────────────────────
print("\n📊 Generating visualizations...")

# A. Model Comparison
names_ = list(results.keys())
r2s_   = [results[n]['r2']       for n in names_]
maes_  = [results[n]['mae']/1e6  for n in names_]
mapes_ = [results[n]['mape']     for n in names_]
cols_  = ['#6366F1','#059669','#D97706','#DC2626']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Model Performance Comparison", fontsize=14, fontweight='bold')
for ax, vals, title in zip(axes,
        [r2s_, maes_, mapes_],
        ["R² Score","MAE (PKR Millions)","MAPE (%)"]):
    bars = ax.barh(names_, vals, color=cols_)
    ax.set_title(title, fontweight='bold')
    for i, v in enumerate(vals):
        ax.text(v*1.01, i, f"{v:.3f}", va='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,"model_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()

# B. Actual vs Predicted
idx = np.random.choice(len(best['actuals']), min(500, len(best['actuals'])), replace=False)
act = np.array(best['actuals'])[idx] / 1e6
pred = best['preds'][idx] / 1e6
plt.figure(figsize=(7, 6))
plt.scatter(act, pred, alpha=0.4, color='#6366F1', s=18)
lim = max(act.max(), pred.max()) * 1.05
plt.plot([0, lim],[0, lim],'r--', linewidth=2, label='Perfect prediction')
plt.title(f"Actual vs Predicted — {best_name}", fontsize=13, fontweight='bold')
plt.xlabel("Actual Price (PKR Millions)"); plt.ylabel("Predicted Price (PKR Millions)")
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,"actual_vs_predicted.png"), dpi=150, bbox_inches='tight')
plt.close()

# C. Price by City
top_cities = df.groupby('city')['price'].median().nlargest(8).index
city_data  = [df[df['city']==c]['price'].values/1e6 for c in top_cities]
plt.figure(figsize=(12, 5))
bp = plt.boxplot(city_data, labels=top_cities, patch_artist=True,
                 boxprops=dict(facecolor='#EEF2FF', color='#6366F1'),
                 medianprops=dict(color='#DC2626', linewidth=2),
                 flierprops=dict(marker='o', markersize=3, alpha=0.3))
plt.title("House Price Distribution by City", fontsize=13, fontweight='bold')
plt.ylabel("Price (PKR Millions)"); plt.xticks(rotation=20)
plt.grid(axis='y', alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,"price_by_city.png"), dpi=150, bbox_inches='tight')
plt.close()

# D. Feature Importance
if best_name in ['Random Forest','XGBoost']:
    fi = pd.Series(best['model'].feature_importances_, index=features).sort_values(ascending=True)
    plt.figure(figsize=(9, 6))
    fi.plot(kind='barh', color=['#6366F1' if v > fi.median() else '#CBD5E1' for v in fi])
    plt.title(f"Feature Importance — {best_name}", fontsize=13, fontweight='bold')
    plt.xlabel("Importance Score"); plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,"feature_importance.png"), dpi=150, bbox_inches='tight')
    plt.close()

# E. Correlation Heatmap
num_cols = ['price','Area Size','bedrooms','baths','house_age','total_rooms']
plt.figure(figsize=(8, 6))
mask = np.triu(np.ones_like(df[num_cols].corr(), dtype=bool))
sns.heatmap(df[num_cols].corr(), mask=mask, annot=True, fmt='.2f',
            cmap='coolwarm', center=0, linewidths=0.5, annot_kws={'size':9})
plt.title("Feature Correlation Heatmap", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,"correlation_heatmap.png"), dpi=150, bbox_inches='tight')
plt.close()

# F. Price by Property Type
pt_med = df.groupby('property_type')['price'].median().sort_values(ascending=False).head(8)
plt.figure(figsize=(11, 4))
pt_med.plot(kind='bar', color='#6366F1', edgecolor='white')
plt.title("Median Price by Property Type", fontsize=13, fontweight='bold')
plt.ylabel("Median Price (PKR)"); plt.xticks(rotation=30, ha='right')
plt.grid(axis='y', alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,"price_by_type.png"), dpi=150, bbox_inches='tight')
plt.close()

print("✅ All visualizations saved!")
print("\n" + "=" * 60)
print("  PIPELINE COMPLETE")
print(f"  Best Model : {best_name}")
print(f"  R² Score   : {best['r2']:.4f}")
print(f"  MAE        : PKR {best['mae']:,.0f}")
print(f"  MAPE       : {best['mape']:.2f}%")
print("=" * 60)
