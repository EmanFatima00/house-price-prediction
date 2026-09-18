# Pakistan House Price Prediction System

> XGBoost regression model trained on 50,000 Zameen.com listings — R²: 0.8785
> Built by Eman Fatima | BS-AI @ PAF-IAST

![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green?style=flat-square)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.5-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## Live Demo
**[Try the live app](YOUR_STREAMLIT_URL_HERE)**

---

## Project Overview

End-to-end regression pipeline for predicting property prices in Pakistan's major cities
using real data from Zameen.com. The dataset contains 168,000+ listings — we trained
on a cleaned sample of 50,000 across 5 cities. The project compares 4 ML models and
deploys the best-performing one as a Streamlit web application.

**Key finding:** Linear and Ridge regression performed poorly (R² = -153, -136),
confirming that property pricing in Pakistan follows highly non-linear patterns.
XGBoost handled these non-linearities effectively with R² = 0.8785.

---

## Results

| Model              | R² Score   | MAE (PKR)  | MAPE     |
|--------------------|------------|------------|----------|
| Linear Regression  | -153.17    | 7,028,578  | 3656.29% |
| Ridge Regression   | -136.92    | 6,937,257  | 3461.80% |
| Random Forest      | 0.8648     | 2,661,320  | 48.66%   |
| **XGBoost**        | **0.8785** | **2,496,729** | **44.57%** |

**Final Model: XGBoost | R²: 0.8785 | MAE: PKR 2.5M**

---

## ML Pipeline

```
Raw Dataset (168,000+ Zameen.com listings)
        ↓
Data Cleaning (price > 0, valid coordinates, IQR outlier removal)
        ↓
Feature Engineering (7 new features)
        ↓
Label Encoding (city, property type, province)
        ↓
Log-transformation of target (price)
        ↓
Train/Test Split (80/20)
        ↓
4 Models Compared
        ↓
XGBoost selected → Deployed on Streamlit
```

---

## Features

- 4 ML models trained and compared
- 7 engineered features: house age, baths per bedroom, total rooms, log area, etc.
- Log-transformation applied to price (reduces skewness, improves regression)
- Interactive Streamlit app — enter property details, get instant price estimate
- Price range visualization (±15% confidence interval)
- 6 charts: model comparison, actual vs predicted, feature importance, city distribution

---

## Dataset

- **Source:** Zameen.com Pakistani Real Estate Dataset (Kaggle)
- **Original size:** 168,446 listings
- **Training sample:** 50,000 (after cleaning and sampling)
- **Cities:** Karachi, Lahore, Islamabad, Rawalpindi, Faisalabad
- **Property types:** Houses, Flats, Plots, Farmhouses, and more

---

## Run Locally

```bash
git clone https://github.com/EmanFatima00/house-price-prediction.git
cd house-price-prediction
pip install -r requirements.txt

# Train the model first
python model_training.py

# Then launch the app
streamlit run app.py
```

> Note: model_training.py reads from zameen_clean.csv (preprocessed sample).
> The full raw dataset (zameen-updated.csv) is also included for reference.

---

## Project Structure

```
house-price-prediction/
│
├── app.py                    # Streamlit web application
├── model_training.py         # Full ML pipeline
├── zameen_clean.csv          # Cleaned 50K sample (used for training)
├── zameen-updated.csv        # Full raw dataset
├── requirements.txt          # Dependencies
├── README.md                 # This file
│
└── outputs/
    ├── house_model.pkl           # Trained XGBoost model
    ├── scaler.pkl                # StandardScaler
    ├── le_city.pkl               # City label encoder
    ├── le_ptype.pkl              # Property type encoder
    ├── le_prov.pkl               # Province encoder
    ├── meta.json                 # Model metadata + city/type lists
    ├── model_comparison.png      # Model performance chart
    ├── actual_vs_predicted.png   # Regression scatter plot
    ├── feature_importance.png    # XGBoost feature importance
    ├── price_by_city.png         # Price distribution by city
    ├── correlation_heatmap.png   # Feature correlations
    └── price_by_type.png         # Median price by property type
```

---

## Author

**Eman Fatima**
BS Artificial Intelligence — Semester 6 | PAF-IAST, Pakistan

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/eman-fatima-99962230b)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat-square&logo=github)](https://github.com/EmanFatima00)

---

*Star this repo if you found it useful.*
