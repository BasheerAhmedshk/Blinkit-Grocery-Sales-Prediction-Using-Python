# Blinkit Grocery Sales Prediction 🛒📊

This project aims to predict sales for grocery items listed on Blinkit using machine learning and Tableau. The analysis identifies key features driving sales and compares various regression models for best performance.

## 📌 Objective
To accurately predict `Item_Outlet_Sales` and gain business insights from sales-related variables using both Python (ML) and Tableau (BI).

---

## 🧰 Tools & Technologies
- Python (Pandas, Scikit-learn, XGBoost, LightGBM, Seaborn)
- Tableau (for dashboarding)  -> upcoming
- Jupyter Notebook
- Excel for raw data

---

## 🔍 Key Steps

1. **Data Cleaning & Preprocessing**
   - Handling missing values (`Item_Weight`, `Outlet_Size`)
   - Encoding categorical features
   - Feature engineering (`Units_Sold_Est`, etc.)

2. **Exploratory Data Analysis (EDA)**
   - Correlation analysis
   - Sales distribution plots
   - Outlet profiling

3. **Model Building**
   - Trained: Linear Regression, Random Forest, XGBoost, Gradient Boosting, LightGBM
   - Evaluated using: R², RMSE, MAE, Cross-validation

4. **Model Evaluation**
   - LightGBM selected as best model with:
     - R²: 0.9985
     - RMSE: 63.78
     - MAE: 32.52

5. **Tableau Dashboard**

---

## 📈 Results
- LightGBM outperformed others with nearly 99.9% accuracy.
- `Item_MRP` and `Units_Sold_Est` were most impactful features.

---

