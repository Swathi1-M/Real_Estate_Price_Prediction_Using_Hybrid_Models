# Real Estate Price Prediction using Hybrid Models

This project predicts the sale price of houses using the **Ames Housing Dataset**. It combines the power of **XGBoost**, **ANN**, and **Linear Regression** to make accurate predictions.

---

## Dataset

- **Source:** Ames Housing Dataset (via Kaggle or openml)
- **Target Variable:** `SalePrice`
- **Key Features Used:**
  - `GrLivArea` (Above ground living area)
  - `OverallQual` (Overall quality rating)
  - `YearBuilt` (Year of construction)
  - `GarageCars` (Garage capacity)
  - `TotalBsmtSF` (Total basement area)

---

## Models Used

- XGBoost Regressor
- Artificial Neural Network (ANN)
- Linear Regression (Meta-model)

---

## Evaluation Metrics

- **R² Score**: 0.88 – Indicates 88% of variance in price is explained by the model.
- **RMSE**: ~$30,600 – The average prediction error in dollars.

---

## Deployment

The project is deployed using **Streamlit**, allowing users to enter property features and receive real-time price predictions.

To run the app:

```bash
streamlit run streamlit_app.py
