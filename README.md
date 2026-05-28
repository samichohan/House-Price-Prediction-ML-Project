# 🏠 House Price Prediction — ML Project



> A machine learning web app that predicts house prices based on property features using a Linear Regression pipeline trained on real Indian housing data.



## 🚀 Live Demo

🔗 **[https://samichohan-house-price-prediction.hf.space](https://samichohan-house-price-prediction.hf.space)**



## 📊 Model Performance

| Model              | R² Score | MAE        | RMSE        |
|--------------------|----------|------------|-------------|
| Linear Regression  | 0.6495   | 979,680    | 1,331,071   |
| XGBoost            | 0.6456   | 995,440    | 1,338,339   |
| Random Forest      | 0.6182   | 1,011,168  | 1,389,249   |
| KNN                | 0.5721   | 1,080,846  | 1,470,708   |
| Decision Tree      | 0.4177   | 1,265,651  | 1,715,615   |


✅ **Best Model: Linear Regression (R² = 0.6495)**


## 🏗️ Project Structure



```
House-Price-Prediction-ML-Project/
├── app.py                  # Streamlit web app
├── requirements.txt        # Dependencies
├── data/
│   └── Housing.csv         # Dataset (545 houses)
├── model/
│   └── house_model.pkl     # Trained ML pipeline
└── notebook/
    └── house_price.ipynb   # Training notebook
```

## ✨ Features

- 🔮 Predict house prices instantly
- 📐 Input: area, bedrooms, bathrooms, stories, parking
- 🏡 Features: mainroad, guestroom, basement, AC, furnishing
- 💰 Output in ₹, Lakhs, and Crores
- ⚡ Cached model loading for fast predictions

## 🛠️ Tech Stack

- **Python 3.13** — core language
- **Streamlit** — web UI
- **scikit-learn 1.6.0** — ML pipeline (preprocessing + model)
- **XGBoost** — boosting model (compared)
- **pandas / numpy** — data handling
- **joblib** — model serialization

## ⚙️ ML Pipeline

```
Input Features
    │
    ▼
ColumnTransformer
    ├── StandardScaler       → [area, bedrooms, bathrooms, stories, parking]
    ├── OneHotEncoder        → [mainroad, guestroom, basement, AC, prefarea]
    └── OrdinalEncoder       → [furnishingstatus]
    │
    ▼
LinearRegression
    │
    ▼
Predicted Price (₹)
```

## 📦 Run Locally

```bash
# Clone the repo
git clone https://github.com/samichohan/House-Price-Prediction-ML-Project
cd House-Price-Prediction-ML-Project

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📁 Dataset

- **Source:** Kaggle — Housing Price Dataset
- **Size:** 545 rows × 13 columns
- **Target:** price (Indian Rupees)
- **Features:** 12 (numeric + categorical)

## 👤 Author

**Sami Chohan**
- GitHub → https://github.com/samichohan
- HuggingFace → https://huggingface.co/samichohan

## 📄 License

MIT License — free to use and modify.
