# Fertilizer Recommendation using XGBoost and MAP@3 Evaluation

This repository contains a machine learning pipeline to predict the top 3 most suitable fertilizers for a given agricultural setup based on environmental and soil parameters. The model uses `XGBoostClassifier` and is evaluated using **Mean Average Precision at 3 (MAP@3)** score.

## 📌 Problem Statement

The goal is to predict the **top 3 fertilizers** for given input features like temperature, humidity, soil type, crop type, and NPK values. The evaluation metric is `MAP@3` — which checks if the actual fertilizer is present in the top 3 predicted results.

---

## 📂 Dataset

- **train.csv** – contains features and the target fertilizer name.
- **test.csv** – contains input data where predictions need to be made.

### Features used:
- **Numerical**: Temperature, Humidity, Moisture, Nitrogen, Phosphorus, Potassium  
- **Categorical**: Soil Type, Crop Type  
- **Target**: Fertilizer Name

---

## ⚙️ Workflow

1. **Data Loading and Cleaning**
   - Removed the `id` column.
   - Encoded the target label using `LabelEncoder`.

2. **Feature Engineering**
   - Numerical features scaled with `StandardScaler`.
   - Categorical features one-hot encoded using `OneHotEncoder`.

3. **Model**
   - Used `XGBoostClassifier` with:
     - `n_estimators = 300`
     - `max_depth = 9`
     - `learning_rate = 0.3`
   - Integrated preprocessing and model into a single `Pipeline`.

4. **Evaluation**
   - Predicted class probabilities and sorted top 3 predictions.
   - Evaluated using custom `MAP@3` function.

5. **Prediction on Test Set**
   - Applied the trained pipeline to the test data.
   - Exported top 3 predicted fertilizer names into a single string.

---

## 🧠 Model Evaluation

The model outputs a probability distribution over all fertilizer classes. Top-3 predictions are extracted, and the MAP@3 score is calculated on the validation set.

```python
MAP@3 Score: 0.3257  
