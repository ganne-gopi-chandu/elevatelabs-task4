# Task 4: Classification with Logistic Regression

This project demonstrates binary classification using logistic regression. We use the **Breast Cancer Wisconsin Diagnostic Dataset** to classify tumors as **Benign (0)** or **Malignant (1)**.

---

## 📦 Dataset Details

- **Source:** Breast Cancer Wisconsin Diagnostic Dataset  
- **Target Column:** `diagnosis`  
  - `M` → 1 (Malignant)  
  - `B` → 0 (Benign)  
- **Features:** 30 numerical features (e.g., radius_mean, texture_mean, etc.)

---

## 🔍 Step-by-Step Explanation

### ✅ `data.py` – Data Preprocessing Script

This script performs **data cleaning and preprocessing** in the following steps:

1. **Load the Dataset**:  
   Loads `data.csv` which contains tumor records with clinical features and diagnosis.

2. **Drop Irrelevant Columns**:  
   - `id`: Identifier column, not useful for prediction.  
   - `Unnamed: 32`: An empty column with only NaN values.

3. **Encode Target Variable**:  
   - Maps the `diagnosis` column to numeric values:
     - `B` (Benign) → `0`
     - `M` (Malignant) → `1`

4. **Handle Missing Values**:  
   - Replaces any missing values in the dataset with the **mean** of their respective columns.

5. **Save Cleaned Data**:  
   - Writes the cleaned dataset to `processed_data.csv` for later use in modeling.

> ✅ **Output**: `processed_data.csv`

---

### 🔁 `process.py` – Model Training and Evaluation Script

This script builds and evaluates the **logistic regression classifier**:

1. **Load Preprocessed Data**:  
   Loads `processed_data.csv` created by `data.py`.

2. **Split into Features and Target**:  
   - `X` = all feature columns (input data)  
   - `y` = `diagnosis` column (output label: 0 or 1)

3. **Train-Test Split**:  
   Splits data into 80% training and 20% testing.

4. **Standardize Features**:  
   Uses `StandardScaler` to normalize the feature values (zero mean, unit variance).

5. **Train Logistic Regression Model**:  
   Trains the model using training data.

6. **Make Predictions**:
   - `y_pred`: Predicted labels for the test data.
   - `y_prob`: Predicted probabilities for each class (used for ROC curve).

7. **Evaluate the Model**:
   - **Accuracy**
   - **Confusion Matrix**
   - **Precision**
   - **Recall**
   - **ROC-AUC Score**

8. **Plot ROC Curve**:
   - Generates the **ROC Curve** as `roc_curve.png`.

---

## 🔧 How to Run the Project

> Ensure you have Python installed and `data.csv` present in your working directory.

## Dependices
- Python
- `pandas`, `numpy`
- `matplotlib`
- `scikit-learn`
