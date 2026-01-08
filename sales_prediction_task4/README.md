# 📊 **Sales Prediction Using Machine Learning**

## **Introduction**

Sales prediction involves forecasting the amount of a product that customers will purchase, taking into account various factors such as advertising expenditure, target audience segmentation, and advertising platform selection.

In businesses that offer products or services, the role of a Data Scientist is crucial for predicting future sales. They utilize machine learning techniques in Python to analyze and interpret data, allowing them to make informed decisions regarding advertising costs. By leveraging these predictions, businesses can optimize their advertising strategies and maximize sales potential. Let's embark on the journey of sales prediction using machine learning in Python.

---

## **Project Overview**

This project aims to build a machine learning model to predict product sales based on advertising budgets across three different media channels: **TV**, **Radio**, and **Newspaper**. The project follows a complete data science pipeline from data exploration to model deployment.

---

## **Table of Contents**

1. [Project Structure](#project-structure)
2. [Dataset Description](#dataset-description)
3. [Installation & Setup](#installation--setup)
4. [Code Implementation](#code-implementation)
5. [Methodology](#methodology)
6. [Results & Output](#results--output)
7. [Key Insights](#key-insights)
8. [Business Applications](#business-applications)
9. [Future Improvements](#future-improvements)

---

## **Project Structure**

```
sales_prediction_project/
│
├── sales_prediction.py              # Main Python script
├── advertising.csv                  # Dataset file
├── sales_predictions.csv            # Generated predictions
├── advertising_analysis.png         # EDA visualizations
├── pairplot.png                    # Pairplot visualization
├── model_performance.png           # Model comparison charts
├── feature_importance.png          # Feature importance plot
└── README.md                       # Documentation
```

---

## **Dataset Description**

The dataset contains 200 observations with advertising budgets and corresponding sales:

| Feature | Description | Type | Range |
|---------|-------------|------|-------|
| TV | Advertising budget spent on TV (in thousands) | float64 | 0.7 - 296.4 |
| Radio | Advertising budget spent on Radio | float64 | 0.0 - 49.6 |
| Newspaper | Advertising budget spent on Newspaper | float64 | 0.3 - 114.0 |
| Sales | Product sales (in thousands) | float64 | 1.6 - 27.0 |

**Statistical Summary:**
```
               TV       Radio   Newspaper       Sales
count  200.000000  200.000000  200.000000  200.000000
mean   147.042500   23.264000   30.554000   15.130500
std     85.854236   14.846809   21.778621    5.283892
min      0.700000    0.000000    0.300000    1.600000
25%     74.375000    9.975000   12.750000   11.000000
50%    149.750000   22.900000   25.750000   16.000000
75%    218.825000   36.525000   45.100000   19.050000
max    296.400000   49.600000  114.000000   27.000000
```

---

## **Installation & Setup**

### **Requirements**
```bash
Python 3.7+
pandas
numpy
matplotlib
seaborn
scikit-learn
```

### **Installation**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### **Running the Project**
```bash
python sales_prediction.py
```

---

## **Code Implementation**

### **Step 1: Import Libraries**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')
```

### **Step 2: Load and Prepare Data**
```python
def load_advertising_data(csv_file='advertising.csv'):
    # Load dataset from CSV or create from provided data
    # Handle missing files and data formatting
    # Return cleaned DataFrame
```

### **Step 3: Exploratory Data Analysis (EDA)**
```python
def perform_eda(df):
    # Display dataset information
    # Show statistical summaries
    # Check for missing values and duplicates
    # Return analyzed DataFrame
```

### **Step 4: Data Visualization**
```python
def visualize_data(df):
    # Create comprehensive visualizations:
    # 1. Distribution plots
    # 2. Correlation heatmap
    # 3. Scatter plots (TV vs Sales, Radio vs Sales, Newspaper vs Sales)
    # 4. Box plots for budget distribution
    # Save visualizations to files
```

### **Step 5: Data Preparation for Modeling**
```python
def prepare_data(df):
    # Separate features and target variable
    # Split data into training and testing sets (80-20 split)
    # Apply feature scaling using StandardScaler
    # Return scaled data and scaler object
```

### **Step 6: Model Training**
```python
def train_models(X_train, X_test, y_train, y_test):
    # Train three different models:
    # 1. Linear Regression
    # 2. Random Forest Regressor
    # 3. Gradient Boosting Regressor
    # Evaluate each model's performance
    # Return results dictionary
```

### **Step 7: Model Evaluation**
```python
def evaluate_models(results, y_test):
    # Compare model performances
    # Visualize actual vs predicted values
    # Create residual plots
    # Display R² and RMSE comparisons
    # Identify best performing model
```

### **Step 8: Making Predictions**
```python
def make_predictions(model, scaler):
    # Use trained model to predict sales for new advertising budgets
    # Example: Predict for three different budget scenarios
    # Save predictions to CSV file
```

### **Step 9: Feature Importance Analysis**
```python
def analyze_feature_importance(model, feature_names):
    # Analyze which features contribute most to predictions
    # Display feature importance/coefficients
    # Create visualization of feature importance
```

---

## **Methodology**

### **1. Data Collection & Preparation**
- Load advertising dataset
- Handle missing values and duplicates
- Data cleaning and preprocessing

### **2. Exploratory Data Analysis**
- Statistical analysis
- Correlation analysis
- Data visualization
- Pattern identification

### **3. Feature Engineering**
- Feature selection (TV, Radio, Newspaper)
- Feature scaling
- Train-test split

### **4. Model Development**
- Implement multiple regression algorithms
- Hyperparameter tuning
- Cross-validation

### **5. Model Evaluation**
- Performance metrics calculation
- Model comparison
- Best model selection

### **6. Prediction & Deployment**
- Make predictions on new data
- Generate insights and recommendations
- Save model outputs

---

## **Results & Output**

### **Console Output Example:**
```
======================================================================
🎯 SALES PREDICTION USING MACHINE LEARNING
======================================================================
📊 Task: Predict sales based on advertising budgets
📈 Features: TV, Radio, Newspaper advertising budgets
🎯 Target: Sales
======================================================================
✅ Data loaded successfully from CSV file!
📊 Dataset Shape: (200, 4)
📋 Columns: ['TV', 'Radio', 'Newspaper', 'Sales']

📊 MODEL COMPARISON:
----------------------------------------
               Model   Test R²  Test RMSE       MAE  CV Score
2  Gradient Boosting  0.958729   1.129300  0.841132  0.922955
1      Random Forest  0.953502   1.198682  0.917150  0.932195
0  Linear Regression  0.905901   1.705215  1.274826  0.887974

🏆 BEST MODEL: Gradient Boosting
   Test R²: 0.9587
   Test RMSE: 1.1293

🎯 Predicted Sales:
----------------------------------------
    TV  Radio  Newspaper  Predicted_Sales
0  300     40         70        23.484459
1  150     30         40        15.293615
2   50     20         10        10.568007
```

### **Generated Files:**
1. **`advertising_analysis.png`** - Comprehensive EDA visualizations
2. **`pairplot.png`** - Pairwise relationships between features
3. **`model_performance.png`** - Model comparison charts
4. **`feature_importance.png`** - Feature importance visualization
5. **`sales_predictions.csv`** - Predictions for new budgets

### **Feature Importance:**
```
🔍 Feature Importance:
----------------------------------------
     Feature  Importance
0         TV    0.852783
1      Radio    0.135441
2  Newspaper    0.011776
```
