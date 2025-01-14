# Car-Prices-ML-Project
Used exploratory data analysis, visualizations and preprocessing to effectively prepare raw data to be fed into different ML algorithms that predict the prices of different cars based on different input features. The accuracy scores are highlighted here.
 
Below is an overview of the project structure, methodology, and results.

---

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Modeling](#modeling)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Conclusion](#conclusion)
- [Dependencies](#dependencies)

---

## Introduction

The project focuses on building a predictive model for car prices based on input features such as engine type, horsepower, mileage, and more. By comparing different regression models, the goal is to identify the most accurate and reliable model using metrics such as R-squared, Mean Squared Error (MSE), and others.

---

## Dataset

The dataset, `cars_price.csv`, includes various attributes of cars and their corresponding prices. Key features include:

- Engine Type
- Horsepower
- Mileage
- Number of Doors
- Fuel Type

The dataset contains missing values, and wrong data types and requires cleaning before modeling.

---

## Data Preprocessing

Data preprocessing steps include:

1. **Handling Missing Values**: Columns with excessive missing values were filled using Simple Imputer. Numerical Columns were filled using the mean strategy whereas categorical columns were filled using the strategy "most_frequent".
2. **Encoding Categorical Variables**: Features such as `engine-type` and `fuel-type` were encoded using one-hot encoding.
3. **Feature Scaling**: Numeric features were scaled using standardization techniques like StandardScaler.
4. All these were done using ColumnTransformers and Pipeline to simplify and streamline the process.

---

## Exploratory Data Analysis (EDA)

EDA highlights the relationships between features and the target variable (car prices):

- Correlation heatmaps to identify significant predictors.
- Scatterplots and boxplots to visualize feature relationships.
- Distribution plots to check the normality of numerical features.

---

## Modeling

The following machine learning regression models were implemented:

1. **Linear Regression**
2. **Decision Tree Regressor**
3. **Random Forest Regressor**
4. **Gradient Boosting Regressor**
5. **K-Nearest Neighbors (KNN) Regressor**

Each model was trained and evaluated using a train-test split methodology, with hyperparameter tuning applied where necessary (e.g., GridSearchCV for Random Forest).

---

## Evaluation Metrics

Models were evaluated using the following metrics:

- **R-squared**: Proportion of variance explained by the model.
- **Mean Squared Error (MSE)**: Average squared difference between actual and predicted values.
---

## Results

The evaluation results of the models are summarized below:

- **Linear Regression**: R-squared = 0.79, MSE = 11304454.96
- **Decision Tree Regressor**: R-squared = 0.77, MSE = 12607820.17
- **Random Forest Regressor**: R-squared = 0.84, MSE = 8699198.93
- **Gradient Boosting Regressor**: R-squared = 0.856, MSE = 7844183.03
- **K-Nearest Neighbors (KNN) Regressor**: R-squared = 0.847, MSE = 8340663.25

Gradient Boosting Regressor achieved the best performance, with the highest R-squared value and the lowest error metrics.

---

## Conclusion

This project successfully demonstrated how to predict car prices using machine-learning regression models. The Gradient Boosting Regressor emerged as the best-performing model, achieving the highest accuracy. These insights could assist car manufacturers, dealerships, and buyers in making informed decisions.

---

## Dependencies

To run the notebook, ensure the following Python libraries are installed:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

---

## How to Run

1. Clone the repository and navigate to the project directory.
2. Place the dataset (`cars_price.csv`) in the appropriate folder.
3. Open the notebook and execute the cells sequentially.
4. View the results and visualizations generated in the notebook.

---




