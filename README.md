# EDA-OLS-California-House-Prediction
A statistical modeling project focused on predicting the median house value in various California districts using Ordinary Least Squares (OLS) Linear Regression. The analysis covers data cleaning, feature engineering, and pipeline implementation.

## Description
This project utilizes a large dataset of California housing features to build a predictive model for house prices. It serves as a practical demonstration of several key Machine Learning concepts: handling categorical data through One-Hot Encoding, addressing missing values, building an sklearn Pipeline, and interpreting the results of a multiple Linear Regression model. The analysis is fully documented in OLS California.md.

## Project Features & Goals
1. House Price Prediction: Predict the target variable: median_house_value.
2. Feature Engineering: Conversion of the categorical ocean_proximity feature into a numerical format.
3. Pipeline Implementation: Streamlining the preprocessing and modeling steps using sklearn.pipeline.
4. Statistical Analysis: Generating and interpreting the correlation matrix to understand feature relationships.

## Getting Started
The project was run on Jupyter Notebook.

### Data
The project depends on the following file, which should be placed in the root directory : California house price dataset.csv (Source: https://www.kaggle.com/datasets/nazishjaveed/california-house-price-prediction)

### Execution
1. Run the analysis: Open and execute the steps within OLS California.md in a compatible Markdown viewer or Jupyter environment.

##  Methodology and Analysis StepsThe OLS California.md file documents the full analytical process from data loading to model preparation.
1. Setup and Data Loading. The session begins by importing necessary libraries and suppressing future warnings, followed by loading the dataset into a pandas DataFrame:
2.  Data Exploration (df.info()). Initial inspection revealed the following structure and data quality issues:Total Entries: 20,640Missing Data: The total_bedrooms column contains 207 non-null values less than the total, indicating missing data that must be imputed or dropped.Categorical Feature: The ocean_proximity column is an object type (categorical) and needs encoding.
3. Feature Engineering and Preprocessing. The analysis prepared the data for Linear Regression by addressing the categorical feature:One-Hot Encoding: The categorical feature ocean_proximity was transformed into separate binary (dummy) variables (e.g., ocean_proximity_INLAND, ocean_proximity_NEAR BAY) using category_encoders.OneHotEncoder.
4. Correlation Analysis. A large correlation matrix was generated (partially visible in the file snippet) to understand the linear relationship between the input features and the target variable (median_house_value), which is a crucial step before fitting an OLS model.
5. Model Pipeline Construction. A machine learning Pipeline was set up to chain the preprocessing (e.g., imputation/encoding) and the modeling steps: Model: sklearn.linear_model.LinearRegression (OLS). Pipeline: The process utilizes sklearn.pipeline.Pipeline or make_pipeline to ensure data leakage is prevented and the workflow is reproducible.
6. Model Training and Evaluation (Implied). The final steps involve splitting the data into training and test sets (train_test_split) and fitting the Pipeline to calculate the final OLS coefficients and evaluate the model's performance metrics.
