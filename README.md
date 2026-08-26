# House Price Prediction - Machine Learning Project

A comprehensive machine learning project for predicting house prices using Python. This project includes data cleaning, exploratory data analysis (EDA), feature engineering, and model building.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Project Structure](#project-structure)
- [Deliverables](#deliverables)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Features Created](#features-created)
- [Results](#results)
- [Technical Details](#technical-details)

## Project Overview

This project analyzes house price data to understand the factors influencing property prices and build a predictive model. It covers the complete data science pipeline from data cleaning to model preparation.

### Key Objectives
- Perform comprehensive Exploratory Data Analysis (EDA)
- Clean and preprocess data
- Create meaningful features
- Identify patterns and correlations
- Prepare data for machine learning modeling

## Dataset Description

The dataset contains information about houses with the following features:

| Feature | Description | Data Type |
|---------|-------------|-----------|
| area_sqft | Area of the house in square feet | float |
| bedrooms | Number of bedrooms | int/float |
| age_years | Age of the house in years | float |
| distance_city_km | Distance from city center in kilometers | float |
| price_lakh | House price in lakhs (Target Variable) | float |

### Sample Data

area_sqft bedrooms age_years distance_city_km price_lakh
0 849.0 5.0 33.0 17.6 66.5
1 2068.0 4.0 18.0 1.6 120.2
2 1352.0 3.0 30.0 9.6 96.7
3 655.0 3.0 36.0 5.2 51.2
4 1062.0 1.0 32.0 24.0 58.4

## Project Structure

## Deliverables

This project implements the following 11 deliverables:

| Number | Deliverable | Status | Description |
|--------|-------------|--------|-------------|
| 24 | Basic EDA and Data Types | Done | Inspected data structure, types, and summary statistics |
| 25 | Missing Values and Duplicates | Done | Identified and handled missing values and duplicates |
| 26 | Impossible Values | Done | Found and removed negative area, unreasonable bedrooms, etc. |
| 27 | Price Outliers | Done | Detected outliers using IQR and Z-score methods |
| 28 | Area and Price Distributions | Done | Plotted histograms, boxplots, and density plots |
| 29 | Relationship Analysis | Done | Analyzed correlations with price through scatter plots |
| 30 | Correlation Heatmap | Done | Created heatmap showing feature relationships |
| 31 | Data Cleaning | Done | Removed duplicates, invalid values, and missing data |
| 32 | Price per sqft | Done | Created price per square foot feature |
| 33 | Additional Features | Done | Engineered multiple new meaningful features |
| 34 | X-y Separation | Done | Split data into features and target variable |

## Installation
Run the complete analysis script that handles everything:
cd "D:\House Price ML"
C:\Users\91637\.local\bin\python3.14.exe "complete_analysis_direct.py"

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Install Dependencies

Create a `requirements.txt` file with:

Install using pip:
```bash
pip install -r requirements.txt

Key Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


Key Findings
Strong Positive Correlation: Area (0.914) has the strongest correlation with price

Moderate Negative Correlation: Age (-0.341) and Distance (-0.290) negatively correlate with price

Weak Correlation: Bedrooms (0.307) shows moderate correlation with price

Data Quality
No missing values after cleaning

No duplicate rows

No invalid values

Dataset Statistics
Total Rows: Varies based on dataset

Total Features: 8 (including engineered features)

Target Variable: price_lakh

Technical Details
Technologies Used
Technology	Purpose
Python 3.14	Programming language
Pandas	Data manipulation and analysis
NumPy	Numerical computing
Matplotlib	Data visualization
Seaborn	Statistical data visualization
Scikit-learn	Machine learning utilities
SciPy	Scientific computing
Key Libraries
python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
Output Files
File	Description
House_Price_Cleaned.csv	Cleaned dataset without invalid values
House_Price_Final.csv	Final dataset with all features
X_train.csv	Training features
X_test.csv	Testing features
y_train.csv	Training target values
y_test.csv	Testing target values
Quick Start
bash
# 1. Navigate to project directory
cd "D:\House Price ML"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run complete analysis
python complete_analysis_direct.py

# 4. View results
# - Check House_Price_Final.csv for processed data
# - Review generated plots
# - See console output for analysis results

