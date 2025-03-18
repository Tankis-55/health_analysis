Overview

Health Analysis is a Python project for analyzing heart disease data. It includes data cleaning, visualization, and machine learning predictions for heart disease risk assessment.

Features

Cleans and processes heart disease datasets.

Trains a Random Forest model to predict heart disease risk.

Saves and loads trained models for future use.

Installation

Clone the repository:

git clone https://github.com/yourusername/health_analysis.git
cd health_analysis

Create a virtual environment and activate it:

python3 -m venv venv
source venv/bin/activate


Usage

Data Cleaning

python scripts/data_cleaning.py

Training the Model

python scripts/train_model.py

Making Predictions

python scripts/predict.py

Model Storage

The trained model is saved as models/heart_disease_model.pkl and can be reloaded for future predictions.

Dependencies

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn


Integrate additional health datasets.

Implement advanced machine learning techniques.
