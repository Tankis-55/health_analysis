import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("data/clean_heart.csv")

X = df.drop(columns=["target"]) 
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Model accuracy:  {accuracy:.2f}")

import joblib
joblib.dump(model, "models/heart_disease_model.pkl")

print("The model is saved!")
print(type(model))
print(model.get_params())

import pickle

with open("models/heart_disease_model.pkl", "rb") as file:
    model = pickle.load(file)

print("The model is loaded: ", model)