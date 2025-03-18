import pickle
import joblib
import numpy as np

with open("models/heart_disease_model.pkl", "rb") as file:
    model = pickle.load(file)

model = joblib.load("models/heart_disease_model.pkl")

sample_patient = np.array([[55, 140, 250, 1, 0, 150, 0, 2.3, 1, 0, 2, 3, 1]])  # Возраст, давление, холестерин и т. д.

prediction = model.predict(sample_patient)

print("Prognosis:", "Heart disease" if prediction[0] == 1 else "No disease")