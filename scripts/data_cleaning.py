import pandas as pd

df = pd.read_csv("data/heart.csv")

print("Missing values before processing:")
print(df.isnull().sum())

df.fillna(df.median(), inplace=True)

df.drop_duplicates(inplace=True)

df = df.dropna()

df.to_csv("data/clean_heart.csv", index=False)

print("Data cleared and saved in clean_heart.csv")