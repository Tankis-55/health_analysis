import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/clean_heart.csv")

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation of features")
plt.savefig("visualizations/correlation_heatmap.png", dpi=300)
plt.show()

sns.histplot(df["age"], bins=20, kde=True)
plt.title("Age distribution")
plt.savefig("visualizations/age_distribution.png", dpi=300)
plt.show()

sns.boxplot(x="sex", y="chol", data=df)
plt.title("Cholesterol level by gender")
plt.savefig("visualizations/cholesterol_by_gender.png", dpi=300)
plt.show()