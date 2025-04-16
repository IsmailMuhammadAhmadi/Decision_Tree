import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Pengumpulan & Persiapan Data
red_wine = pd.read_csv("dataset/winequality-red.csv", sep=';')
white_wine = pd.read_csv("dataset/winequality-white.csv", sep=';')

red_wine["type"] = "red"
white_wine["type"] = "white"
wine_data = pd.concat([red_wine, white_wine], axis=0)
wine_data["type"] = wine_data["type"].map({"red": 0, "white": 1})

# 2. Pilih Fitur 
features = ['alcohol', 'sulphates', 'citric acid', 'volatile acidity']
X = wine_data[features]
y = wine_data["quality"]

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Visualisasi Distribusi Label
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.countplot(x=y_train, ax=ax[0], palette='Blues')
ax[0].set_title("Distribusi Label - Training")
sns.countplot(x=y_test, ax=ax[1], palette='Greens')
ax[1].set_title("Distribusi Label - Testing")
plt.tight_layout()
plt.show()

# 5. Training Model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 6. Evaluasi Model
y_pred = model.predict(X_test)
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

# 7. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.title("Confusion Matrix")
plt.show()

# 8. Accuracy
print(f"Akurasi: {accuracy_score(y_test, y_pred):.4f}")

# 9. Visualisasi Tree
plt.figure(figsize=(16, 8))
plot_tree(model, feature_names=features, class_names=[str(i) for i in sorted(y.unique())], filled=True)
plt.title("Decision Tree - Wine Quality (Simplified Features)")
plt.show()

# 10. Uji Coba Data Luar
external_sample = X.iloc[[10]]
external_true = y.iloc[10]
external_pred = model.predict(external_sample)[0]

print("\n=== Uji Coba Data Luar ===")
print("Data:", external_sample.to_dict(orient='records')[0])
print("Kelas Asli (quality):", external_true)
print("Prediksi Model (quality):", external_pred)
