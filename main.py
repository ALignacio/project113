# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 2. LOAD DATASET
# This looks for the file named 'phishing.csv' in your folder
filename = 'phishing.csv'

try:
    data = pd.read_csv(filename)
    print("✅ Dataset Loaded Successfully!")
    print(f"Dataset Shape: {data.shape} (Rows, Columns)")
except FileNotFoundError:
    print("❌ Error: 'phishing.csv' not found. Make sure it is in the same folder as this script!")
    exit()

# 3. PRE-PROCESSING
# Drop index column if it exists
if 'Index' in data.columns:
    data = data.drop(columns=['Index'])

# Separate features (X) and target (y)
# X = Data (URL length, SSL status, etc.)
# y = Answer (1 for Legitimate, -1 for Phishing)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. TRAIN THE AI MODEL
print("\n🔄 Training the Random Forest Model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("✅ Model Trained!")

# 5. EVALUATE PERFORMANCE
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n--- 📊 RESULTS ---")
print(f"Accuracy Score: {acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 6. SHOW GRAPHS (Confusion Matrix)
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Phishing Detection)')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# 7. SHOW TOP INDICATORS (For your report)
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

print("\n--- 🔑 TOP 5 SECURITY THREAT INDICATORS ---")
for f in range(5):
    print(f"{f+1}. {X.columns[indices[f]]}: {importances[indices[f]]:.4f}")