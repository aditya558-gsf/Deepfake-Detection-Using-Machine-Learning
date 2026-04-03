import pickle
import numpy as np

# =========================
# LOAD DATA
# =========================
with open('power_spectrum_10000.pkl', 'rb') as f:
    data = pickle.load(f)

X = data["power_spectrum"]
y = data["label"]

# =========================
# TRAIN-TEST SPLIT
# =========================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# MODEL (Random Forest)
# =========================
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42
)

# =========================
# TRAIN MODEL
# =========================
model.fit(X_train, y_train)

# =========================
# PREDICTION
# =========================
y_pred = model.predict(X_test)

# =========================
# EVALUATION
# =========================
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("\n===== RESULTS =====")

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# =========================
# OVERFITTING CHECK
# =========================
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print("\nTrain Accuracy:", train_acc)
print("Test Accuracy:", test_acc)

# =========================
# SAVE MODEL
# =========================
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nModel saved successfully!")
