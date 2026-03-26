# Final Fraud Detector

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# 1. loading the sample data
data = pd.read_csv("fraud.csv")

X = data.drop("Class", axis=1)
y = data["Class"]

# Keep original copy for rule-based system
X_original = X.copy()


# 2. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Split original data too
X_train_orig, X_test_orig, _, _ = train_test_split(
    X_original, y, test_size=0.2, random_state=42
)


# 3. rule based system
def rule_based_check(amount):
    if amount > 10000:
        return 1
    return 0


# 4.  scaliang and model testing here
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=5000, class_weight='balanced')
model.fit(X_train, y_train)


# 5. sample test predictions

# Scaled sample for model
sample = X_test[0].reshape(1, -1)

# Original sample for rule-based check (FIXED)
amount = X_test_orig.iloc[0]["Amount"]

print("\n--- FRAUD DETECTION SYSTEM ---")

# Rule-based check
if rule_based_check(amount):
    print("⚠️ High-risk transaction detected by rule-based system")
else:
    print("Checking with ML model...")

# ML prediction
prediction = model.predict(sample)[0]
probability = model.predict_proba(sample)[0][1]

print("Prediction (0 = Normal, 1 = Fraud):", prediction)
print("Fraud Probability:", probability)


# 6. model evaluation
y_pred = model.predict(X_test)

print("\n--- EVALUATION ---")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# 7. matrix visuallization
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Confusion Matrix")
plt.show()


# 8. custom test


print("\n--- CUSTOM TEST ---")

# Use scaled data (NOT .iloc)
custom = X_test[1].reshape(1, -1)

pred = model.predict(custom)[0]
prob = model.predict_proba(custom)[0][1]

print("Custom Prediction:", pred)
print("Custom Fraud Probability:", prob)
print("NOTE: High recall ensures most frauds are detected.")