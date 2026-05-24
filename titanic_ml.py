import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────
df = pd.read_csv("Titanic-Dataset.csv")
print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

# ─────────────────────────────────────────
# 2. DATA PREPROCESSING
# ─────────────────────────────────────────

# Drop irrelevant columns
df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)

# Fill missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Encode categorical columns
le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])           # male=1, female=0
df["Embarked"] = le.fit_transform(df["Embarked"]) # S=2, C=0, Q=1

print("\nProcessed Data Sample:")
print(df.head())

# ─────────────────────────────────────────
# 3. FEATURE SELECTION
# ─────────────────────────────────────────
X = df.drop(columns=["Survived"])
y = df["Survived"]

# ─────────────────────────────────────────
# 4. TRAIN-TEST SPLIT
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ─────────────────────────────────────────
# 5. FEATURE SCALING
# ─────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─────────────────────────────────────────
# 6. TRAIN MULTIPLE MODELS & COMPARE
# ─────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM":                 SVC(probability=True, random_state=42),
    "KNN":                 KNeighborsClassifier(n_neighbors=5),
}

results = {}
for name, model in models.items():
    # Use scaled data for LR, SVM, KNN; unscaled for RF
    X_tr = X_train_scaled if name != "Random Forest" else X_train
    X_te = X_test_scaled  if name != "Random Forest" else X_test

    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, model.predict_proba(X_te)[:, 1])
    cv   = cross_val_score(model, X_tr, y_train, cv=5, scoring="accuracy").mean()

    results[name] = {"Accuracy": acc, "ROC-AUC": auc, "CV Accuracy": cv}
    print(f"\n{'='*40}\n{name}")
    print(f"  Accuracy   : {acc:.4f}")
    print(f"  ROC-AUC    : {auc:.4f}")
    print(f"  CV Accuracy: {cv:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Did Not Survive", "Survived"]))

# ─────────────────────────────────────────
# 7. HYPERPARAMETER TUNING (Random Forest)
# ─────────────────────────────────────────
print("\nTuning Random Forest...")
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
}
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=5, scoring="accuracy", n_jobs=-1
)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
print("Best Params:", grid_search.best_params_)
print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")

# ─────────────────────────────────────────
# 8. EVALUATION PLOTS
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Titanic Survival Prediction - Model Evaluation", fontsize=15, fontweight="bold")

# --- Confusion Matrix (Best RF) ---
y_pred_best = best_rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["Did Not Survive", "Survived"],
            yticklabels=["Did Not Survive", "Survived"])
axes[0].set_title("Confusion Matrix\n(Best Random Forest)")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# --- ROC Curves (All Models) ---
for name, model in models.items():
    X_te = X_test_scaled if name != "Random Forest" else X_test
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_te)[:, 1])
    auc = roc_auc_score(y_test, model.predict_proba(X_te)[:, 1])
    axes[1].plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")
axes[1].plot([0, 1], [0, 1], "k--")
axes[1].set_title("ROC Curves - All Models")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

# --- Feature Importances (Best RF) ---
feat_imp = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=True)
feat_imp.plot(kind="barh", ax=axes[2], color="#3498db", edgecolor="white")
axes[2].set_title("Feature Importances\n(Best Random Forest)")
axes[2].set_xlabel("Importance Score")
axes[2].grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig("titanic_model_evaluation.png", dpi=150, bbox_inches="tight")
plt.show()

# ─────────────────────────────────────────
# 9. PREDICT ON A NEW PASSENGER
# ─────────────────────────────────────────
# Example: 30-year-old female, 1st class, no siblings, no parents, fare=50, embarked S
new_passenger = pd.DataFrame([{
    "Pclass": 1, "Sex": 0, "Age": 30, "SibSp": 0,
    "Parch": 0, "Fare": 50, "Embarked": 2
}])

prediction = best_rf.predict(new_passenger)[0]
probability = best_rf.predict_proba(new_passenger)[0][1]
print(f"\n{'='*40}")
print(f"New Passenger Prediction:")
print(f"  Survived : {'YES ✓' if prediction == 1 else 'NO ✗'}")
print(f"  Survival Probability: {probability:.2%}")