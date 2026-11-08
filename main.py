import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             classification_report, accuracy_score)

os.makedirs("figures", exist_ok=True)
RANDOM_STATE = 42

# ---------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------
df = pd.read_csv(r"C:\My Stuff\Programming\Coding\VS Code\Python\Free Time\School Stuff\AI Project\heart.csv")

# Rename original Kaggle columns to the friendlier names (skipped if already renamed)
rename_map = {
    "dataset": "Hospital", "cp": "chest_pain_type", "trestbps": "resting_BP",
    "chol": "Cholestrol", "fbs": "fasting_blood_sugar", "restecg": "ecg_obs_rest",
    "thalch": "max_heartrate", "num": "Target",
}
df = df.rename(columns=rename_map)

print("Shape:", df.shape)
print(df.head())

# ---------------------------------------------------------------
# 2. CLEAN DATA
# ---------------------------------------------------------------
# 2a. Zero cholesterol / zero BP is physically impossible -> treat as missing
df["Cholestrol"] = df["Cholestrol"].replace(0, np.nan)
df["resting_BP"] = df["resting_BP"].replace(0, np.nan)

# 2b. Target: 0 = no disease, 1-4 = disease  ->  binary 0/1
df["Target"] = (df["Target"] > 0).astype(int)

# 2c. Missing-value percentage per column (used for a plot and for the drop decision)
missing_pct = (df.isnull().mean() * 100).sort_values(ascending=False)
print("\nMissing % per column:\n", missing_pct.round(1))

# 2d. Drop columns that are not useful features:
#     id = just a row number, Hospital = would let the model learn the hospital,
#     ca and thal = more than 50% missing (too little real information to fill)
df = df.drop(columns=["id", "Hospital", "ca", "thal"])

# 2e. Define column groups
num_cols = ["age", "resting_BP", "Cholestrol", "max_heartrate", "oldpeak"]
cat_cols = ["sex", "chest_pain_type", "fasting_blood_sugar",
            "ecg_obs_rest", "exang", "slope"]

# True/False columns contain NaN too; convert every category value to text
# (keeping NaN as NaN) so the imputer and encoder treat them all the same way
for c in cat_cols:
    df[c] = df[c].map(lambda v: np.nan if pd.isna(v) else str(v))

# ---------------------------------------------------------------
# 3. EDA PLOTS
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

counts = df["Target"].value_counts().sort_index()
axes[0].bar(["No disease (0)", "Disease (1)"], counts.values, color=["#4C9F70", "#D1495B"])
axes[0].set_title("Class balance")
axes[0].set_ylabel("Patients")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 5, str(v), ha="center")

top_missing = missing_pct[missing_pct > 0]
axes[1].barh(top_missing.index, top_missing.values, color="#EDAE49")
axes[1].invert_yaxis()
axes[1].set_title("Missing values (%) before cleaning")
axes[1].set_xlabel("% missing")

plt.tight_layout()
plt.savefig("figures/1_eda.png", dpi=150)
plt.show()

# Distribution of numeric features split by Target
fig, axes = plt.subplots(1, len(num_cols), figsize=(16, 3.5))
for ax, col in zip(axes, num_cols):
    ax.hist(df.loc[df["Target"] == 0, col].dropna(), bins=20, alpha=0.6,
            label="No disease", color="#4C9F70")
    ax.hist(df.loc[df["Target"] == 1, col].dropna(), bins=20, alpha=0.6,
            label="Disease", color="#D1495B")
    ax.set_title(col)
axes[0].legend()
plt.tight_layout()
plt.savefig("figures/2_distributions.png", dpi=150)
plt.show()

# ---------------------------------------------------------------
# 4. PREPROCESSING + MODELS (inside Pipelines to avoid data leakage)
# ---------------------------------------------------------------
X = df.drop(columns=["Target"])
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print("\nTrain size:", X_train.shape, " Test size:", X_test.shape)

numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])
categorical_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipe, num_cols),
    ("cat", categorical_pipe, cat_cols),
])

models = {
    "Decision Tree": Pipeline([
        ("prep", preprocessor),
        ("model", DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE)),
    ]),
    "Logistic Regression": Pipeline([
        ("prep", preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ]),
}

# ---------------------------------------------------------------
# 5. 5-FOLD CROSS-VALIDATION (on the training set only)
# ---------------------------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
metrics = ["accuracy", "precision", "recall", "f1"]
cv_results = {}

print("\n5-fold cross-validation (mean +/- std):")
for name, pipe in models.items():
    scores = cross_validate(pipe, X_train, y_train, cv=skf, scoring=metrics)
    cv_results[name] = {m: scores[f"test_{m}"] for m in metrics}
    print(f"\n{name}")
    for m in metrics:
        s = cv_results[name][m]
        print(f"  {m:<10}: {s.mean():.3f} +/- {s.std():.3f}")

# Plot: CV comparison
x = np.arange(len(metrics))
width = 0.35
fig, ax = plt.subplots(figsize=(8, 4.5))
for i, (name, color) in enumerate(zip(cv_results, ["#2E86AB", "#F18F01"])):
    means = [cv_results[name][m].mean() for m in metrics]
    stds = [cv_results[name][m].std() for m in metrics]
    ax.bar(x + i * width - width / 2, means, width, yerr=stds,
           capsize=4, label=name, color=color)
ax.set_xticks(x)
ax.set_xticklabels([m.capitalize() for m in metrics])
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title("5-fold cross-validation: model comparison")
ax.legend()
plt.tight_layout()
plt.savefig("figures/3_cv_comparison.png", dpi=150)
plt.show()

# Plot: score of each individual fold
fig, ax = plt.subplots(figsize=(7, 4))
for name, color in zip(cv_results, ["#2E86AB", "#F18F01"]):
    ax.plot(range(1, 6), cv_results[name]["accuracy"], marker="o", label=name, color=color)
ax.set_xticks(range(1, 6))
ax.set_xlabel("Fold")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy in each fold")
ax.legend()
plt.tight_layout()
plt.savefig("figures/4_fold_accuracy.png", dpi=150)
plt.show()

# ---------------------------------------------------------------
# 6. FINAL TEST-SET EVALUATION
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (name, pipe) in zip(axes, models.items()):
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print(f"\n===== {name} (test set) =====")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
    print(classification_report(y_test, y_pred, target_names=["No disease", "Disease"]))
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["No disease", "Disease"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(name)
plt.tight_layout()
plt.savefig("figures/5_confusion_matrices.png", dpi=150)
plt.show()

# ---------------------------------------------------------------
# 7. INTERPRETATION PLOTS
# ---------------------------------------------------------------
feature_names = [n.split("__", 1)[1] for n in
                 models["Decision Tree"].named_steps["prep"].get_feature_names_out()]

# 7a. The decision tree itself
tree = models["Decision Tree"].named_steps["model"]
plt.figure(figsize=(20, 9))
plot_tree(tree, feature_names=feature_names, class_names=["No disease", "Disease"],
          filled=True, rounded=True, fontsize=8)
plt.title("Decision Tree (max_depth = 4)")
plt.savefig("figures/6_decision_tree.png", dpi=150)
plt.show()

# 7b. Decision tree feature importance
imp = pd.Series(tree.feature_importances_, index=feature_names).sort_values()
imp = imp[imp > 0].tail(10)
plt.figure(figsize=(7, 4.5))
plt.barh(imp.index, imp.values, color="#2E86AB")
plt.title("Decision Tree: top feature importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("figures/7_tree_importance.png", dpi=150)
plt.show()

# 7c. Logistic regression coefficients (positive = pushes towards disease)
logreg = models["Logistic Regression"].named_steps["model"]
coef = pd.Series(logreg.coef_[0], index=feature_names).sort_values()
top = pd.concat([coef.head(7), coef.tail(7)])
colors = ["#D1495B" if v > 0 else "#4C9F70" for v in top.values]
plt.figure(figsize=(8, 5.5))
plt.barh(top.index, top.values, color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Logistic Regression: strongest coefficients\n(red = raises risk, green = lowers risk)")
plt.xlabel("Coefficient")
plt.tight_layout()
plt.savefig("figures/8_logreg_coefficients.png", dpi=150)
plt.show()