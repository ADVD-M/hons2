
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter

# --- Honest RF (same implementation as app.py) ---
class HonestRandomForestClassifier:
    def __init__(self, n_estimators=100, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.forest = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self.leaf_stats_ = []
        self.classes_ = None

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        rng = np.random.default_rng(self.random_state)
        idx = rng.permutation(len(X))
        mid = len(X) // 2
        struct_idx, est_idx = idx[:mid], idx[mid:]
        self.forest.fit(X[struct_idx], y[struct_idx])
        self.classes_ = self.forest.classes_
        leaf_ids = self.forest.apply(X[est_idx])
        y_est = y[est_idx]
        self.leaf_stats_ = []
        for t in range(self.n_estimators):
            stats = {}
            for leaf, label in zip(leaf_ids[:, t], y_est):
                stats.setdefault(leaf, Counter())
                stats[leaf][label] += 1
            self.leaf_stats_.append(stats)
        return self

    def predict_proba(self, X):
        X = np.array(X)
        n_classes = len(self.classes_)
        c2i = {c: i for i, c in enumerate(self.classes_)}
        leaf_ids = self.forest.apply(X)
        proba = np.zeros((len(X), n_classes))
        for t in range(self.n_estimators):
            stats = self.leaf_stats_[t]
            for i, leaf in enumerate(leaf_ids[:, t]):
                counts = stats.get(leaf, {})
                total = sum(counts.values()) or 1
                for label, cnt in counts.items():
                    proba[i, c2i[label]] += cnt / total
        return proba / self.n_estimators

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

# --- Data Loading ---
try:
    df = pd.read_csv('Customer segmentation/Customer segmentation/Train.csv')
    print("Loaded Train.csv successfully.")
except FileNotFoundError:
    print("Train.csv not found, creating dummy data.")
    df = pd.DataFrame({
        'Age': np.random.randint(18, 90, 100),
        'Income': np.random.rand(100) * 100000,
        'Category': np.random.choice(['A', 'B', 'C'], 100),
        'Target': np.random.choice(['Yes', 'No'], 100)
    })

# --- Preprocessing ---
target_col = 'Segmentation' if 'Segmentation' in df.columns else 'Target'
y = df[target_col]
X = df.drop(columns=[target_col])

num_cols = X.select_dtypes(include=np.number).columns
if len(num_cols) > 0:
    imputer_num = SimpleImputer(strategy='median')
    X[num_cols] = imputer_num.fit_transform(X[num_cols])

cat_cols = X.select_dtypes(include='object').columns
if len(cat_cols) > 0:
    for col in cat_cols:
        X[col] = X[col].fillna(X[col].mode()[0])
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

le_y = LabelEncoder()
y = le_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Data Preprocessing Complete.")
print(f"X_train shape: {X_train.shape}")

# --- Model Training ---
print("\nTraining Standard RF...")
rf = RandomForestClassifier(n_estimators=10, random_state=42)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
print(f"Standard RF Accuracy: {accuracy_score(y_test, pred_rf):.4f}")

print("\nTraining Honest RF (split-sample)...")
hon_rf = HonestRandomForestClassifier(n_estimators=10, random_state=42)
hon_rf.fit(X_train, y_train)
pred_hon = hon_rf.predict(X_test)
print(f"Honest RF Accuracy:   {accuracy_score(y_test, pred_hon):.4f}")

# --- Covariate Shift Test ---
shift_feature = 'Age'
if shift_feature in X_train.columns:
    print(f"\nApplying 2-sigma shift to '{shift_feature}'...")
    std_dev = X_train[shift_feature].std()
    X_test_shifted = X_test.copy()
    X_test_shifted[shift_feature] += 2.0 * std_dev
    print(f"Standard RF Shifted Accuracy: {accuracy_score(y_test, rf.predict(X_test_shifted)):.4f}")
    print(f"Honest RF   Shifted Accuracy: {accuracy_score(y_test, hon_rf.predict(X_test_shifted)):.4f}")
else:
    print(f"Feature '{shift_feature}' not in columns: {list(X_train.columns)}")

print("\nVerification Complete.")
