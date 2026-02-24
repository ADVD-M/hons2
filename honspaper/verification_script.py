
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys

# Mock Data Generation if file not found, but we should try to use the real one
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

# Preprocessing Logic
target_col = 'Segmentation' if 'Segmentation' in df.columns else 'Target'
y = df[target_col]
X = df.drop(columns=[target_col])

# Handle Missing
num_cols = X.select_dtypes(include=np.number).columns
if len(num_cols) > 0:
    imputer_num = SimpleImputer(strategy='median')
    X[num_cols] = imputer_num.fit_transform(X[num_cols])

cat_cols = X.select_dtypes(include='object').columns
if len(cat_cols) > 0:
    for col in cat_cols:
        # Fill missing first
        X[col] = X[col].fillna(X[col].mode()[0])
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

# Encode Target
le_y = LabelEncoder()
y = le_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Data Preprocessing Complete.")
print(f"X_train shape: {X_train.shape}")

# Model Training
print("\nTraining Standard RF...")
rf = RandomForestClassifier(n_estimators=10, random_state=42)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
print(f"Standard RF Accuracy: {accuracy_score(y_test, pred_rf):.4f}")

# Honest RF
try:
    from econml.grf import ProbabilityForest
    print("\nTraining Honest RF (econml)...")
    # honest=True is the key parameter here
    hon_rf = ProbabilityForest(n_estimators=10, honest=True, random_state=42)
    hon_rf.fit(X_train, y_train)
    pred_hon = np.argmax(hon_rf.predict(X_test), axis=1)
    print(f"Honest RF Accuracy: {accuracy_score(y_test, pred_hon):.4f}")
except ImportError:
    print("\n[WARNING] econml not found or failed to import.")
    print("Please install via: pip install econml")
except Exception as e:
    print(f"\n[ERROR] Honest RF failed: {e}")

# Shift Logic Test
shift_feature = 'Age'
if shift_feature in X_train.columns:
    print(f"\nApplying Shift to {shift_feature}...")
    std_dev = X_train[shift_feature].std()
    X_test_shifted = X_test.copy()
    X_test_shifted[shift_feature] += (2.0 * std_dev) # 2 sigma shift
    
    pred_rf_shift = rf.predict(X_test_shifted)
    print(f"Standard RF Shifted Accuracy: {accuracy_score(y_test, pred_rf_shift):.4f}")
    
    if 'hon_rf' in locals():
        pred_hon_shift = np.argmax(hon_rf.predict(X_test_shifted), axis=1)
        print(f"Honest RF Shifted Accuracy: {accuracy_score(y_test, pred_hon_shift):.4f}")
else:
    print(f"Feature {shift_feature} not in columns: {X_train.columns}")

print("\nVerification Complete.")
