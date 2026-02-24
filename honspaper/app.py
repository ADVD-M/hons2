
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter

# --- Honest Random Forest (Wager & Athey 2018 split-sample approach) ---
class HonestRandomForestClassifier:
    """
    Honest RF: structure half determines splits, estimation half re-estimates
    leaf-node class probabilities — eliminating in-sample prediction bias.
    """
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

# --- App Config ---
st.set_page_config(page_title="Model Robustness Lab", layout="wide")

# --- Helper Functions ---
@st.cache_data
def load_data(file):
    if file is not None:
        return pd.read_csv(file)
    return None

def preprocess_data(df, target_col):
    """
    Handles missing values and encodes categorical variables.
    Returns X, y and the encoders for potential inverse transform (not used here but good practice).
    """
    if df is None:
        return None, None
    
    # Drop rows where target is missing
    df = df.dropna(subset=[target_col])
    
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    # Handle Missing Values
    # Numerical: Median
    num_cols = X.select_dtypes(include=np.number).columns
    if len(num_cols) > 0:
        imputer_num = SimpleImputer(strategy='median')
        X[num_cols] = imputer_num.fit_transform(X[num_cols])
        
    # Categorical: Mode + Label Encoding
    cat_cols = X.select_dtypes(include='object').columns
    if len(cat_cols) > 0:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        X[cat_cols] = imputer_cat.fit_transform(X[cat_cols])
        
        le = LabelEncoder()
        for col in cat_cols:
            X[col] = le.fit_transform(X[col].astype(str))
            
    return X, y

def train_models(X_train, y_train):
    """
    Trains Standard RF and Honest RF (split-sample method).
    """
    models = {}

    # Standard RF
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models['Standard RF'] = rf

    # Honest RF (no external dependency)
    hon_rf = HonestRandomForestClassifier(n_estimators=100, random_state=42)
    hon_rf.fit(X_train, y_train)
    models['Honest RF'] = hon_rf

    return models

def evaluate_covariate_shift(models, X_train, X_test, y_test, shift_feature, intensities):
    """
    Evaluates models under varying covariate shift intensities.
    Returns a dataframe of results.
    """
    results = []
    
    # Base standard deviation for the shift
    if shift_feature in X_train.columns:
        std_dev = X_train[shift_feature].std()
    else:
        return pd.DataFrame() # Should not happen with UI checks
        
    for intensity in intensities:
        X_test_shifted = X_test.copy()
        
        # Apply Shift: Add intensity * std_dev to the feature
        X_test_shifted[shift_feature] = X_test_shifted[shift_feature] + (intensity * std_dev)
        
        row = {'Intensity': intensity}

        for name, model in models.items():
            pred = model.predict(X_test_shifted)
            row[f'{name} Accuracy'] = accuracy_score(y_test, pred)

        results.append(row)

    return pd.DataFrame(results)

# --- Main UI ---
st.title("🛡️ Model Robustness Lab: Standard vs. Honest Forests")

# Sidebar
st.sidebar.header("1. Data & Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV (e.g. Train.csv)", type="csv")

if uploaded_file:
    df = load_data(uploaded_file)
    st.write("Data Preview:", df.head())
    
    # Column Selection
    all_cols = df.columns.tolist()
    target_col = st.sidebar.selectbox("Select Target Variable", all_cols, index=len(all_cols)-1)
    
    # Preprocess
    # We need to encode target if it is categorical for the models to work smoothly
    # Let's do a quick encoding of y inside the app logic to be safe
    X, y = preprocess_data(df, target_col)
    
    # If y is object/string, encode it
    if y.dtype == 'object':
        le_y = LabelEncoder()
        y = le_y.fit_transform(y)
    
    # Feature Selection for Shift
    # Only numerical features allow meaningful mean shift
    num_features = X.select_dtypes(include=np.number).columns.tolist()
    if not num_features:
        st.error("No numerical features found for shifting!")
        st.stop()
        
    shift_feature = st.sidebar.selectbox("Select Feature to Shift", num_features, index=0)
    
    # Intensity Slider
    shift_intensity = st.sidebar.slider("Current Shift Intensity (Visualization)", 0.0, 3.0, 0.5, 0.1)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    if st.sidebar.button("Train Models"):
        with st.spinner("Training Models..."):
            models = train_models(X_train, y_train)
        
        st.success("Models Trained!")
        
        # --- Metrics Section ---
        st.subheader("2. Performance Metrics")
        
        # Calculate accuracy at current slider intensity
        std_dev = X_train[shift_feature].std()
        X_test_shifted_current = X_test.copy()
        X_test_shifted_current[shift_feature] += (shift_intensity * std_dev)
        
        cols = st.columns(len(models))
        for idx, (name, model) in enumerate(models.items()):
            pred = model.predict(X_test_shifted_current)
            acc = accuracy_score(y_test, pred)
            cols[idx].metric(f"{name} Accuracy", f"{acc:.2%}")
            
        # --- Decay Curve ---
        st.subheader("3. Accuracy Decay Curve")
        intensities = np.linspace(0, 3.0, 10)
        decay_df = evaluate_covariate_shift(models, X_train, X_test, y_test, shift_feature, intensities)
        
        # Melt for Plotly
        decay_melted = decay_df.melt(id_vars='Intensity', var_name='Model', value_name='Accuracy')
        
        fig_line = px.line(decay_melted, x='Intensity', y='Accuracy', color='Model', 
                           title=f'Accuracy Degradation as {shift_feature} Shifts',
                           markers=True)
        st.plotly_chart(fig_line, use_container_width=True)
        
        # --- Distribution Plot ---
        st.subheader("4. Feature Distribution Shift")
        
        # Original vs Shifted Data for Histogram
        # We take a sample to avoid overcrowding
        sample_size = min(1000, len(X_test))
        x_orig = X_test[shift_feature].sample(sample_size, random_state=42)
        x_shift = X_test_shifted_current[shift_feature].sample(sample_size, random_state=42)
        
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=x_orig, name='Original Test Set', opacity=0.6, marker_color='blue'))
        fig_dist.add_trace(go.Histogram(x=x_shift, name='Shifted Test Set', opacity=0.6, marker_color='red'))
        
        fig_dist.update_layout(title=f"Distribution of '{shift_feature}' (Original vs Shifted)",
                               barmode='overlay')
        st.plotly_chart(fig_dist, use_container_width=True)

else:
    st.info("Please upload a CSV file to begin.")

