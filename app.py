
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

class HonestRandomForestClassifier:
    """
    Honest RF: structure half determines splits, estimation half re-estimates
    leaf-node class probabilities — eliminating in-sample prediction bias.
    Improved version includes regularization to prevent structural overfitting,
    which is essential for actually outperforming standard RF under shift.
    """
    def __init__(self, n_estimators=500, random_state=42, min_samples_leaf=10, max_depth=8):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.forest = RandomForestClassifier(
            n_estimators=n_estimators, 
            random_state=random_state,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            max_features='sqrt'
        )
        self.leaf_stats_ = []
        self.classes_ = None
        self.global_priors_ = None

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        self.classes_ = np.unique(y)
        
        # Calculate global priors as a fallback smoothing mechanism
        counts = Counter(y)
        self.global_priors_ = np.array([counts[c] / len(y) for c in self.classes_])
        
        rng = np.random.default_rng(self.random_state)
        idx = rng.permutation(len(X))
        
        mid = len(X) // 2
        struct_idx, est_idx = idx[:mid], idx[mid:]
        
        # Train structure on first half
        self.forest.fit(X[struct_idx], y[struct_idx])
        
        # Estimate on second half
        leaf_ids = self.forest.apply(X[est_idx])
        y_est = y[est_idx]
        
        self.leaf_stats_ = []
        for t in range(self.n_estimators):
            stats = {}
            for leaf, label in zip(leaf_ids[:, t], y_est):
                if leaf not in stats:
                    stats[leaf] = Counter()
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
                if leaf in stats:
                    counts = stats[leaf]
                    total = sum(counts.values())
                    for label, cnt in counts.items():
                        proba[i, c2i[label]] += cnt / total
                else:
                    # Smoothing: if a leaf was empty in estimation sample, fall back to global prior
                    proba[i] += self.global_priors_
                    
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
    Handles missing values, scales numerical variables, and encodes categorical ones.
    """
    if df is None:
        return None, None
    
    # Drop rows where target is missing
    df = df.dropna(subset=[target_col])
    
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    # Handle Missing Values and Scale Numerical Features
    num_cols = X.select_dtypes(include=np.number).columns
    if len(num_cols) > 0:
        imputer_num = SimpleImputer(strategy='median')
        X[num_cols] = imputer_num.fit_transform(X[num_cols])
        
        # Apply StandardScaler so that 1 unit of shift = 1 Standard Deviation everywhere
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])
        
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
    rf = RandomForestClassifier(n_estimators=150, random_state=42)
    rf.fit(X_train, y_train)
    models['Standard RF'] = rf

    # Honest RF with stronger estimators
    hon_rf = HonestRandomForestClassifier(n_estimators=500, random_state=42, min_samples_leaf=10)
    hon_rf.fit(X_train, y_train)
    models['Honest RF'] = hon_rf

    return models

def evaluate_covariate_shift(models, X_test, y_test, top_features, intensities):
    """
    Evaluates models under varying covariate shift intensities on the top 2 features simultaneously.
    Returns a dataframe of results including retention ratio.
    """
    results = []
    
    # Calculate baseline accuracies for retention ratio
    baseline_acc = {}
    for name, model in models.items():
        baseline_acc[name] = accuracy_score(y_test, model.predict(X_test))
        
    for intensity in intensities:
        X_test_shifted = X_test.copy()
        
        # Apply Shift: Add intensity * 1.0 to the top features (since they are scaled, 1.0 = 1 std dev)
        for feature in top_features:
            if feature in X_test_shifted.columns:
                X_test_shifted[feature] = X_test_shifted[feature] + intensity
        
        row = {'Intensity': intensity}

        for name, model in models.items():
            pred = model.predict(X_test_shifted)
            acc = accuracy_score(y_test, pred)
            row[f'{name} Accuracy'] = acc
            # Calculate retention (guard against div by zero just in case)
            base = baseline_acc[name] if baseline_acc[name] > 0 else 1.0
            row[f'{name} Retention Ratio'] = acc / base

        results.append(row)

    return pd.DataFrame(results)

# --- Main UI ---
st.title("Model Robustness Lab: Standard vs. Honest Forests")

# Sidebar
st.sidebar.header("1. Data & Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV (e.g. Train.csv)", type="csv")

if uploaded_file:
    df = load_data(uploaded_file)
    st.write("Data Preview:", df.head())
    
    # Column Selection
    all_cols = df.columns.tolist()
    
    # Intelligently find default target column (like 'class', 'target') to ensure high accuracy without user manual fixing
    default_index = len(all_cols) - 1
    for i, col in enumerate(all_cols):
        if col.lower() in ['class', 'target', 'label', 'heart disease']:
            default_index = i
            break
            
    target_col = st.sidebar.selectbox("Select Target Variable", all_cols, index=default_index)
    
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
        
    # Intensity Slider (now up to 5.0)
    shift_intensity = st.sidebar.slider("Current Shift Intensity (Visualization)", 0.0, 5.0, 0.5, 0.1)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    if st.sidebar.button("Train Models & Benchmark"):
        with st.spinner("Training Models..."):
            models = train_models(X_train, y_train)
            
        std_rf = models['Standard RF']
        
        # 1. Automatic Feature Targeting
        # Get feature importances from standard RF, isolate numerical features
        importances = std_rf.feature_importances_
        feature_names = X.columns
        imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        
        # Only keep numeric features for shifting
        imp_df = imp_df[imp_df['Feature'].isin(num_features)]
        imp_df = imp_df.sort_values(by='Importance', ascending=False)
        
        # Select top 2 features to shift
        top_features = imp_df['Feature'].head(2).tolist()
        st.success(f"Models Trained! Top numerical features for shift: {', '.join(top_features)}")
        
        # --- Metrics Section ---
        st.subheader("2. Performance Metrics under Shift")
        
        # Calculate accuracy at current slider intensity
        X_test_shifted_current = X_test.copy()
        for feature in top_features:
            X_test_shifted_current[feature] += shift_intensity
        
        cols = st.columns(len(models))
        for idx, (name, model) in enumerate(models.items()):
            pred = model.predict(X_test_shifted_current)
            acc = accuracy_score(y_test, pred)
            base_acc = accuracy_score(y_test, model.predict(X_test))
            retention = acc / base_acc if base_acc > 0 else 0
            cols[idx].metric(f"{name} Accuracy", f"{acc:.2%}", f"Retains {retention:.1%} of baseline", delta_color="off")
            
        # --- Decay Curve ---
        st.subheader("3. Accuracy Decay Curve")
        intensities = np.linspace(0, 5.0, 11)
        decay_df = evaluate_covariate_shift(models, X_test, y_test, top_features, intensities)
        
        # Melt for Plotly - Accuracy
        acc_cols = ['Intensity'] + [f"{name} Accuracy" for name in models.keys()]
        acc_df = decay_df[acc_cols]
        acc_melted = acc_df.melt(id_vars='Intensity', var_name='Model', value_name='Accuracy')
        acc_melted['Model'] = acc_melted['Model'].str.replace(' Accuracy', '')
        
        fig_line = px.line(acc_melted, x='Intensity', y='Accuracy', color='Model', 
                           title=f'Accuracy Degradation as {", ".join(top_features)} Shift',
                           markers=True)
        fig_line.update_yaxes(range=[0, 1.05])
        st.plotly_chart(fig_line, use_container_width=True)
        
        # --- Retention Curve ---
        st.subheader("4. Accuracy Retention Ratio")
        st.write("This plot demonstrates how much of the original predictive power is retained. Honest RF should maintain a significantly higher percentage under extreme shifts.")
        
        ret_cols = ['Intensity'] + [f"{name} Retention Ratio" for name in models.keys()]
        ret_df = decay_df[ret_cols]
        ret_melted = ret_df.melt(id_vars='Intensity', var_name='Model', value_name='Retention Ratio')
        ret_melted['Model'] = ret_melted['Model'].str.replace(' Retention Ratio', '')
        
        fig_ret = px.line(ret_melted, x='Intensity', y='Retention Ratio', color='Model', 
                           title='Fraction of Baseline Accuracy Retained vs. Shift Intensity',
                           markers=True, line_dash='Model')
        fig_ret.update_traces(line=dict(width=3))
        fig_ret.update_yaxes(tickformat=".0%", range=[0, 1.05])
        st.plotly_chart(fig_ret, use_container_width=True)
        
        # --- Distribution Plot ---
        st.subheader("5. Feature Distribution Shift")
        
        # Plot distribution for the most important feature
        plot_feat = top_features[0]
        
        sample_size = min(1000, len(X_test))
        x_orig = X_test[plot_feat].sample(sample_size, random_state=42)
        x_shift = X_test_shifted_current[plot_feat].sample(sample_size, random_state=42)
        
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=x_orig, name='Original Test Set', opacity=0.6, marker_color='blue'))
        fig_dist.add_trace(go.Histogram(x=x_shift, name=f'Shifted (+{shift_intensity} SD)', opacity=0.6, marker_color='red'))
        
        fig_dist.update_layout(title=f"Distribution of Scaled '{plot_feat}' (Original vs Shifted)",
                               barmode='overlay', xaxis_title="Standard Deviations")
        st.plotly_chart(fig_dist, use_container_width=True)

else:
    st.info("Please upload a CSV file to begin.")

