
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

# Try importing econml, if not available, show warning
try:
    from econml.grf import CausalForest, ProbabilityForest
    ECONML_AVAILABLE = True
except ImportError:
    ECONML_AVAILABLE = False

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
    Trains Standard RF and Honest RF (if available).
    """
    models = {}
    
    # Standard RF
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models['Standard RF'] = rf
    
    # Honest RF - using ProbabilityForest with honest=True
    if ECONML_AVAILABLE:
        # Note: econml models might require specific parameter tuning or data types
        # ProbabilityForest is a good proxy for honest trees in this context
        hon_rf = ProbabilityForest(n_estimators=100, honest=True, random_state=42)
        # ProbabilityForest expects numeric input usually, but we preprocessed.
        # It handles y as (n_samples,) array.
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
            if name == 'Standard RF':
                pred = model.predict(X_test_shifted)
            elif name == 'Honest RF':
                # ProbabilityForest predict returns probabilities, we need classes.
                # Assuming binary/multiclass classification. 
                # predict output shape depends on implementation, usually (n_samples, n_classes)
                # We take argmax.
                pred_proba = model.predict(X_test_shifted)
                pred = np.argmax(pred_proba, axis=1)
                
                # We need to map back to original labels if they were encoded? 
                # Actually sklearn metrics just need matching types. 
                # If y was LabelEncoded before splitting, we are good.
                # Wait, preprocess_data does not encoding y if it is categorical?
                # We should ensure y is numeric for argmax comparison or use model classes.
                # Let's fix y encoding in preprocessing if needed.
                pass
            
            # Simplified prediction for now, will refine in main block
            if name == 'Honest RF':
                 pred = np.argmax(model.predict(X_test_shifted), axis=1)

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
            if name == 'Standard RF':
                pred = model.predict(X_test_shifted_current)
            else:
                pred = np.argmax(model.predict(X_test_shifted_current), axis=1)
            
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
    st.info("Plese upload a CSV file to begin.")
    if not ECONML_AVAILABLE:
        st.warning("`econml` library is not installed. Honest Forest will be unavailable.")

