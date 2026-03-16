import pandas as pd
import numpy as np

# Load the dataset
print("Loading star_classification.csv...")
df = pd.read_csv('star_classification.csv')

# Reduce size to 10K random samples
if len(df) > 10000:
    print(f"Sampling 10,000 random rows from {len(df)} total rows...")
    df = df.sample(n=10000, random_state=42)

# Identify the target column dynamically or default to 'class' if it exists
target_col = 'class' if 'class' in df.columns else df.columns[-1]

# Check if it's continuous
if pd.api.types.is_numeric_dtype(df[target_col]) and df[target_col].nunique() > 20:
    print(f"Target column '{target_col}' is continuous. Binarizing at median...")
    median_val = df[target_col].median()
    # Convert to 0 or 1 based on median
    df[target_col] = (df[target_col] > median_val).astype(int)
    
# Save the cleaned dataset
output_file = 'star_classification_cleaned.csv'
df.to_csv(output_file, index=False)
print(f"Dataset successfully processed and saved as '{output_file}' with shape {df.shape}.")
