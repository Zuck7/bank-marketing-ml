"""
Preprocess bank-full.csv for TFX pipeline.
- Converts semicolon delimiter to comma
- Converts label y: yes/no → 1/0
- Saves to data/ folder for CsvExampleGen
"""
import pandas as pd
import os

# Read the semicolon-delimited CSV
df = pd.read_csv('bank-full.csv', sep=';')

# Convert label to numeric (required for TFX binary classification)
df['y'] = df['y'].map({'yes': 1, 'no': 0})

# Verify conversion
print(f"Shape: {df.shape}")
print(f"Label distribution:\n{df['y'].value_counts()}")
print(f"Any nulls: {df.isnull().any().any()}")

# Save as comma-separated CSV
os.makedirs('data', exist_ok=True)
df.to_csv('data/bank_marketing.csv', index=False)

print(f"\nSaved to data/bank_marketing.csv")
