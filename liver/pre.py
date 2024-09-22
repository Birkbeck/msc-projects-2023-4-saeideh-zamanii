import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.impute import SimpleImputer

# Load the dataset (assuming it's in CSV or TSV format)
data = pd.read_csv('../dataset/liver.tsv', delimiter='\t')

# Step 1: Remove columns with only missing values
# Drop columns where all values are NaN
data = data.dropna(axis=1, how='all')

# Step 2: Handle missing values
# Fill missing numeric columns with the median value
numeric_cols = data.select_dtypes(include=[np.number]).columns

# Use SimpleImputer to fill missing numeric values
imputer = SimpleImputer(strategy='median')
data[numeric_cols] = imputer.fit_transform(data[numeric_cols])

# Fill missing categorical columns with 'missing' or a placeholder
non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns
data[non_numeric_cols] = data[non_numeric_cols].fillna('missing')

# Step 3: Encoding Categorical Columns
# Convert all categorical columns to numeric using LabelEncoder
for col in non_numeric_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))

# Step 4: Discretize Numeric Columns
# Now, discretize numeric columns after ensuring no NaNs are present
discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
X_numeric_discretized = discretizer.fit_transform(data[numeric_cols])

# Step 5: Combine discretized numeric columns with encoded non-numeric columns
X_final = np.hstack([X_numeric_discretized, data[non_numeric_cols].values])

# Step 6: Save the cleaned dataset
X_final_df = pd.DataFrame(X_final, columns=list(numeric_cols) + list(non_numeric_cols))

# Assume 'Person Neoplasm Status' or another column is your target variable for classification
# If 'Person Neoplasm Status' is your target column, do this:
y = data['Person Neoplasm Status']

# Save the final cleaned data
X_final_df.to_csv('cleaned_liver_cancer_data.csv', index=False)

print("Data cleaning complete. Dataset is ready for CFS.")
