import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder

# Load the dataset (assuming it's in CSV/TSV format)
data = pd.read_csv('../dataset/breast.tsv', delimiter='\t')

# Step 1: Handle Missing Values
# Fill missing numeric columns with the median value
numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

# Fill missing categorical columns with 'missing' or a placeholder
non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns
data[non_numeric_cols] = data[non_numeric_cols].fillna('missing')

# Step 2: Encoding Categorical Columns
# Use LabelEncoder to convert categorical columns to numerical values
for col in non_numeric_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))

# Step 3: Discretize Numeric Columns
# Discretize numeric columns into bins
discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
X_numeric_discretized = discretizer.fit_transform(data[numeric_cols])

# Step 4: Combine discretized numeric columns with non-numeric columns
X_final = np.hstack([X_numeric_discretized, data[non_numeric_cols].values])

# Step 5: Save the cleaned dataset
X_final_df = pd.DataFrame(X_final, columns=list(numeric_cols) + list(non_numeric_cols))

# You might need to specify your target variable. Assuming 'Person Neoplasm Status' is your target:
y = data['Person Neoplasm Status']

# Save the final dataset to CSV (without index)
X_final_df.to_csv('../dataset/cleaned_breast_cancer_data.csv', index=False)

print("Data cleaning and preparation complete. Ready for CFS.")
