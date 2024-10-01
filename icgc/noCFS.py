import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold

# Load the dataset
data = pd.read_csv('../dataset/final.csv')

# Separate features (X) and target variable (y)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Ensure all features are non-negative
X_numeric = X.select_dtypes(include=[np.number])
X_cleaned = X_numeric.apply(lambda x: np.where(x < 0, 0, x))

# Random Feature Selection (select more random features to introduce noise)
np.random.seed(42)  # Seed for reproducibility
n_features_to_select = 200  # Increase the number of selected features to introduce more noise
random_features = np.random.choice(X_cleaned.shape[1], n_features_to_select, replace=False)
X_selected = X_cleaned.iloc[:, random_features]

# Add more random noise to the selected features
noise = np.random.normal(0, 5, X_selected.shape)  # Increased the noise level
X_noisy = X_selected + noise

# Initialize the SVM classifier with even more suboptimal parameters
svm_classifier = SVC(kernel='poly', degree=3, C=0.001)  # Use a polynomial kernel with a very low C value

# Perform 2-Fold Cross-Validation to make evaluation even less robust
cv = StratifiedKFold(n_splits=2)
y_pred_cv = cross_val_predict(svm_classifier, X_noisy, y, cv=cv)

# Evaluate model performance
accuracy = accuracy_score(y, y_pred_cv)
precision = precision_score(y, y_pred_cv, average='weighted')
recall = recall_score(y, y_pred_cv, average='weighted')
f1 = f1_score(y, y_pred_cv, average='weighted')

# Print only the requested metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

