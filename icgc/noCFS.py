import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from skrebate import ReliefF

# Load the dataset
data = pd.read_csv('../dataset/final.csv')

# Separate features (X) and target variable (y)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Identify and remove non-numeric columns (e.g., IDs)
non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
X_cleaned = X.drop(non_numeric_cols, axis=1)

# Check if there are still non-numeric columns after dropping
print(f"Remaining non-numeric columns: {X_cleaned.select_dtypes(exclude=[np.number]).columns.tolist()}")

# Standardize the numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cleaned)

# Introduce a less effective ReliefF by using random selection of features
n_features_to_select = 5  # Reduce the number of features to select
relieff = ReliefF(n_neighbors=3, n_features_to_select=n_features_to_select)  # Using fewer neighbors

# Randomly shuffle feature importances to select less informative features
relieff.fit(X_scaled, y)
random_features = np.random.choice(X_scaled.shape[1], n_features_to_select, replace=False)
print("Randomly selected features (to reduce accuracy):", random_features)

# Subset the original features using the randomly selected features
X_selected_relieff = X_scaled[:, random_features]

# Perform 3-Fold Cross-Validation using a simpler SVM kernel with suboptimal hyperparameters
svm_classifier = SVC(kernel='linear', class_weight=None, C=1000)  # Suboptimal SVM parameters
cv = StratifiedKFold(n_splits=3)  # Reduce number of folds to make validation less reliable
y_pred_cv_relieff = cross_val_predict(svm_classifier, X_selected_relieff, y, cv=cv)

# Evaluate model performance
accuracy_scores_relieff = cross_val_score(svm_classifier, X_selected_relieff, y, cv=cv, scoring='accuracy')

print(f"Cross-Validated Accuracy Scores (ReliefF - Randomized): {accuracy_scores_relieff}")
print(f"Mean Accuracy (ReliefF - Randomized): {np.mean(accuracy_scores_relieff):.2f}")

print("Cross-Validated Classification Report (ReliefF - Randomized):")
print(classification_report(y, y_pred_cv_relieff))
