import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from ..module.mutual_information import su_calculation  # Ensure this is implemented correctly

# Load the cleaned dataset
data = pd.read_csv('../dataset/cleaned_liver_cancer_data.csv')

# Features and target (assuming 'Person Neoplasm Status' is the target column)
X = data.drop(columns=['Person Neoplasm Status'])
y = data['Person Neoplasm Status']

# Convert to NumPy arrays
X = X.values
y = y.values

# Define merit calculation function (Symmetrical Uncertainty or SU calculation)
def merit_calculation(X, y):
    n_samples, n_features = X.shape
    rff = 0
    rcf = 0
    for i in range(n_features):
        fi = X[:, i]
        rcf += su_calculation(fi, y)  # Symmetrical uncertainty between feature and target
        for j in range(n_features):
            if j > i:
                fj = X[:, j]
                rff += su_calculation(fi, fj)  # Symmetrical uncertainty between features
    rff *= 2
    merits = rcf / np.sqrt(n_features + rff)
    return merits

# CFS feature selection function
def cfs(X, y):
    n_samples, n_features = X.shape
    F = []
    M = []  # Merit values
    while True:
        merit = -100000000000
        idx = -1
        for i in range(n_features):
            if i not in F:
                F.append(i)
                t = merit_calculation(X[:, F], y)  # Calculate merit of the current feature subset
                if t > merit:
                    merit = t
                    idx = i
                F.pop()
        F.append(idx)
        M.append(merit)
        # Stopping condition: Stop if merit does not improve over 5 iterations
        if len(M) > 5:
            if M[-1] <= M[-2] and M[-2] <= M[-3] and M[-3] <= M[-4] and M[-4] <= M[-5]:
                break
    return np.array(F)

# Step 1: Apply CFS to select features
selected_features = cfs(X, y)
print("Selected features:", selected_features)

# Step 2: Subset the features based on CFS selection
X_selected = X[:, selected_features]

# Step 3: Standardize the selected features
scaler = StandardScaler()
X_selected = scaler.fit_transform(X_selected)

# Step 4: SVM Classifier with Cross-Validation
svm_classifier = SVC(kernel='rbf', class_weight='balanced', C=0.1)

# Step 5: Cross-validation setup
cv = StratifiedKFold(n_splits=5)

# Step 6: Cross-validated predictions
y_pred_cv = cross_val_predict(svm_classifier, X_selected, y, cv=cv)

# Step 7: Accuracy scores using cross-validation
accuracy_scores = cross_val_score(svm_classifier, X_selected, y, cv=cv, scoring='accuracy')

# Step 8: Print results
print(f"Cross-Validated Accuracy Scores: {accuracy_scores}")
print(f"Mean Accuracy: {np.mean(accuracy_scores):.2f}")

# Classification report
print("Cross-Validated Classification Report:")
print(classification_report(y, y_pred_cv))
