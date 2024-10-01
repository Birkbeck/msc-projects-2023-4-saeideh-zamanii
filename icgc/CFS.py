import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from mutual_information import su_calculation

# Load the dataset
data = pd.read_csv('../dataset/final.csv')

# Separate features (X) and target variable (y)
X = data.iloc[:, :-1]  
y = data.iloc[:, -1]   

X = X.values
y = y.values

# Merit calculation function
def merit_calculation(X, y):
    n_samples, n_features = X.shape
    rff = 0
    rcf = 0
    for i in range(n_features):
        fi = X[:, i]
        rcf += su_calculation(fi, y)  
        for j in range(n_features):
            if j > i:
                fj = X[:, j]
                rff += su_calculation(fi, fj)
    rff *= 2
    merits = rcf / np.sqrt(n_features + rff)
    return merits

# Correlation-based feature selection (CFS) function
def cfs(X, y):
    n_samples, n_features = X.shape
    F = []
    M = []  
    while True:
        merit = -100000000000
        idx = -1
        for i in range(n_features):
            if i not in F:
                F.append(i)
                t = merit_calculation(X[:, F], y)  
                if t > merit:
                    merit = t
                    idx = i
                F.pop()
        F.append(idx)
        M.append(merit)
        if len(M) > 5:
            if M[-1] <= M[-2] and M[-2] <= M[-3] and M[-3] <= M[-4] and M[-4] <= M[-5]:
                break
    return np.array(F)

# Apply CFS to select features
selected_features = cfs(X, y)

# Subset the original features using the selected indices
X_selected = X[:, selected_features]

# Standardize the selected features
scaler = StandardScaler()
X_selected = scaler.fit_transform(X_selected)

# Initialize SVM Classifier
svm_classifier = SVC(kernel='rbf', class_weight='balanced', C=0.1)  

# Perform 5-Fold Cross-Validation
cv = StratifiedKFold(n_splits=5)
y_pred_cv = cross_val_predict(svm_classifier, X_selected, y, cv=cv)

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
