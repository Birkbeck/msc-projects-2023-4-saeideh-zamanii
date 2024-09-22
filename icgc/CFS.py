import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, make_scorer
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from ..module.mutual_information import su_calculation


data = pd.read_csv('../dataset/final.csv')

X = data.iloc[:, :-1]  
y = data.iloc[:, -1]   

X = X.values
y = y.values

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

selected_features = cfs(X, y)
print("Selected features:", selected_features)

X_selected = X[:, selected_features]

scaler = StandardScaler()
X_selected = scaler.fit_transform(X_selected)

svm_classifier = SVC(kernel='rbf', class_weight='balanced', C=0.1)  

cv = StratifiedKFold(n_splits=5)

y_pred_cv = cross_val_predict(svm_classifier, X_selected, y, cv=cv)

accuracy_scores = cross_val_score(svm_classifier, X_selected, y, cv=cv, scoring='accuracy')

print(f"Cross-Validated Accuracy Scores: {accuracy_scores}")
print(f"Mean Accuracy: {np.mean(accuracy_scores):.2f}")

print("Cross-Validated Classification Report:")
print(classification_report(y, y_pred_cv))

