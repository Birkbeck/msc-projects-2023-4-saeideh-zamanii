import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Load the cleaned dataset
data = pd.read_csv('../dataset/cleaned_breast_cancer_data.csv')

# Features and target
X = data.drop(columns=['Person Neoplasm Status'])  # All columns except the target
y = data['Person Neoplasm Status']  # The target column for classification

# Step 1: Check Class Distribution
unique_classes, class_counts = np.unique(y, return_counts=True)
print("Class Distribution before SMOTE:")
for cls, count in zip(unique_classes, class_counts):
    print(f"Class {cls}: {count} instances")

# If class distribution is imbalanced or only one class is present, apply SMOTE
if len(unique_classes) <= 1:
    raise ValueError("The target variable has only one class. Classification requires at least two classes.")
else:
    print("Applying SMOTE to balance the classes...")
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)

# Step 2: Check Class Distribution After SMOTE
unique_classes_balanced, class_counts_balanced = np.unique(y_balanced, return_counts=True)
print("Class Distribution after SMOTE:")
for cls, count in zip(unique_classes_balanced, class_counts_balanced):
    print(f"Class {cls}: {count} instances")

# Step 3: Standardize the Features
scaler = StandardScaler()
X_balanced = scaler.fit_transform(X_balanced)

# Step 4: Set Up SVM Classifier
svm_classifier = SVC(kernel='rbf', class_weight='balanced', C=0.1)

# Step 5: Cross-validation Setup
cv = StratifiedKFold(n_splits=5)

# Step 6: Cross-Validated Predictions
y_pred_cv = cross_val_predict(svm_classifier, X_balanced, y_balanced, cv=cv)

# Step 7: Accuracy Scores Using Cross-Validation
accuracy_scores = cross_val_score(svm_classifier, X_balanced, y_balanced, cv=cv, scoring='accuracy')

# Step 8: Print Results
print(f"Cross-Validated Accuracy Scores: {accuracy_scores}")
print(f"Mean Accuracy: {np.mean(accuracy_scores):.2f}")

# Step 9: Classification Report
print("Cross-Validated Classification Report:")
print(classification_report(y_balanced, y_pred_cv))
