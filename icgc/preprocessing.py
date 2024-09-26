import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

data = pd.read_csv('../dataset/icgc_train_data.csv')

numeric_cols = data.select_dtypes(include=[np.number]).columns
non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns

data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
data[non_numeric_cols] = data[non_numeric_cols].fillna('missing')  

discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
X_numeric_discretized = discretizer.fit_transform(data[numeric_cols])

X_final = np.hstack([X_numeric_discretized, data[non_numeric_cols].values])


y = data.iloc[:, -1]

X_final_df = pd.DataFrame(X_final, columns=list(numeric_cols) + list(non_numeric_cols))

final_data = pd.concat([X_final_df, y.reset_index(drop=True)], axis=1)

final_data.to_csv('final.csv', index=False)

