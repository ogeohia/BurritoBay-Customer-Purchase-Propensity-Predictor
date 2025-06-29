# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import RandomOverSampler


# Initial data loading
df = pd.read_csv('/content/drive/MyDrive/online_shoppers_intention.csv')

# Initial data split (before feature engineering)
X = df.drop('Revenue', axis=1)
y = df['Revenue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Encoding categorical features for feature importance analysis (original data)
categorical_cols = X.select_dtypes(include=['object', 'bool']).columns
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Data preprocessing for clustering (One-Hot Encoding)
# Identify categorical columns again, excluding 'Revenue'
categorical_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
if 'Revenue' in categorical_cols:
    categorical_cols.remove('Revenue')

# Identify numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Identify the remainder column(s) - in this case, it's 'Revenue'
all_original_cols = df.columns.tolist()
remainder_cols = [col for col in all_original_cols if col not in numerical_cols and col not in categorical_cols]

# Create a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

# Apply preprocessing for clustering
df_processed_array = preprocessor.fit_transform(df)

# Convert to DataFrame for clustering (dropping Revenue)
numerical_feature_names = numerical_cols
categorical_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
all_feature_names = numerical_feature_names + list(categorical_feature_names) + remainder_cols
df_processed = pd.DataFrame(df_processed_array, columns=all_feature_names)

df_features_for_clustering = df_processed.drop('Revenue', axis=1)

# Apply K-Means Clustering (adds 'Cluster' column to the original df)
optimal_k = 3
kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_model.fit(df_features_for_clustering)
df['Cluster'] = kmeans_model.labels_

# Data Preparation with 'Cluster' Feature for modeling
# Separate features (X) and target variable (y) from the DataFrame with the 'Cluster' column
X = df.drop('Revenue', axis=1)
y = df['Revenue']

# Handle categorical features again, including the new 'Cluster' column as a categorical one
categorical_cols_with_cluster = X.select_dtypes(include=['object', 'bool', 'int64']).columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols_to_encode = [col for col in categorical_cols_with_cluster if col not in numerical_cols or col == 'Cluster']

# Apply Label Encoding to categorical columns (including 'Cluster')
X_encoded = X.copy()
for col in categorical_cols_to_encode:
    if col in X_encoded.columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))

# Split data into training and testing sets BEFORE resampling (using the encoded data with Cluster)
X_train_encoded, X_test_encoded, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

# Apply Resampling to the Training Data
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train_encoded, y_train)

# Feature selection is also part of preparing data for modeling, but the selection itself is done later.
# The code to select features from the encoded/resampled data would follow this preprocessing.