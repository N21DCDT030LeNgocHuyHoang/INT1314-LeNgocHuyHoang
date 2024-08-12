# Import libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("H:/NAM 4/HK1-2025/01_NMTTNT_Nhap mon tri tue nhan tao/BUOI 1 - 12-08/heart_disease_prediction.csv")


# Display the first few rows and check data types
print(df.head())
print(df.dtypes)
print(df.dtypes.value_counts())
print(df.describe())
print(df.isna().sum())

# Describe categorical data
print(df.describe(include=['object']))
print(df["FastingBS"].unique(), df["HeartDisease"].unique())

# Define categorical columns
categorical_cols = ["Sex", "ChestPainType", "FastingBS", "RestingECG", "ExerciseAngina", "ST_Slope", "HeartDisease"]

# Plot count plots for categorical features
fig = plt.figure(figsize=(16, 15))
for idx, col in enumerate(categorical_cols):
    ax = plt.subplot(4, 2, idx+1)
    sns.countplot(x=df[col], ax=ax)
    # Add data labels to each bar
    for container in ax.containers:
        ax.bar_label(container, label_type="center")

fig = plt.figure(figsize=(16, 15))
for idx, col in enumerate(categorical_cols[:-1]):
    ax = plt.subplot(4, 2, idx+1)
    sns.countplot(x=df[col], hue=df["HeartDisease"], ax=ax)
    # Add data labels to each bar
    for container in ax.containers:
        ax.bar_label(container, label_type="center")

# Clean the dataset by removing rows with RestingBP = 0
df_clean = df.copy()
df_clean = df_clean[df_clean["RestingBP"] != 0]

# Handle Cholesterol = 0 based on HeartDisease status
heartdisease_mask = df_clean["HeartDisease"] == 0
cholesterol_without_heartdisease = df_clean.loc[heartdisease_mask, "Cholesterol"]
cholesterol_with_heartdisease = df_clean.loc[~heartdisease_mask, "Cholesterol"]

df_clean.loc[heartdisease_mask, "Cholesterol"] = cholesterol_without_heartdisease.replace(to_replace=0, value=cholesterol_without_heartdisease.median())
df_clean.loc[~heartdisease_mask, "Cholesterol"] = cholesterol_with_heartdisease.replace(to_replace=0, value=cholesterol_with_heartdisease.median())

print(df_clean[["Cholesterol", "RestingBP"]].describe())

# Convert categorical variables to dummy variables
df_clean = pd.get_dummies(df_clean, drop_first=True)
print(df_clean.head())

# Plot correlation heatmap
correlations = abs(df_clean.corr())
plt.figure(figsize=(12, 8))
sns.heatmap(correlations, annot=True, cmap="Blues")

plt.figure(figsize=(12, 8))
sns.heatmap(correlations[correlations > 0.3], annot=True, cmap="Blues")

# Split the data into features and target
X = df_clean.drop(["HeartDisease"], axis=1)
y = df_clean["HeartDisease"]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=417)

# Define selected features
features = [
    "Oldpeak",
    "Sex_M",
    "ExerciseAngina_Y",
    "ST_Slope_Flat",
    "ST_Slope_Up"
]

# Train k-NN on individual features and evaluate accuracy
for feature in features:
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train[[feature]], y_train)
    accuracy = knn.score(X_val[[feature]], y_val)
    print(f"The k-NN classifier trained on {feature} and with k = 3 has an accuracy of {accuracy*100:.2f}%")

# Scale the features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train[features])
X_val_scaled = scaler.transform(X_val[features])

# Train k-NN on the scaled features
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)
accuracy = knn.score(X_val_scaled, y_val)
print(f"Accuracy: {accuracy*100:.2f}")

# Re-split data for model optimization
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=417)

# Scale the training features
X_train_scaled = scaler.fit_transform(X_train[features])

# Define hyperparameter grid for GridSearchCV
grid_params = {
    "n_neighbors": range(1, 20),
    "metric": ["minkowski", "manhattan"]
}

# Perform grid search to find the best hyperparameters
knn = KNeighborsClassifier()
knn_grid = GridSearchCV(knn, grid_params, scoring='accuracy')
knn_grid.fit(X_train_scaled, y_train)
print(f"Best Grid Search Score: {knn_grid.best_score_*100:.2f}%")
print(f"Best Grid Search Parameters: {knn_grid.best_params_}")

# Evaluate model on test set with the best hyperparameters
X_test_scaled = scaler.transform(X_test[features])
predictions = knn_grid.best_estimator_.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy on test set: {accuracy*100:.2f}%")

# Distribution of patients by sex
print("Distribution of patients by their sex in the entire dataset")
print(X.Sex_M.value_counts())

print("\nDistribution of patients by their sex in the training dataset")
print(X_train.Sex_M.value_counts())

print("\nDistribution of patients by their sex in the test dataset")
print(X_test.Sex_M.value_counts())
