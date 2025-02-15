import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
import joblib


# Load the dataset
dataset = pd.read_csv('data/insurance.csv')

# Data Preprocessing
# Encode 'sex' and 'smoker'
dataset['sex'] = dataset['sex'].apply(lambda x: 0 if x == 'female' else 1)
dataset['smoker'] = dataset['smoker'].apply(lambda x: 0 if x == 'no' else 1)

# One-hot for 'region'
region_dummies = pd.get_dummies(dataset['region'], drop_first=True)
dataset = pd.concat([region_dummies, dataset], axis=1)
dataset.drop(['region'], axis=1, inplace=True)

# Splitting data into features and target
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Train-Test split
X_train, X_test, y_train, y_test = (
    train_test_split(X, y, test_size=0.2, random_state=42))

# Initialize and train the XGBoost Regressor
model = xgb.XGBRegressor(max_depth=2, learning_rate=0.1, n_estimators=100)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R2 Score: {r2:.3f}")

# Cross-validation (optional)
# r2_scores = cross_val_score(estimator=model, X=X, y=y, scoring='r2', cv=10)
# print(f"Average R2 Score: {r2_scores.mean():.3f}")
# print(f"Standard Deviation: {r2_scores.std():.3f}")

# Save the trained model as a .pkl file
joblib.dump(model, 'model/model.pkl')
print("Model saved as 'model.pkl'")



