
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load datasets (assumes files are in same directory)
features_df = pd.read_csv('features.csv')
stores_df = pd.read_csv('stores.csv')
train_df = pd.read_csv('train.csv')

# Preprocessing
features_df['Date'] = pd.to_datetime(features_df['Date'])
train_df['Date'] = pd.to_datetime(train_df['Date'])

df = train_df.merge(features_df, on=['Store', 'Date', 'IsHoliday'], how='left')
df = df.merge(stores_df, on='Store', how='left')

markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
df[markdown_cols] = df[markdown_cols].fillna(0)
df['Type'] = df['Type'].map({'A': 0, 'B': 1, 'C': 2})
df['IsHoliday'] = df['IsHoliday'].astype(int)

features = ['Store', 'Dept', 'Type', 'Size', 'Temperature', 'Fuel_Price',
            'CPI', 'Unemployment', 'IsHoliday'] + markdown_cols
target = 'Weekly_Sales'

X = df[features]
y = df[target]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Use subset for training to avoid memory issues
X_small = X_train.sample(n=20000, random_state=42)
y_small = y_train.loc[X_small.index]

model = RandomForestRegressor(n_estimators=10, random_state=42)
model.fit(X_small, y_small)

y_pred = model.predict(X_val[:5000])
rmse = np.sqrt(mean_squared_error(y_val[:5000], y_pred))
print(f"Validation RMSE: {rmse:.2f}")
