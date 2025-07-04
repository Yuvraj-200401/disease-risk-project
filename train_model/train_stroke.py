import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv('../dataset/stroke.csv')

# Drop unnecessary columns
df = df.drop(['id'], axis=1)
df = df.dropna()  # or use imputation

# One-hot encode categorical
df = pd.get_dummies(df)

X = df.drop('stroke', axis=1)
y = df['stroke']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, '../backend/model/stroke_model.pkl')
print("✅ Stroke model saved.")
