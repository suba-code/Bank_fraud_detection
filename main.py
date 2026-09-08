import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load
df = pd.read_csv("transactionscopy.csv")
print(f"Rows: {len(df)}")
print(df['Class'].value_counts())

# Scale Time and Amount properly
scaler_time = StandardScaler()
scaler_amount = StandardScaler()
df['Time_scaled'] = scaler_time.fit_transform(df[['Time']])
df['Amount_scaled'] = scaler_amount.fit_transform(df[['Amount']])

# Prepare X, y
feature_cols = ['Time_scaled', 'Amount_scaled'] + [f'V{i}' for i in range(1, 29)]
X = df[feature_cols]
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred))

# Graph 1
plt.figure()
sns.countplot(x='Class', data=df)
plt.title('Class Distribution 0=Normal 1=Fraud')
plt.savefig('class_distribution.png')
plt.close()

# Graph 2
plt.figure()
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()

# Graph 3
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
plt.figure(figsize=(8,5))
importances.plot(kind='bar')
plt.title('Top 10 Features')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

print("\nDONE! 3 images saved in your project folder.")
print("Check left side - PythonProject4 folder for .png files")