import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, classification_report

# Load the dataset
df = pd.read_csv('c:/Users/shriv/AppData/Roaming/Python/Python312/site-packages/AI/Pokemon.csv')

# Inspect dataset
print(df.head())

# Drop irrelevant columns
drop_columns = ['#']
df = df.drop(columns=drop_columns, errors='ignore')

# Convert categorical data
label_encoder = LabelEncoder()
df['Type 1'] = label_encoder.fit_transform(df['Type 1'])
df['Type 2'] = df['Type 2'].fillna('None')
df['Type 2'] = label_encoder.fit_transform(df['Type 2'])
df['Legendary'] = df['Legendary'].astype(int)

# **Creating Target Variable (Mega Evolution Classification)**
# Manually label Pokémon as Mega Evolution based on their Name
df['Mega_Evolution'] = df['Name'].apply(lambda x: 1 if 'Mega' in x else 0)

# Define features and target
feature_cols = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Total', 'Legendary']
X = df[feature_cols]
y = df['Mega_Evolution']

# Print first few values of y to verify
print("First few values of y:", y.head())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.2f}')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('roc_curve.png')
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_prob)
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.savefig('precision_recall_curve.png')
plt.show()

# Save final predictions
df_output = pd.read_csv('c:/Users/shriv/AppData/Roaming/Python/Python312/site-packages/AI/Pokemon.csv')
df_output['Mega_Evolution'] = model.predict(scaler.transform(X))
df_output['Mega_Evolution'] = df_output['Mega_Evolution'].apply(lambda x: 'Yes' if x == 1 else 'No')
df_output[['Name', 'Mega_Evolution']].to_csv('c:/Users/shriv/AppData/Roaming/Python/Python312/site-packages/AI/pokemon_mega_predictions.csv', index=False)
print(df_output.head())
print("Predictions saved to pokemon_mega_predictions.csv")
