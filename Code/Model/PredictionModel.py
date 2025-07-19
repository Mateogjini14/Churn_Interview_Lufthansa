import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import numpy as np
df = pd.read_csv("../../Dataset/telecom_churn.csv")
df['data_used'] = df['data_used'].abs()
df['calls_made'] = df['calls_made'].abs()
df['sms_sent'] = df['sms_sent'].abs()
df_encoded = df.copy()
for col in df_encoded.columns:
    if df_encoded[col].dtype == 'object':
        df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])
X = df_encoded.drop('churn', axis=1)
y = df_encoded['churn']
smote_tomek = SMOTETomek(random_state=42)
X_balanced, y_balanced = smote_tomek.fit_resample(X, y)
x_train, x_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)
numeric_cols = ['age', 'estimated_salary', 'data_used', 'calls_made', 'sms_sent']
scaler = MinMaxScaler()
x_train[numeric_cols] = scaler.fit_transform(x_train[numeric_cols])
x_test[numeric_cols] = scaler.transform(x_test[numeric_cols])
models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=2000),
    'Random Forest': RandomForestClassifier(class_weight='balanced'),
    'Naive Bayes': GaussianNB(),
    'XGBoost': XGBClassifier(scale_pos_weight=4, eval_metric='logloss')
}
accuracy_results = {}
classification_reports = {}
roc_auc_scores = {}
for model_name, model in tqdm(models.items(), desc="Fitting models"):
    model.fit(x_train, y_train)
    y_test_pred = model.predict(x_test)
    accuracy_results[model_name] = accuracy_score(y_test, y_test_pred)
    classification_reports[model_name] = classification_report(y_test, y_test_pred)
for model_name, model in models.items():
    if hasattr(model, "predict_proba"):
        y_probs = model.predict_proba(x_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        roc_auc_scores[model_name] = roc_auc
    else:
        roc_auc_scores[model_name] = None
for model_name in models.keys():
    print(f"--- Classification Report for {model_name} ---")
    print(classification_reports[model_name])
    print(f'\n{model_name} Accuracy: {accuracy_results[model_name]:.4f}')
    if roc_auc_scores[model_name] is not None:
        print(f"\n{model_name} ROC AUC: {roc_auc_scores[model_name]:.4f}")
    else:
        print(f"\n{model_name} ROC AUC: Not available")
for model_name, model in models.items():
    if hasattr(model, "predict_proba"):
        y_probs = model.predict_proba(x_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {model_name}")
        plt.legend()
        plt.grid(True)
        plt.show()
        print("\n")
for model_name, model in models.items():
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()
    print("\n")
for model_name, model in models.items():
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = x_train.columns
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(12, 6 + len(feature_names) * 0.2))
        plt.title(f"Feature Importances - {model_name}")
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.ylabel("Importance")
        plt.tight_layout()
        plt.show()
        print("\n")