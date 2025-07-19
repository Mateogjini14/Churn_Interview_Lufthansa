import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
df = pd.read_csv("../../Dataset/telecom_churn.csv")
print(tabulate(df.head(), headers='keys', tablefmt='pretty'))
print(df.shape)
print(tabulate(df.describe(include="all").T, headers='keys', tablefmt='pretty'))
print(df.info())
print(df.isna().sum())
print(df.duplicated().sum())
churned_count = df['churn'].value_counts()
print(df['churn'].value_counts())
print(df['churn'].value_counts(normalize=True) * 100)
ratio = churned_count[1]/churned_count[0]
print(f"Churn-to-Retention Ratio: {ratio:.2f}")
sns.set_theme(style='whitegrid')
features = ['age', 'estimated_salary', 'data_used', 'calls_made', 'sms_sent']
for i, feature in enumerate(features):
    plt.figure(figsize=(10, 5))
    sns.kdeplot(df[feature], color='skyblue',linewidth=2)
    plt.title(f"Distribution of {feature}", fontsize=12)
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.show()
    print("\n")
df['data_used_bin'] = pd.cut(df['data_used'], bins=30)
df['sms_sent_bin'] = pd.cut(df['sms_sent'], bins=30)
df['calls_made_bin'] = pd.cut(df['calls_made'], bins=30)
df['estimated_salary_bin'] = pd.cut(df['estimated_salary'], bins=30)
features = ['telecom_partner','gender','state','city','estimated_salary_bin', 'age', 'data_used_bin', 'sms_sent_bin', 'calls_made_bin']
for feature in features:
    churn_rate = (df.groupby(feature)['churn'].mean() * 100).sort_values(ascending=False)
    if feature == 'age':
      plt.figure(figsize=(29, 5))
    else:
      plt.figure(figsize=(17, 5))
    ax = sns.barplot(x=churn_rate.index, y=churn_rate.values, palette='coolwarm')
    plt.title(f'Churn Rate by {feature}', fontsize=14)
    plt.ylabel("Churn Rate %")
    plt.xlabel(feature)
    plt.xticks(rotation=45)
    plt.ylim(0, 25)
    for p in ax.patches:
        height = p.get_height()
        if not pd.isna(height):
           ax.annotate(f'{height:.1f}%', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom', fontsize=11, color='black', xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()
    plt.show()
    print("\n")
features = ['telecom_partner', 'gender', 'city', 'state']
for feature in features:
    if feature == 'state':
        plt.figure(figsize=(20, 7))
    else:
        plt.figure(figsize=(9, 7))
    ax = sns.countplot(data=df, x=feature, palette='coolwarm')
    plt.xticks(rotation=45)
    plt.title(f"Customer Count by {feature}")
    plt.xlabel(feature)
    plt.ylabel("Count")
    for p in ax.patches:
        height = p.get_height()
        if not pd.isna(height):
            ax.annotate(f'{int(height)}',(p.get_x() + p.get_width() / 2., height), ha='center', va='bottom', fontsize=11, color='black', xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()
    plt.show()
    print("\n")
sns.set_theme(style='whitegrid')
features = ['age', 'estimated_salary', 'data_used', 'calls_made', 'sms_sent']
for feature in features:
    plt.figure(figsize=(10, 5))
    sns.histplot(df[feature], bins=30, color='skyblue')
    plt.title(f"Distribution of {feature}", fontsize=12)
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.show()
    print("\n")
numeric_cols = ['age','num_dependents', 'estimated_salary', 'calls_made', 'sms_sent', 'data_used', 'churn']
for col in numeric_cols[:-1]:
    plt.figure(figsize=(6, 5))
    ax = sns.boxplot(x='churn', y=col, data=df, palette='coolwarm')
    plt.title(f"Distribution of {col.title()} by Churn", fontsize=13)
    plt.xlabel("Churn (0 = No, 1 = Yes)", fontsize=11)
    plt.ylabel(col.title(), fontsize=11)
    plt.tight_layout()
    plt.show()
    print("\n")
plt.figure(figsize=(17, 8))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".5f")
plt.title("Correlation Heatmap of Numerical Features")
plt.show()
df['data_used'] = df['data_used'].abs()
df['calls_made'] = df['calls_made'].abs()
df['sms_sent'] = df['sms_sent'].abs()
print("\n")
print(tabulate(df.head(), headers='keys', tablefmt='pretty'))
print("\n")
print(tabulate(df.describe(include="all").T, headers='keys', tablefmt='pretty'))
print("\n")
print("Number of duplicated values", df.duplicated().sum())
print("\n")
sns.set_theme(style='whitegrid')
features = ['age', 'estimated_salary', 'data_used', 'calls_made', 'sms_sent']
for feature in features:
    plt.figure(figsize=(10, 5))
    sns.kdeplot(df[feature], color='skyblue', linewidth=2)
    plt.title(f"Distribution of {feature}", fontsize=12)
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.show()
    print("\n")
for feature in features:
    plt.figure(figsize=(10, 5))
    sns.histplot(df[feature], bins=30, color='skyblue')
    plt.title(f"Distribution of {feature}", fontsize=12)
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.show()
    print("\n")
for col in numeric_cols[:-1]:
    plt.figure(figsize=(6, 5))
    ax = sns.boxplot(x='churn', y=col, data=df, palette='coolwarm')
    plt.title(f"Distribution of {col.title()} by Churn", fontsize=13)
    plt.xlabel("Churn (0 = No, 1 = Yes)", fontsize=11)
    plt.ylabel(col.title(), fontsize=11)
    plt.tight_layout()
    plt.show()
    print("\n")
plt.figure(figsize=(15, 8))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".5f")
plt.title("Correlation Heatmap of Numerical Features")
plt.show()