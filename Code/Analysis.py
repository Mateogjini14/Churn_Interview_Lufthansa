import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

pd.set_option('display.max_columns', None)
df = pd.read_csv("../Dataset/telecom_churn.csv")

print("="*100)
print("OUTLOOK OF TABLE CONTENT".center(100))
print("="*100)
print(tabulate(df.head(), headers='keys', tablefmt='pretty'))

print("\n"*2)
print("="*100)
print("DATA SHAPE".center(100))
print("="*100)
print(df.shape)


print("\n"*2)
print("="*100)
print("DATA DESCRIPTION".center(100))
print("="*100)
print(tabulate(df.describe(include="all").T, headers='keys', tablefmt='pretty'))


print("\n"*2)
print("="*100)
print("DATA INFO".center(100))
print("="*100)
print(df.info())

print("\n"*2)
print("="*100)
print("CHECK FOR NULL VALUES".center(100))
print("="*100)
print(df.isna().sum())


print("\n"*2)
print("="*100)
print("CHECK FOR DUPLICATED VALUES".center(100))
print("="*100)
print(df.duplicated().sum())


print("\n"*2)
print("="*100)
print("CHURN CLASS IMBALANCE".center(100))
print("="*100)
print(df['churn'].value_counts())
print("\n")
print(df['churn'].value_counts(normalize=True) * 100)
