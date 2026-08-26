"""
DELIVERABLE 25: Find missing values and duplicate rows
"""
import pandas as pd
import numpy as np

df = pd.read_csv("House Price.csv")

print("="*60)
print("DELIVERABLE 25: Missing Values and Duplicate Rows")
print("="*60)

print("\n🔍 Missing Values per Column:")
missing_values = df.isnull().sum()
print(missing_values)

print(f"\n📊 Total Missing Values: {df.isnull().sum().sum()}")

if df.isnull().sum().sum() > 0:
    print("\n❌ Columns with Missing Values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    
    print("\n📄 Rows with Missing Values:")
    print(df[df.isnull().any(axis=1)])
else:
    print("\n✅ No missing values found!")

print(f"\n🔄 Number of Duplicate Rows: {df.duplicated().sum()}")

if df.duplicated().sum() > 0:
    print("\n📄 Duplicate Rows:")
    print(df[df.duplicated()])
    
    duplicate_counts = df[df.duplicated(keep=False)].groupby(list(df.columns)).size()
    print("\n📊 Duplicate Row Counts:")
    print(duplicate_counts)
else:
    print("\n✅ No duplicate rows found!")

print("\n✅ Deliverable 25 completed!")