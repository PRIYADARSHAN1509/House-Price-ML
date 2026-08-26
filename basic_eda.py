"""
DELIVERABLE 24: Perform basic EDA and inspect data types
"""
import pandas as pd
import numpy as np


df = pd.read_csv("House Price.csv")

print("="*60)
print("DELIVERABLE 24: Basic EDA and Data Types")
print("="*60)


print(f"\n📊 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")

print("\n📋 Column Names:")
print(df.columns.tolist())


print("\n📝 Data Types:")
print(df.dtypes)


print("\n👀 First 5 rows:")
print(df.head())


print("\n👀 Last 5 rows:")
print(df.tail())

# Summary statistics
print("\n📈 Summary Statistics:")
print(df.describe())

print("\nℹ️ Dataset Info:")
print(df.info())

print(f"\n💾 Memory Usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

print("\n✅ Deliverable 24 completed!")
