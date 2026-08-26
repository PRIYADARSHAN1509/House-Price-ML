"""
DELIVERABLE 31: Clean duplicates, invalid values, and missing values
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("House Price.csv")

print("="*60)
print("DELIVERABLE 31: Clean Duplicates, Invalid Values, and Missing Values")
print("="*60)

df_cleaned = df.copy()

print(f"\n📊 Original dataset: {df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} columns")

changes_log = []


before_dup = len(df_cleaned)
df_cleaned = df_cleaned.drop_duplicates()
after_dup = len(df_cleaned)
dup_removed = before_dup - after_dup
changes_log.append(f"Removed {dup_removed} duplicate rows")
print(f"\n🔄 Before removing duplicates: {before_dup}")
print(f"🔄 After removing duplicates: {after_dup}")
print(f"🔄 Duplicates removed: {dup_removed}")


invalid_conditions = {
    'negative_area': df_cleaned['area_sqft'] < 0,
    'negative_bedrooms': df_cleaned['bedrooms'] < 0,
    'excessive_bedrooms': df_cleaned['bedrooms'] > 10,
    'negative_age': df_cleaned['age_years'] < 0,
    'negative_distance': df_cleaned['distance_city_km'] < 0,
    'negative_price': df_cleaned['price_lakh'] < 0
}

before_invalid = len(df_cleaned)
invalid_mask = pd.DataFrame(invalid_conditions).any(axis=1)
invalid_count = invalid_mask.sum()

print(f"\n🔴 Invalid rows found: {invalid_count}")

for condition, mask in invalid_conditions.items():
    if mask.sum() > 0:
        print(f"  - {condition}: {mask.sum()} rows")

df_cleaned = df_cleaned[~invalid_mask]
after_invalid = len(df_cleaned)
changes_log.append(f"Removed {invalid_count} invalid rows")

print(f"\n✅ After removing invalid values: {after_invalid} rows")

missing_before = df_cleaned.isnull().sum().sum()
print(f"\n❓ Missing values before handling: {missing_before}")

if missing_before > 0:

    print("\nMissing values per column:")
    print(df_cleaned.isnull().sum()[df_cleaned.isnull().sum() > 0])
    
   
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype in ['int64', 'float64']:
           
            median_val = df_cleaned[col].median()
            df_cleaned[col] = df_cleaned[col].fillna(median_val)
            changes_log.append(f"Filled missing values in {col} with median: {median_val}")
    
    missing_after = df_cleaned.isnull().sum().sum()
    print(f"\n✅ Missing values after handling: {missing_after}")
else:
    print("✅ No missing values found")


print("\n📝 Data types before conversion:")
print(df_cleaned.dtypes)

if df_cleaned['bedrooms'].dtype == 'float64':
    df_cleaned['bedrooms'] = df_cleaned['bedrooms'].astype(int)

print(f"\n📊 Final dataset: {df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} columns")
print(f"Total changes made: {len(changes_log)}")

print("\n📋 Changes Summary:")
for i, change in enumerate(changes_log, 1):
    print(f"  {i}. {change}")


fig, axes = plt.subplots(2, 3, figsize=(18, 10))


axes[0, 0].hist(df['area_sqft'], bins=30, alpha=0.5, label='Before', color='red')
axes[0, 0].hist(df_cleaned['area_sqft'], bins=30, alpha=0.5, label='After', color='blue')
axes[0, 0].set_title('Area Distribution - Before vs After')
axes[0, 0].set_xlabel('Area (sqft)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()


axes[0, 1].hist(df['price_lakh'], bins=30, alpha=0.5, label='Before', color='red')
axes[0, 1].hist(df_cleaned['price_lakh'], bins=30, alpha=0.5, label='After', color='blue')
axes[0, 1].set_title('Price Distribution - Before vs After')
axes[0, 1].set_xlabel('Price (Lakhs)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].legend()


axes[0, 2].hist(df['bedrooms'], bins=range(0, 11), alpha=0.5, label='Before', color='red')
axes[0, 2].hist(df_cleaned['bedrooms'], bins=range(0, 11), alpha=0.5, label='After', color='blue')
axes[0, 2].set_title('Bedrooms Distribution - Before vs After')
axes
