"""
DELIVERABLE 26: Identify impossible values such as negative area and unreasonable bedroom counts
"""
import pandas as pd
import numpy as np


df = pd.read_csv("House Price.csv")

print("="*60)
print("DELIVERABLE 26: Identify Impossible Values")
print("="*60)

print("\n🏠 Checking for Impossible Values:")

print("\n🔴 Negative Values Check:")
negative_checks = {
    'area_sqft': (df['area_sqft'] < 0, "negative area"),
    'bedrooms': (df['bedrooms'] < 0, "negative bedrooms"),
    'age_years': (df['age_years'] < 0, "negative age"),
    'distance_city_km': (df['distance_city_km'] < 0, "negative distance"),
    'price_lakh': (df['price_lakh'] < 0, "negative price")
}

for col, (mask, desc) in negative_checks.items():
    count = mask.sum()
    if count > 0:
        print(f"  ❌ {col}: {count} rows with {desc}")
        print(f"     Values: {df[mask][col].tolist()}")
    else:
        print(f"  ✅ {col}: No {desc} values")


print("\n🔵 Zero Values Check:")
zero_checks = {
    'area_sqft': (df['area_sqft'] == 0, "zero area"),
    'bedrooms': (df['bedrooms'] == 0, "zero bedrooms"),
    'age_years': (df['age_years'] == 0, "zero age"),
    'price_lakh': (df['price_lakh'] == 0, "zero price")
}

for col, (mask, desc) in zero_checks.items():
    count = mask.sum()
    if count > 0:
        print(f"  ⚠️ {col}: {count} rows with {desc}")
    else:
        print(f"  ✅ {col}: No {desc} values")

print("\n🛏️ Unreasonable Bedroom Counts:")
bedroom_stats = df['bedrooms'].describe()
print(f"  Min: {bedroom_stats['min']}")
print(f"  Max: {bedroom_stats['max']}")
print(f"  Mean: {bedroom_stats['mean']:.2f}")
print(f"  Median: {bedroom_stats['50%']}")

unreasonable_bedrooms = df[(df['bedrooms'] < 0) | (df['bedrooms'] > 10)]
print(f"\n  Rows with <0 or >10 bedrooms: {len(unreasonable_bedrooms)}")
if len(unreasonable_bedrooms) > 0:
    print("\n  Unreasonable Bedroom Values:")
    print(unreasonable_bedrooms[['bedrooms']].value_counts().sort_index())

print("\n📐 Unreasonable Area Values:")
area_stats = df['area_sqft'].describe()
print(f"  Min: {area_stats['min']:.2f}")
print(f"  Max: {area_stats['max']:.2f}")
print(f"  Mean: {area_stats['mean']:.2f}")
print(f"  Median: {area_stats['50%']:.2f}")

unreasonable_area = df[(df['area_sqft'] < 0) | (df['area_sqft'] > 10000)]
print(f"\n  Rows with <0 or >10000 sqft: {len(unreasonable_area)}")
if len(unreasonable_area) > 0:
    print("\n  Unreasonable Area Values:")
    print(unreasonable_area[['area_sqft']].head())

print("\n🏚️ Unreasonable Age Values:")
age_stats = df['age_years'].describe()
print(f"  Min: {age_stats['min']:.2f}")
print(f"  Max: {age_stats['max']:.2f}")
print(f"  Mean: {age_stats['mean']:.2f}")
print(f"  Median: {age_stats['50%']:.2f}")

unreasonable_age = df[(df['age_years'] < 0) | (df['age_years'] > 100)]
print(f"\n  Rows with <0 or >100 years: {len(unreasonable_age)}")
if len(unreasonable_age) > 0:
    print("\n  Unreasonable Age Values:")
    print(unreasonable_age[['age_years']].head())

invalid_mask = (df['area_sqft'] < 0) | (df['bedrooms'] < 0) | (df['bedrooms'] > 10) | \
               (df['age_years'] < 0) | (df['distance_city_km'] < 0) | (df['price_lakh'] < 0)
total_invalid = invalid_mask.sum()

print(f"\n📊 Total rows with any invalid values: {total_invalid}")
print(f"   Percentage of dataset: {(total_invalid/len(df))*100:.2f}%")

print("\n✅ Deliverable 26 completed!")