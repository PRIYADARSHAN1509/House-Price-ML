"""
DELIVERABLE 27: Find price outliers
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("House Price.csv")

print("="*60)
print("DELIVERABLE 27: Find Price Outliers")
print("="*60)

Q1 = df['price_lakh'].quantile(0.25)
Q3 = df['price_lakh'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("\n📊 Price Statistics:")
print(f"  Q1 (25th percentile): {Q1:.2f} lakhs")
print(f"  Q3 (75th percentile): {Q3:.2f} lakhs")
print(f"  IQR: {IQR:.2f} lakhs")
print(f"  Lower Bound: {lower_bound:.2f} lakhs")
print(f"  Upper Bound: {upper_bound:.2f} lakhs")
print(f"  Min: {df['price_lakh'].min():.2f} lakhs")
print(f"  Max: {df['price_lakh'].max():.2f} lakhs")
print(f"  Mean: {df['price_lakh'].mean():.2f} lakhs")
print(f"  Median: {df['price_lakh'].median():.2f} lakhs")
print(f"  Std Dev: {df['price_lakh'].std():.2f} lakhs")


price_outliers = df[(df['price_lakh'] < lower_bound) | (df['price_lakh'] > upper_bound)]

print(f"\n🔴 Price Outliers Found: {len(price_outliers)}")
print(f"   Percentage of dataset: {(len(price_outliers)/len(df))*100:.2f}%")

if len(price_outliers) > 0:
    print("\n📋 Price Outlier Values (sorted):")
    outlier_values = price_outliers['price_lakh'].sort_values()
    print(outlier_values)
    
    print("\n📋 Outlier Rows:")
    print(price_outliers)
 
    print("\n📊 Outlier Statistics:")
    print(f"  Min Outlier: {price_outliers['price_lakh'].min():.2f}")
    print(f"  Max Outlier: {price_outliers['price_lakh'].max():.2f}")
    print(f"  Mean Outlier: {price_outliers['price_lakh'].mean():.2f}")


from scipy import stats
z_scores = np.abs(stats.zscore(df['price_lakh']))
z_outliers = df[z_scores > 3]

print(f"\n📊 Using Z-score method (threshold=3):")
print(f"  Z-score Outliers Found: {len(z_outliers)}")
if len(z_outliers) > 0:
    print(f"  Outlier prices: {z_outliers['price_lakh'].tolist()}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))


axes[0].boxplot(df['price_lakh'])
axes[0].set_title('Boxplot of House Prices')
axes[0].set_ylabel('Price (Lakhs)')
axes[0].axhline(y=lower_bound, color='g', linestyle='--', label=f'Lower Bound: {lower_bound:.1f}')
axes[0].axhline(y=upper_bound, color='r', linestyle='--', label=f'Upper Bound: {upper_bound:.1f}')
axes[0].legend()


axes[1].hist(df['price_lakh'], bins=30, edgecolor='black', alpha=0.7)
axes[1].axvline(x=lower_bound, color='g', linestyle='--', label='Lower Bound')
axes[1].axvline(x=upper_bound, color='r', linestyle='--', label='Upper Bound')
axes[1].set_title('Price Distribution with Outlier Bounds')
axes[1].set_xlabel('Price (Lakhs)')
axes[1].set_ylabel('Frequency')
axes[1].legend()


colors = ['red' if (x < lower_bound or x > upper_bound) else 'blue' for x in df['price_lakh']]
axes[2].scatter(range(len(df)), df['price_lakh'], c=colors, alpha=0.6)
axes[2].axhline(y=lower_bound, color='g', linestyle='--', label='Lower Bound')
axes[2].axhline(y=upper_bound, color='r', linestyle='--', label='Upper Bound')
axes[2].set_title('Price Values with Outliers Highlighted')
axes[2].set_xlabel('Data Point Index')
axes[2].set_ylabel('Price (Lakhs)')
axes[2].legend()

plt.tight_layout()
plt.show()

print("\n✅ Deliverable 27 completed!")