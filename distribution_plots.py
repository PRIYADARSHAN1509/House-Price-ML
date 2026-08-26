"""
DELIVERABLE 28: Plot area and price distributions
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("House Price.csv")

print("="*60)
print("DELIVERABLE 28: Plot Area and Price Distributions")
print("="*60)

print("\n📊 Area Distribution Statistics:")
area_stats = df['area_sqft'].describe()
print(area_stats)

print("\n💰 Price Distribution Statistics:")
price_stats = df['price_lakh'].describe()
print(price_stats)


fig = plt.figure(figsize=(16, 12))

plt.subplot(3, 3, 1)
plt.hist(df['area_sqft'], bins=30, edgecolor='black', alpha=0.7, color='blue')
plt.xlabel('Area (sqft)')
plt.ylabel('Frequency')
plt.title('Area Distribution')
plt.axvline(df['area_sqft'].mean(), color='red', linestyle='--', label=f'Mean: {df["area_sqft"].mean():.1f}')
plt.axvline(df['area_sqft'].median(), color='green', linestyle='--', label=f'Median: {df["area_sqft"].median():.1f}')
plt.legend()


plt.subplot(3, 3, 2)
plt.boxplot(df['area_sqft'])
plt.ylabel('Area (sqft)')
plt.title('Area Boxplot')

plt.subplot(3, 3, 3)
sns.kdeplot(df['area_sqft'], fill=True)
plt.xlabel('Area (sqft)')
plt.ylabel('Density')
plt.title('Area Density Plot')

plt.subplot(3, 3, 4)
plt.hist(df['price_lakh'], bins=30, edgecolor='black', alpha=0.7, color='orange')
plt.xlabel('Price (Lakhs)')
plt.ylabel('Frequency')
plt.title('Price Distribution')
plt.axvline(df['price_lakh'].mean(), color='red', linestyle='--', label=f'Mean: {df["price_lakh"].mean():.1f}')
plt.axvline(df['price_lakh'].median(), color='green', linestyle='--', label=f'Median: {df["price_lakh"].median():.1f}')
plt.legend()


plt.subplot(3, 3, 5)
plt.boxplot(df['price_lakh'])
plt.ylabel('Price (Lakhs)')
plt.title('Price Boxplot')

plt.subplot(3, 3, 6)
sns.kdeplot(df['price_lakh'], fill=True, color='orange')
plt.xlabel('Price (Lakhs)')
plt.ylabel('Density')
plt.title('Price Density Plot')

plt.subplot(3, 3, 7)
from scipy import stats
stats.probplot(df['area_sqft'], dist="norm", plot=plt)
plt.title('QQ Plot - Area')


plt.subplot(3, 3, 8)
stats.probplot(df['price_lakh'], dist="norm", plot=plt)
plt.title('QQ Plot - Price')

plt.subplot(3, 3, 9)
plt.hist(df['area_sqft'], bins=30, alpha=0.5, label='Area', color='blue')
plt.hist(df['price_lakh'], bins=30, alpha=0.5, label='Price', color='orange')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Area and Price Combined')
plt.legend()

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist(np.log(df['area_sqft']), bins=30, edgecolor='black', alpha=0.7, color='blue')
axes[0, 0].set_title('Log-Transformed Area')
axes[0, 0].set_xlabel('log(Area)')
axes[0, 0].set_ylabel('Frequency')

axes[0, 1].hist(np.log(df['price_lakh']), bins=30, edgecolor='black', alpha=0.7, color='orange')
axes[0, 1].set_title('Log-Transformed Price')
axes[0, 1].set_xlabel('log(Price)')
axes[0, 1].set_ylabel('Frequency')

axes[1, 0].scatter(df['area_sqft'], df['price_lakh'], alpha=0.5)
axes[1, 0].set_xlabel('Area (sqft)')
axes[1, 0].set_ylabel('Price (Lakhs)')
axes[1, 0].set_title('Original Scale')

axes[1, 1].scatter(np.log(df['area_sqft']), np.log(df['price_lakh']), alpha=0.5)
axes[1, 1].set_xlabel('log(Area)')
axes[1, 1].set_ylabel('log(Price)')
axes[1, 1].set_title('Log-Log Scale')

plt.tight_layout()
plt.show()

print("\n✅ Deliverable 28 completed!")
