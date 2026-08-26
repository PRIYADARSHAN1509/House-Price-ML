"""
DELIVERABLE 29: Analyze area vs price, bedrooms vs price, and age vs price
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


df = pd.read_csv("House Price.csv")

print("="*60)
print("DELIVERABLE 29: Analyze Relationships with Price")
print("="*60)

area_corr = df['area_sqft'].corr(df['price_lakh'])
bedrooms_corr = df['bedrooms'].corr(df['price_lakh'])
age_corr = df['age_years'].corr(df['price_lakh'])
distance_corr = df['distance_city_km'].corr(df['price_lakh'])

print("\n📊 Correlations with Price:")
print(f"  Area vs Price: {area_corr:.4f}")
print(f"  Bedrooms vs Price: {bedrooms_corr:.4f}")
print(f"  Age vs Price: {age_corr:.4f}")
print(f"  Distance vs Price: {distance_corr:.4f}")


fig, axes = plt.subplots(2, 3, figsize=(18, 12))

axes[0, 0].scatter(df['area_sqft'], df['price_lakh'], alpha=0.6, s=30)
axes[0, 0].set_xlabel('Area (sqft)')
axes[0, 0].set_ylabel('Price (Lakhs)')
axes[0, 0].set_title(f'Area vs Price (Corr: {area_corr:.3f})')

slope, intercept = stats.linregress(df['area_sqft'], df['price_lakh'])[:2]
x_line = np.array([df['area_sqft'].min(), df['area_sqft'].max()])
axes[0, 0].plot(x_line, slope * x_line + intercept, 'r-', linewidth=2)


axes[0, 1].scatter(df['bedrooms'], df['price_lakh'], alpha=0.6, s=30)
axes[0, 1].set_xlabel('Bedrooms')
axes[0, 1].set_ylabel('Price (Lakhs)')
axes[0, 1].set_title(f'Bedrooms vs Price (Corr: {bedrooms_corr:.3f})')

jittered_bedrooms = df['bedrooms'] + np.random.normal(0, 0.05, len(df))
axes[0, 1].scatter(jittered_bedrooms, df['price_lakh'], alpha=0.3, s=20, color='green')

slope2, intercept2 = stats.linregress(df['bedrooms'], df['price_lakh'])[:2]
x_line2 = np.array([df['bedrooms'].min(), df['bedrooms'].max()])
axes[0, 1].plot(x_line2, slope2 * x_line2 + intercept2, 'r-', linewidth=2)


axes[0, 2].scatter(df['age_years'], df['price_lakh'], alpha=0.6, s=30)
axes[0, 2].set_xlabel('Age (Years)')
axes[0, 2].set_ylabel('Price (Lakhs)')
axes[0, 2].set_title(f'Age vs Price (Corr: {age_corr:.3f})')

slope3, intercept3 = stats.linregress(df['age_years'], df['price_lakh'])[:2]
x_line3 = np.array([df['age_years'].min(), df['age_years'].max()])
axes[0, 2].plot(x_line3, slope3 * x_line3 + intercept3, 'r-', linewidth=2)


df.boxplot(column='price_lakh', by='bedrooms', ax=axes[1, 0])
axes[1, 0].set_title('Price by Number of Bedrooms')
axes[1, 0].set_xlabel('Bedrooms')
axes[1, 0].set_ylabel('Price (Lakhs)')
plt.setp(axes[1, 0].get_xticklabels(), rotation=45)


df['age_category'] = pd.cut(df['age_years'], bins=[-1, 10, 30, 100], labels=['New (0-10)', 'Moderate (10-30)', 'Old (30+)'])
df.boxplot(column='price_lakh', by='age_category', ax=axes[1, 1])
axes[1, 1].set_title('Price by Age Category')
axes[1, 1].set_xlabel('Age Category')
axes[1, 1].set_ylabel('Price (Lakhs)')
plt.setp(axes[1, 1].get_xticklabels(), rotation=45)


hb = axes[1, 2].hexbin(df['area_sqft'], df['price_lakh'], gridsize=30, cmap='YlOrRd')
axes[1, 2].set_xlabel('Area (sqft)')
axes[1, 2].set_ylabel('Price (Lakhs)')
axes[1, 2].set_title('Area vs Price (Hexbin)')
plt.colorbar(hb, ax=axes[1, 2], label='Count')

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

sns.regplot(x='area_sqft', y='price_lakh', data=df, ax=axes[0], scatter_kws={'alpha':0.5}, 
            line_kws={'color': 'red', 'linewidth': 2})
axes[0].set_title('Area vs Price with Regression Line')

sns.regplot(x='bedrooms', y='price_lakh', data=df, ax=axes[1], scatter_kws={'alpha':0.5}, 
            x_jitter=0.1, line_kws={'color': 'red', 'linewidth': 2})
axes[1].set_title('Bedrooms vs Price with Regression Line')

sns.regplot(x='age_years', y='price_lakh', data=df, ax=axes[2], scatter_kws={'alpha':0.5},
            line_kws={'color': 'red', 'linewidth': 2})
axes[2].set_title('Age vs Price with Regression Line')

plt.tight_layout()
plt.show()


sns.jointplot(x='area_sqft', y='price_lakh', data=df, kind='reg', height=8)
plt.suptitle('Area vs Price - Joint Distribution', y=1.02)
plt.show()

print("\n📊 Price Statistics by Bedroom Count:")
bedroom_stats = df.groupby('bedrooms')['price_lakh'].agg(['count', 'mean', 'median', 'std']).round(2)
print(bedroom_stats)

print("\n✅ Deliverable 29 completed!")