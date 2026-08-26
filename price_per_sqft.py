"""
DELIVERABLE 32: Create price_per_sqft
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


try:
    df = pd.read_csv("House_Price_Cleaned.csv")
    print("✅ Loaded cleaned dataset")
except:
    df = pd.read_csv("House Price.csv")
    print("⚠️ Cleaned dataset not found, using original dataset")

print("="*60)
print("DELIVERABLE 32: Create price_per_sqft")
print("="*60)


df['price_per_sqft'] = df['price_lakh'] / df['area_sqft']

print("\n📊 price_per_sqft Statistics:")
print(df['price_per_sqft'].describe())

print("\n📊 Original vs New Feature:")
comparison = pd.DataFrame({
    'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
    'Price (Lakhs)': [df['price_lakh'].mean(), df['price_lakh'].median(), 
                       df['price_lakh'].std(), df['price_lakh'].min(), df['price_lakh'].max()],
    'Area (sqft)': [df['area_sqft'].mean(), df['area_sqft'].median(), 
                    df['area_sqft'].std(), df['area_sqft'].min(), df['area_sqft'].max()],
    'Price per sqft': [df['price_per_sqft'].mean(), df['price_per_sqft'].median(), 
                       df['price_per_sqft'].std(), df['price_per_sqft'].min(), df['price_per_sqft'].max()]
})
print(comparison.round(2))

fig, axes = plt.subplots(2, 3, figsize=(18, 10))


axes[0, 0].hist(df['price_per_sqft'], bins=30, edgecolor='black', alpha=0.7, color='purple')
axes[0, 0].axvline(df['price_per_sqft'].mean(), color='red', linestyle='--', 
                   label=f"Mean: {df['price_per_sqft'].mean():.2f}")
axes[0, 0].axvline(df['price_per_sqft'].median(), color='green', linestyle='--', 
                   label=f"Median: {df['price_per_sqft'].median():.2f}")
axes[0, 0].set_xlabel('Price per sqft (Lakhs)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Price per sqft Distribution')
axes[0, 0].legend()


axes[0, 1].boxplot(df['price_per_sqft'])
axes[0, 1].set_ylabel('Price per sqft (Lakhs)')
axes[0, 1].set_title('Price per sqft Boxplot')


sns.kdeplot(df['price_per_sqft'], fill=True, ax=axes[0, 2], color='purple')
axes[0, 2].set_xlabel('Price per sqft (Lakhs)')
axes[0, 2].set_ylabel('Density')
axes[0, 2].set_title('Price per sqft Density Plot')


axes[1, 0].scatter(df['area_sqft'], df['price_per_sqft'], alpha=0.5, color='purple')
axes[1, 0].set_xlabel('Area (sqft)')
axes[1, 0].set_ylabel('Price per sqft (Lakhs)')
axes[1, 0].set_title('Area vs Price per sqft')
axes[1, 0].axhline(y=df['price_per_sqft'].mean(), color='red', linestyle='--', label='Mean')
axes[1, 0].legend()


df.boxplot(column='price_per_sqft', by='bedrooms', ax=axes[1, 1])
axes[1, 1].set_title('Price per sqft by Bedrooms')
axes[1, 1].set_xlabel('Bedrooms')
axes[1, 1].set_ylabel('Price per sqft (Lakhs)')
plt.setp(axes[1, 1].get_xticklabels(), rotation=45)

axes[1, 2].scatter(df['price_lakh'], df['price_per_sqft'], alpha=0.5, color='purple')
axes[1, 2].set_xlabel('Price (Lakhs)')
axes[1, 2].set_ylabel('Price per sqft (Lakhs)')
axes[1, 2].set_title('Price vs Price per sqft')

plt.tight_layout()
plt.show()

print("\n🏠 Top 5 properties with highest price per sqft:")
highest_ppsqft = df.nlargest(5, 'price_per_sqft')[['area_sqft', 'bedrooms', 'age_years', 
                                                     'distance_city_km', 'price_lakh', 'price_per_sqft']]
print(highest_ppsqft.round(2))

print("\n🏠 Top 5 properties with lowest price per sqft:")
lowest_ppsqft = df.nsmallest(5, 'price_per_sqft')[['area_sqft', 'bedrooms', 'age_years', 
                                                    'distance_city_km', 'price_lakh', 'price_per_sqft']]
print(lowest_ppsqft.round(2))


df.to_csv('House_Price_With_PPSQFT.csv', index=False)
print("\n💾 Dataset with price_per_sqft saved as 'House_Price_With_PPSQFT.csv'")

print("\n✅ Deliverable 32 completed!")