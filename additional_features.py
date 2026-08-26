"""
DELIVERABLE 33: Create one additional meaningful feature
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

try:
    df = pd.read_csv("House_Price_With_PPSQFT.csv")
    print("✅ Loaded dataset with price_per_sqft")
except:
    df = pd.read_csv("House Price.csv")
    df['price_per_sqft'] = df['price_lakh'] / df['area_sqft']
    print("⚠️ Created price_per_sqft on the fly")

print("="*60)
print("DELIVERABLE 33: Create One Additional Meaningful Feature")
print("="*60)


print("\n🔧 Creating Multiple Meaningful Features...")

df['age_category'] = pd.cut(df['age_years'], 
                            bins=[-1, 5, 20, 50, 100], 
                            labels=['New (0-5 yrs)', 'Moderate (5-20 yrs)', 
                                    'Old (20-50 yrs)', 'Very Old (50+ yrs)'])


df['size_category'] = pd.cut(df['area_sqft'], 
                             bins=[-1, 1000, 2000, 3500, 10000], 
                             labels=['Small', 'Medium', 'Large', 'Mansion'])

df['distance_category'] = pd.cut(df['distance_city_km'], 
                                 bins=[-1, 5, 15, 30, 100], 
                                 labels=['City Center', 'Suburban', 'Outskirts', 'Rural'])


df['age_area_interaction'] = df['age_years'] * df['area_sqft'] / 1000


scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(df[['area_sqft', 'bedrooms', 'price_per_sqft']])
df['property_score'] = normalized_features.mean(axis=1)


df['area_per_bedroom'] = df['area_sqft'] / df['bedrooms']


df['price_category'] = pd.cut(df['price_lakh'], 
                              bins=[-1, 50, 100, 200, 1000], 
                              labels=['Affordable', 'Moderate', 'Expensive', 'Luxury'])

df['log_price'] = np.log(df['price_lakh'])

print("\n📋 New Features Created:")
print(f"  1. age_category: Categorical - Age groups")
print(f"  2. size_category: Categorical - Property size groups")
print(f"  3. distance_category: Categorical - Location groups")
print(f"  4. age_area_interaction: Numerical - Interaction between age and area")
print(f"  5. property_score: Numerical - Composite score of area, bedrooms, and price_per_sqft")
print(f"  6. area_per_bedroom: Numerical - Area per bedroom")
print(f"  7. price_category: Categorical - Price groups")
print(f"  8. log_price: Numerical - Log transformed price")

print("\n📊 Sample of new features:")
sample_cols = ['area_sqft', 'price_lakh', 'price_per_sqft', 'age_category', 'size_category', 
               'distance_category', 'age_area_interaction', 'property_score', 
               'area_per_bedroom', 'price_category', 'log_price']
print(df[sample_cols].head(10))


print("\n📊 Statistics of Numerical New Features:")
numerical_features = ['age_area_interaction', 'property_score', 'area_per_bedroom', 'log_price']
print(df[numerical_features].describe().round(3))


fig, axes = plt.subplots(2, 4, figsize=(20, 12))


df.boxplot(column='price_lakh', by='age_category', ax=axes[0, 0])
axes[0, 0].set_title('Price by Age Category')
axes[0, 0].set_xlabel('Age Category')
axes[0, 0].set_ylabel('Price (Lakhs)')
plt.setp(axes[0, 0].get_xticklabels(), rotation=45)


df.boxplot(column='price_lakh', by='size_category', ax=axes[0, 1])
axes[0, 1].set_title('Price by Size Category')
axes[0, 1].set_xlabel('Size Category')
axes[0, 1].set_ylabel('Price (Lakhs)')
plt.setp(axes[0, 1].get_xticklabels(), rotation=45)


df.boxplot(column='price_lakh', by='distance_category', ax=axes[0, 2])
axes[0, 2].set_title('Price by Distance Category')
axes[0, 2].set_xlabel('Distance Category')
axes[0, 2].set_ylabel('Price (Lakhs)')
plt.setp(axes[0, 2].get_xticklabels(), rotation=45)


axes[0, 3].hist(df['price_category'], edgecolor='black')
axes[0, 3].set_title('Price Category Distribution')
axes[0, 3].set_xlabel('Price Category')
axes[0, 3].set_ylabel('Count')
plt.setp(axes[0, 3].get_xticklabels(), rotation=45)


axes[1, 0].scatter(df['age_area_interaction'], df['price_lakh'], alpha=0.5)
axes[1, 0].set_xlabel('Age × Area Interaction')
axes[1, 0].set_ylabel('Price (Lakhs)')
axes[1, 0].set_title('Age-Area Interaction vs Price')


axes[1, 1].scatter(df['property_score'], df['price_lakh'], alpha=0.5, color='green')
axes[1, 1].set_xlabel('Property Score')
axes[1, 1].set_ylabel('Price (Lakhs)')
axes[1, 1].set_title('Property Score vs Price')


axes[1, 2].scatter(df['area_per_bedroom'], df['price_lakh'], alpha=0.5, color='orange')
axes[1, 2].set_xlabel('Area per Bedroom')
axes[1, 2].set_ylabel('Price (Lakhs)')
axes[1, 2].set_title('Area per Bedroom vs Price')


axes[1, 3].hist(df['log_price'], bins=30, edgecolor='black', alpha=0.7, color='red')
axes[1, 3].axvline(df['log_price'].mean(), color='blue', linestyle='--', 
                   label=f"Mean: {df['log_price'].mean():.2f}")
axes[1, 3].set_xlabel('Log(Price)')
axes[1, 3].set_ylabel('Frequency')
axes[1, 3].set_title('Log Price Distribution')
axes[1, 3].legend()

plt.tight_layout()
plt.show()


print("\n📈 Correlation of New Features with Price:")
new_features_corr = df[['age_area_interaction', 'property_score', 'area_per_bedroom', 
                        'log_price', 'price_lakh']].corr()['price_lakh'].sort_values(ascending=False)
print(new_features_corr.round(4))

df.to_csv('House_Price_With_Features.csv', index=False)
print("\n💾 Dataset with all features saved as 'House_Price_With_Features.csv'")

print("\n✅ Deliverable 33 completed!")
