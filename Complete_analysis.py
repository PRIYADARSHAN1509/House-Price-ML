"""
Complete House Price Analysis - Using Original CSV File
All Deliverables 24-34 in One File
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Load the original dataset
df = pd.read_csv("House Price.csv")
print(f"✅ Loaded House Price.csv")
print(f"📊 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# ============================================================================
# DATA CLEANING (DELIVERABLE 31)
# ============================================================================
print("\n" + "="*60)
print("DELIVERABLE 31: Data Cleaning")
print("="*60)

# Create a copy for cleaning
df_original = df.copy()

# Remove duplicates
df_cleaned = df_original.drop_duplicates()
print(f"✅ Duplicates removed: {len(df_original) - len(df_cleaned)}")

# Remove invalid values
invalid_mask = (df_cleaned['area_sqft'] < 0) | \
               (df_cleaned['bedrooms'] < 0) | \
               (df_cleaned['bedrooms'] > 10) | \
               (df_cleaned['age_years'] < 0) | \
               (df_cleaned['distance_city_km'] < 0) | \
               (df_cleaned['price_lakh'] < 0)

df_cleaned = df_cleaned[~invalid_mask]
print(f"✅ Invalid values removed")

# Handle missing values
if df_cleaned.isnull().sum().sum() > 0:
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype in ['int64', 'float64']:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
    print(f"✅ Missing values filled")

# Convert data types
if df_cleaned['bedrooms'].dtype == 'float64':
    df_cleaned['bedrooms'] = df_cleaned['bedrooms'].astype(int)

# Use cleaned data for all analysis
df = df_cleaned
print(f"✅ Final cleaned shape: {df.shape[0]} rows × {df.shape[1]} columns")

# Save cleaned dataset
df.to_csv('House_Price_Cleaned.csv', index=False)
print(f"💾 Cleaned dataset saved as 'House_Price_Cleaned.csv'")

# ============================================================================
# DELIVERABLE 24: Basic EDA and Data Types
# ============================================================================
print("\n" + "="*60)
print("DELIVERABLE 24: Basic EDA and Data Types")
print("="*60)

print("\n📋 Column Names and Data Types:")
print(df.dtypes)

print("\n📊 First 5 rows:")
print(df.head())

print("\n📈 Summary Statistics:")
print(df.describe())

# ============================================================================
# DELIVERABLE 25: Missing Values and Duplicate Rows
# ============================================================================
print("\n" + "="*60)
print("DELIVERABLE 25: Missing Values and Duplicate Rows")
print("="*60)

print(f"\n🔍 Missing Values per Column:")
print(df.isnull().sum())
print(f"\nTotal Missing Values: {df.isnull().sum().sum()}")
print(f"\nDuplicate Rows: {df.duplicated().sum()}")

# ============================================================================
# DELIVERABLE 26: Identify Impossible Values
# ============================================================================
print("\n" + "="*60)
print("DELIVERABLE 26: Identify Impossible Values")
print("="*60)

print("\n🔴 Checking for invalid values:")
invalid_checks = {
    'Negative Area': (df['area_sqft'] < 0).sum(),
    'Negative Bedrooms': (df['bedrooms'] < 0).sum(),
    'Negative Age': (df['age_years'] < 0).sum(),
    'Negative Distance': (df['distance_city_km'] < 0).sum(),
    'Negative Price': (df['price_lakh'] < 0).sum(),
    'Unreasonable Bedrooms (>10)': (df['bedrooms'] > 10).sum(),
    'Zero Area': (df['area_sqft'] == 0).sum(),
    'Zero Price': (df['price_lakh'] == 0).sum()
}

for check, count in invalid_checks.items():
    status = "❌" if count > 0 else "✅"
    print(f"  {status} {check}: {count}")

# ============================================================================
# DELIVERABLE 27: Find Price Outliers
# ============================================================================
print("\n" + "="*60)
print("DELIVERABLE 27: Find Price Outliers")
print("="*60)

Q1 = df['price_lakh'].quantile(0.25)
Q3 = df['price_lakh'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

price_outliers = df[(df['price_lakh'] < lower_bound) | (df['price_lakh'] > upper_bound)]

print(f"\n📊 Price Statistics:")
print(f"  Q1: {Q1:.2f} lakhs")
print(f"  Q3: {Q3:.2f} lakhs")
print(f"  IQR: {IQR:.2f} lakhs")
print(f"  Lower Bound: {lower_bound:.2f} lakhs")
print(f"  Upper Bound: {upper_bound:.2f} lakhs")
print(f"  Price Outliers: {len(price_outliers)} ({len(price_outliers)/len(df)*100:.2f}%)")

# ============================================================================
# DELIVERABLE 28: Plot Area and Price Distributions
# ============================================================================
print("\n" + "="*60)
print("DELIVERABLE 28: Plot Area and Price Distributions")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Area distribution
axes[0, 0].hist(df['area_sqft'], bins=30, edgecolor='black', alpha=0.7, color='blue')
axes[0, 0].axvline(df['area_sqft'].mean(), color='red', linestyle='--', 
                   label=f"Mean: {df['area_sqft'].mean():.1f}")
axes[0, 0].axvline(df['area_sqft'].median(), color='green', linestyle='--', 
                   label=f"Median: {df['area_sqft'].median():.1f}")
axes[0, 0].set_xlabel('Area (sqft)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Area Distribution')
axes[0, 0].legend()

# Area boxplot
axes[0, 1].boxplot(df['area_sqft'])
axes[0, 1].set_ylabel('Area (sqft)')
axes[0, 1].set_title('Area Boxplot')

# Price distribution
axes[0, 2].hist(df['price_lakh'], bins=30, edgecolor='black', alpha=0.7, color='orange')
axes[0, 2].axvline(df['price_lakh'].mean(), color='red', linestyle='--', 
                   label=f"Mean: {df['price_lakh'].mean():.1f}")
axes[0, 2].axvline(df['price_lakh'].median(), color='green', linestyle='--', 
                   label=f"Median: {df['price_lakh'].median():.1f}")
axes[0, 2].set_xlabel('Price (Lakhs)')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].set_title('Price Distribution')
axes[0, 2].legend()

# Price boxplot
axes[1, 0].boxplot(df['price_lakh'])
axes[1, 0].set_ylabel('Price (Lakhs)')
axes[1, 0].set_title('Price Boxplot')

# Area KDE
sns.kdeplot(df['area_sqft'], fill=True, ax=axes[1, 1], color='blue')
axes[1, 1].set_xlabel('Area (sqft)')
axes[1, 1].set_ylabel('Density')
axes[1, 1].set_title('Area Density Plot')

# Price KDE
sns.kdeplot(df['price_lakh'], fill=True, ax=axes[1, 2], color='orange')
axes[1, 2].set_xlabel('Price (Lakhs)')
axes[1, 2].set_ylabel('Density')
axes[1, 2].set_title('Price Density Plot')

plt.tight_layout()
plt.show()

# ============================================================================
# DELIVERABLE 29: Analyze Area vs Price, Bedrooms vs Price, Age vs Price
# ============================================================================
print("\n" + "="*60)
print("DELIVERABLE 29: Analyze Relationships with Price")
print("="*60)

# Calculate correlations
area_corr = df['area_sqft'].corr(df['price_lakh'])
bedrooms_corr = df['bedrooms'].corr(df['price_lakh'])
age_corr = df['age_years'].corr(df['price_lakh'])
distance_corr = df['distance_city_km'].corr(df['price_lakh'])

print(f"\n📈 Correlations with Price:")
print(f"  Area vs Price: {area_corr:.4f}")
print(f"  Bedrooms vs Price: {bedrooms_corr:.4f}")
print(f"  Age vs Price: {age_corr:.4f}")
print(f"  Distance vs Price: {distance_corr:.4f}")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Area vs Price
axes[0, 0].scatter(df['area_sqft'], df['price_lakh'], alpha=0.6, s=30)
slope, intercept = stats.linregress(df['area_sqft'], df['price_lakh'])[:2]
x_line = np.array([df['area_sqft'].min(), df['area_sqft'].max()])
axes[0, 0].plot(x_line, slope * x_line + intercept, 'r-', linewidth=2)
axes[0, 0].set_xlabel('Area (sqft)')
axes[0, 0].set_ylabel('Price (Lakhs)')
axes[0, 0].set_title(f'Area vs Price (Corr: {area_corr:.3f})')

# Bedrooms vs Price
axes[0, 1].scatter(df['bedrooms'], df['price_lakh'], alpha=0.6, s=30)
slope2, intercept2 = stats.linregress(df['bedrooms'], df['price_lakh'])[:2]
x_line2 = np.array([df['bedrooms'].min(), df['bedrooms'].max()])
axes[0, 1].plot(x_line2, slope2 * x_line2 + intercept2, 'r-', linewidth=2)
axes[0, 1].set_xlabel('Bedrooms')
axes[0, 1].set_ylabel('Price (Lakhs)')
axes[0, 1].set_title(f'Bedrooms vs Price (Corr: {bedrooms_corr:.3f})')

# Age vs Price
axes[0, 2].scatter(df['age_years'], df['price_lakh'], alpha=0.6, s=30)
slope3, intercept3 = stats.linregress(df['age_years'], df['price_lakh'])[:2]
x_line3 = np.array([df['age_years'].min(), df['age_years'].max()])
axes[0, 2].plot(x_line3, slope3 * x_line3 + intercept3, 'r-', linewidth=2)
axes[0, 2].set_xlabel('Age (Years)')
axes[0, 2].set_ylabel('Price (Lakhs)')
axes[0, 2].set_title(f'Age vs Price (Corr: {age_corr:.3f})')

# Bedrooms boxplot
df.boxplot(column='price_lakh', by='bedrooms', ax=axes[1, 0])
axes[1, 0].set_title('Price by Number of Bedrooms')
axes[1, 0].set_xlabel('Bedrooms')
axes[1, 0].set_ylabel('Price (Lakhs)')
plt.setp(axes[1, 0].get_xticklabels(), rotation=45)

# Age Category boxplot
df['age_category'] = pd.cut(df['age_years'], bins=[-1, 10, 30, 100], 
                            labels=['New (0-10)', 'Moderate (10-30)', 'Old (30+)'])
df.boxplot(column='price_lakh', by='age_category', ax=axes[1, 1])
axes[1, 1].set_title('Price by Age Category')
axes[1, 1].set_xlabel('Age Category')
axes[1, 1].set_ylabel('Price (Lakhs)')
plt.setp(axes[1, 1].get_xticklabels(), rotation=45)

# Hexbin
hb = axes[1, 2].hexbin(df['area_sqft'], df['price_lakh'], gridsize=30, cmap='YlOrRd')
axes[1, 2].set_xlabel('Area (sqft)')
axes[1, 2].set_ylabel('Price (Lakhs)')
axes[1, 2].set_title('Area vs Price (Hexbin)')
plt.colorbar(hb, ax=axes[1, 2], label='Count')

plt.tight_layout()
plt.show()

# ============================================================================
# DELIVERABLE 30: Create a Correlation Heatmap
# ============================================================================
print("\n" + "="*60)
print("DELIVERABLE 30: Create a Correlation Heatmap")
print("="*60)

correlation_matrix = df[['area_sqft', 'bedrooms', 'age_years', 'distance_city_km', 'price_lakh']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.3f', 
            square=True, linewidths=0.5, cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Correlation Heatmap of House Price Features')
plt.show()

print("\n📈 Correlations with Price:")
print(correlation_matrix['price_lakh'].sort_values(ascending=False).round(4))

# ============================================================================
# DELIVERABLE 32: Create price_per_sqft
# ============================================================================
print("\n" + "="*60)
print("DELIVERABLE 32: Create price_per_sqft")
print("="*60)

df['price_per_sqft'] = df['price_lakh'] / df['area_sqft']
print(f"\n📊 price_per_sqft Statistics:")
print(df['price_per_sqft'].describe().round(2))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(df['price_per_sqft'], bins=30, edgecolor='black', alpha=0.7, color='purple')
axes[0].set_xlabel('Price per sqft (Lakhs)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Price per sqft Distribution')

axes[1].boxplot(df['price_per_sqft'])
axes[1].set_ylabel('Price per sqft (Lakhs)')
axes[1].set_title('Price per sqft Boxplot')

sns.kdeplot(df['price_per_sqft'], fill=True, ax=axes[2], color='purple')
axes[2].set_xlabel('Price per sqft (Lakhs)')
axes[2].set_ylabel('Density')
axes[2].set_title('Price per sqft Density')

plt.tight_layout()
plt.show()

# ============================================================================
# DELIVERABLE 33: Create One Additional Meaningful Feature
# ============================================================================
print("\n" + "="*60)
print("DELIVERABLE 33: Create Additional Meaningful Features")
print("="*60)

# Create multiple features
df['age_area_interaction'] = df['age_years'] * df['area_sqft'] / 1000
df['area_per_bedroom'] = df['area_sqft'] / df['bedrooms']

# Normalize and create property score
scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(df[['area_sqft', 'bedrooms', 'price_per_sqft']])
df['property_score'] = normalized_features.mean(axis=1)

print("\n📊 New Features Created:")
print(f"  1. age_area_interaction: Age × Area / 1000")
print(f"  2. area_per_bedroom: Area per bedroom")
print(f"  3. property_score: Composite score (Area, Bedrooms, Price per sqft)")

print("\n📊 New Feature Statistics:")
print(df[['age_area_interaction', 'area_per_bedroom', 'property_score']].describe().round(2))

# ============================================================================
# DELIVERABLE 34: Use price_lakh as Target, Separate X and y
# ============================================================================
print("\n" + "="*60)
print("DELIVERABLE 34: Use price_lakh as Target, Separate X and y")
print("="*60)

# Select features
features = ['area_sqft', 'bedrooms', 'age_years', 'distance_city_km', 
            'price_per_sqft', 'age_area_interaction', 'area_per_bedroom', 'property_score']

X = df[features]
y = df['price_lakh']

print(f"\n📊 Features (X): {X.shape}")
print(f"   Features: {X.columns.tolist()}")
print(f"\n📊 Target (y): {y.shape}")
print(f"   Target: price_lakh")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n📊 Train-Test Split:")
print(f"  X_train: {X_train.shape}")
print(f"  X_test: {X_test.shape}")
print(f"  y_train: {y_train.shape}")
print(f"  y_test: {y_test.shape}")

# Save final dataset
df.to_csv('House_Price_Final.csv', index=False)
print("\n💾 Final dataset saved as 'House_Price_Final.csv'")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*60)
print("🎉 ALL DELIVERABLES COMPLETED SUCCESSFULLY!")
print("="*60)

print("\n📋 Deliverables Summary:")
deliverables = [
    "24. Basic EDA and data types",
    "25. Missing values and duplicate rows",
    "26. Impossible values identification",
    "27. Price outliers detection",
    "28. Area and price distributions",
    "29. Area vs Price, Bedrooms vs Price, Age vs Price analysis",
    "30. Correlation heatmap",
    "31. Data cleaning (included)",
    "32. Price per sqft creation",
    "33. Additional meaningful features",
    "34. Separate X and y"
]

for i, d in enumerate(deliverables, 1):
    print(f"  ✅ {d}")

print(f"\n📊 Final Dataset:")
print(f"  Rows: {len(df)}")
print(f"  Columns: {len(df.columns)}")
print(f"  Features: {len(features)}")
print(f"  Target: price_lakh")

print("\n📂 Files Generated:")
print("  - House_Price_Cleaned.csv (cleaned dataset)")
print("  - House_Price_Final.csv (final dataset with all features)")
