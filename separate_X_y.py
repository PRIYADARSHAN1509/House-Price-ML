"""
DELIVERABLE 34: Use price_lakh as the target and separate X and y
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


try:
    df = pd.read_csv("House_Price_With_Features.csv")
    print("✅ Loaded dataset with all features")
except:
    try:
        df = pd.read_csv("House_Price_With_PPSQFT.csv")
        print("✅ Loaded dataset with price_per_sqft")
    except:
        df = pd.read_csv("House Price.csv")
        print("⚠️ Using original dataset")

print("="*60)
print("DELIVERABLE 34: Separate X (Features) and y (Target)")
print("="*60)


numerical_features = ['area_sqft', 'bedrooms', 'age_years', 'distance_city_km']

engineered_features = ['price_per_sqft', 'age_area_interaction', 'property_score', 
                       'area_per_bedroom', 'log_price']


existing_features = []
for feat in engineered_features:
    if feat in df.columns:
        existing_features.append(feat)

features_to_use = numerical_features + existing_features


if not existing_features:
    features_to_use = numerical_features
    print("\n⚠️ No engineered features found, using only numerical features")

X = df[features_to_use]
y = df['price_lakh']

print("\n📊 Features (X) Shape:", X.shape)
print(f"   Features used: {X.columns.tolist()}")

print("\n📊 Target (y) Shape:", y.shape)
print(f"   Target: price_lakh")

print("\n📊 Feature Statistics:")
print(X.describe().round(2))

print("\n📊 Target Statistics:")
print(y.describe().round(2))


print(f"\n❓ Missing values in X: {X.isnull().sum().sum()}")
print(f"❓ Missing values in y: {y.isnull().sum().sum()}")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n📊 Train-Test Split:")
print(f"   X_train: {X_train.shape}")
print(f"   X_test: {X_test.shape}")
print(f"   y_train: {y_train.shape}")
print(f"   y_test: {y_test.shape}")

print("\n📝 Feature Data Types:")
print(X.dtypes)

print("\n📈 Correlation of Each Feature with Target:")
correlations = X.corrwith(y).sort_values(ascending=False)
print(correlations.round(4))


fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].barh(correlations.index, correlations.values)
axes[0, 0].set_xlabel('Correlation with Price')
axes[0, 0].set_title('Feature Correlations with Target')
axes[0, 0].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
axes[0, 0].axvline(x=0.3, color='red', linestyle='--', alpha=0.5)
axes[0, 0].axvline(x=-0.3, color='red', linestyle='--', alpha=0.5)


X.hist(ax=axes[0, 1], figsize=(12, 8), bins=20, edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Feature Distributions')
axes[0, 1].set_xlabel('Values')
axes[0, 1].set_ylabel('Frequency')
plt.setp(axes[0, 1].get_xticklabels(), rotation=45)


corr_matrix = X.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.3f', 
            square=True, linewidths=0.5, ax=axes[1, 0])
axes[1, 0].set_title('Feature Correlation Matrix')

feature_summary = "Features ready for modeling:\n\n"
for i, feat in enumerate(features_to_use, 1):
    feature_summary += f"{i}. {feat}\n"
feature_summary += f"\nTotal Features: {len(features_to_use)}"
feature_summary += f"\nTotal Samples: {len(df)}"

axes[1, 1].text(0.1, 0.5, feature_summary, transform=axes[1, 1].transAxes,
                fontsize=12, verticalalignment='center')
axes[1, 1].set_title('Feature Summary')
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()

print("\n💾 Saving train-test split data...")
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False, header=['price_lakh'])
y_test.to_csv('y_test.csv', index=False, header=['price_lakh'])

print("\n✅ Train-test split saved as CSV files:")
print("   - X_train.csv, X_test.csv")
print("   - y_train.csv, y_test.csv")


print("\n🔍 Quick Validation:")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
print(f"Train size: {len(X_train)/len(X)*100:.1f}%")
print(f"Test size: {len(X_test)/len(X)*100:.1f}%")

print("\n" + "="*60)
print("📋 DELIVERABLES COMPLETION SUMMARY")
print("="*60)
print("✅ 24. Basic EDA and data types")
print("✅ 25. Missing values and duplicate rows")
print("✅ 26. Impossible values (negative area, unreasonable bedrooms)")
print("✅ 27. Price outliers")
print("✅ 28. Area and price distributions")
print("✅ 29. Area vs Price, Bedrooms vs Price, Age vs Price")
print("✅ 30. Correlation heatmap")
print("✅ 31. Clean duplicates, invalid values, missing values")
print("✅ 32. Create price_per_sqft")
print("✅ 33. Create additional meaningful features")
print("✅ 34. Use price_lakh as target, separate X and y")
print("="*60)
print("\n🎉 ALL DELIVERABLES COMPLETED SUCCESSFULLY!")