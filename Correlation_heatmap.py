"""
DELIVERABLE 30: Create a correlation heatmap
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("House Price.csv")

print("="*60)
print("DELIVERABLE 30: Create a Correlation Heatmap")
print("="*60)

correlation_matrix = df.corr()

print("\n📊 Correlation Matrix:")
print(correlation_matrix.round(4))

print("\n📈 Correlations with Target (price_lakh):")
correlations_with_target = correlation_matrix['price_lakh'].sort_values(ascending=False)
print(correlations_with_target.round(4))


fig, axes = plt.subplots(2, 2, figsize=(16, 14))


sns.heatmap(correlation_matrix, 
            annot=True, 
            cmap='coolwarm', 
            fmt='.3f',
            square=True, 
            linewidths=0.5,
            cbar_kws={'label': 'Correlation Coefficient'},
            ax=axes[0, 0])
axes[0, 0].set_title('Correlation Heatmap - All Features')


mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, 
            mask=mask,
            annot=True, 
            cmap='coolwarm', 
            fmt='.3f',
            square=True, 
            linewidths=0.5,
            cbar_kws={'label': 'Correlation Coefficient'},
            ax=axes[0, 1])
axes[0, 1].set_title('Correlation Heatmap (Lower Triangle)')


corr_with_target = correlation_matrix[['price_lakh']].drop('price_lakh')
sns.heatmap(corr_with_target, 
            annot=True, 
            cmap='RdYlGn', 
            fmt='.3f',
            cbar_kws={'label': 'Correlation Coefficient'},
            ax=axes[1, 0])
axes[1, 0].set_title('Correlation with Price')

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist


linkage_matrix = linkage(pdist(correlation_matrix), method='average')
sns.clustermap(correlation_matrix, 
               annot=True, 
               fmt='.3f',
               cmap='coolwarm',
               figsize=(10, 10))
plt.suptitle('Clustered Correlation Heatmap', y=1.02)

plt.tight_layout()
plt.show()


print("\n🔗 Strong Positive Correlations (>0.5):")
strong_pos = correlation_matrix[(correlation_matrix > 0.5) & (correlation_matrix < 1.0)]
print(strong_pos.stack().sort_values(ascending=False))

print("\n🔗 Strong Negative Correlations (<-0.5):")
strong_neg = correlation_matrix[(correlation_matrix < -0.5) & (correlation_matrix > -1.0)]
print(strong_neg.stack().sort_values(ascending=True))


top_features = correlation_matrix['price_lakh'].sort_values(ascending=False).head(6).index.tolist()
print(f"\n📊 Pairplot for top features: {top_features}")
sns.pairplot(df[top_features], height=2.5)
plt.suptitle('Pairplot of Top Features with Price', y=1.02)
plt.show()

print("\n✅ Deliverable 30 completed!")
