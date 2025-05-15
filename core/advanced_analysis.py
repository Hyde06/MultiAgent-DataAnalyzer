import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import logging
from core.column_analyzer import ColumnAnalyzer

logger = logging.getLogger(__name__)

class AdvancedAnalyzer:
    def __init__(self, df: pd.DataFrame):
        """Initialize the advanced analyzer with a DataFrame"""
        self.df = df
        self.column_analyzer = ColumnAnalyzer(df)
        self.analysis_columns = self.column_analyzer.get_analysis_columns()
        self.numeric_cols = self.analysis_columns['correlation']
        self.categorical_cols = self.analysis_columns['categorical_analysis']
        
    def detect_outliers(self, column: str, method: str = 'zscore') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Detect outliers in a numeric column using different methods"""
        if column not in self.numeric_cols:
            raise ValueError(f"Column {column} is not a numeric analysis column")
            
        data = self.df[column].dropna()
        outliers = pd.DataFrame()
        stats_dict = {}
        
        if method == 'zscore':
            z_scores = np.abs(stats.zscore(data))
            outliers = self.df[z_scores > 3]
            stats_dict = {
                'method': 'Z-Score',
                'threshold': 3,
                'outlier_count': len(outliers),
                'percentage': (len(outliers) / len(self.df)) * 100
            }
            
        elif method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = self.df[(self.df[column] < (Q1 - 1.5 * IQR)) | 
                             (self.df[column] > (Q3 + 1.5 * IQR))]
            stats_dict = {
                'method': 'IQR',
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR,
                'outlier_count': len(outliers),
                'percentage': (len(outliers) / len(self.df)) * 100
            }
            
        return outliers, stats_dict
        
    def perform_pca(self, n_components: int = 2) -> Dict[str, Any]:
        """Perform Principal Component Analysis on numeric columns"""
        if len(self.numeric_cols) < 2:
            raise ValueError("Need at least 2 numeric columns for PCA")
            
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.df[self.numeric_cols])
        
        # Perform PCA
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(scaled_data)
        
        # Create results dictionary
        results = {
            'components': pca_result,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'feature_importance': dict(zip(self.numeric_cols, 
                                        np.abs(pca.components_[0]))),
            'total_variance_explained': sum(pca.explained_variance_ratio_)
        }
        
        return results
        
    def perform_clustering(self, n_clusters: int = 3) -> Dict[str, Any]:
        """Perform K-means clustering on numeric columns"""
        if len(self.numeric_cols) < 2:
            raise ValueError("Need at least 2 numeric columns for clustering")
            
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.df[self.numeric_cols])
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        
        # Calculate cluster statistics
        cluster_stats = {}
        for i in range(n_clusters):
            cluster_data = self.df[clusters == i]
            cluster_stats[f'cluster_{i}'] = {
                'size': len(cluster_data),
                'percentage': (len(cluster_data) / len(self.df)) * 100,
                'centroid': kmeans.cluster_centers_[i].tolist()
            }
            
        results = {
            'clusters': clusters,
            'cluster_stats': cluster_stats,
            'inertia': kmeans.inertia_,
            'centroids': kmeans.cluster_centers_.tolist()
        }
        
        return results
        
    def analyze_correlations(self) -> Dict[str, Any]:
        """Analyze correlations between numeric columns"""
        if len(self.numeric_cols) < 2:
            raise ValueError("Need at least 2 numeric columns for correlation analysis")
            
        # Calculate correlation matrix
        corr_matrix = self.df[self.numeric_cols].corr()
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > 0.7:  # Strong correlation threshold
                    strong_correlations.append({
                        'pair': (corr_matrix.columns[i], corr_matrix.columns[j]),
                        'correlation': corr
                    })
                    
        results = {
            'correlation_matrix': corr_matrix,
            'strong_correlations': strong_correlations,
            'average_correlation': corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)].mean()
        }
        
        return results
        
    def generate_advanced_visualizations(self) -> List[Tuple[str, Any]]:
        """Generate advanced visualizations for the dataset"""
        visualizations = []
        
        # PCA visualization if applicable
        if len(self.numeric_cols) >= 2:
            try:
                pca_results = self.perform_pca()
                fig = px.scatter(
                    x=pca_results['components'][:, 0],
                    y=pca_results['components'][:, 1],
                    title="PCA Visualization",
                    labels={'x': 'PC1', 'y': 'PC2'}
                )
                visualizations.append(("PCA", fig))
            except Exception as e:
                logger.warning(f"Could not create PCA visualization: {str(e)}")
        
        # Clustering visualization if applicable
        if len(self.numeric_cols) >= 2:
            try:
                cluster_results = self.perform_clustering()
                fig = px.scatter(
                    x=self.df[self.numeric_cols[0]],
                    y=self.df[self.numeric_cols[1]],
                    color=cluster_results['clusters'],
                    title="Cluster Analysis",
                    labels={'x': self.numeric_cols[0], 'y': self.numeric_cols[1]}
                )
                visualizations.append(("Clustering", fig))
            except Exception as e:
                logger.warning(f"Could not create clustering visualization: {str(e)}")
        
        # Correlation heatmap
        if len(self.numeric_cols) >= 2:
            try:
                corr_results = self.analyze_correlations()
                fig = px.imshow(
                    corr_results['correlation_matrix'],
                    labels=dict(color="Correlation"),
                    title="Advanced Correlation Heatmap"
                )
                visualizations.append(("Correlation Heatmap", fig))
            except Exception as e:
                logger.warning(f"Could not create correlation heatmap: {str(e)}")
        
        return visualizations 