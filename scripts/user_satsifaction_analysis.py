import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean

class EngagementScoreCalculator:
    def __init__(self, df):
        self.df = df
        self.scaler = StandardScaler()

    def preprocess_data(self):
        # Handle missing values by filling them with the mean, only for numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
    
    def compute_experience_metrics(self):
        self.df['Average Throughput'] = (self.df['Avg Bearer TP DL (kbps)'] + 
                                         self.df['Avg Bearer TP UL (kbps)']) / 2
        self.df['TCP Retransmission'] = (self.df['TCP DL Retrans. Vol (Bytes)'] + 
                                         self.df['TCP UL Retrans. Vol (Bytes)']) / 2
    
    def perform_clustering(self):
        metrics = self.df[['Average Throughput', 'TCP Retransmission']]
        scaled_metrics = self.scaler.fit_transform(metrics)
        
        kmeans = KMeans(n_clusters=3, random_state=42)
        self.df['Cluster'] = kmeans.fit_predict(scaled_metrics)
        
        # Identify the less engaged cluster (lowest average throughput)
        cluster_centers = pd.DataFrame(self.scaler.inverse_transform(kmeans.cluster_centers_), 
                                       columns=['Average Throughput', 'TCP Retransmission'])
        less_engaged_cluster_idx = cluster_centers['Average Throughput'].idxmin()
        less_engaged_cluster_center = cluster_centers.loc[less_engaged_cluster_idx]
        
        return less_engaged_cluster_center
    
    def calculate_engagement_score(self, less_engaged_cluster_center):
        # Calculate the Euclidean distance from each user to the less engaged cluster center
        def euclidean_distance(row):
            user_point = row[['Average Throughput', 'TCP Retransmission']].values
            return euclidean(user_point, less_engaged_cluster_center)
        
        self.df['Engagement Score'] = self.df.apply(euclidean_distance, axis=1)
    
    def run(self):
        self.preprocess_data()
        self.compute_experience_metrics()
        less_engaged_cluster_center = self.perform_clustering()
        self.calculate_engagement_score(less_engaged_cluster_center)
        return self.df[['MSISDN/Number', 'Engagement Score']]

