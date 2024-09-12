import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

class EngagementScoreCalculator:
    def __init__(self, df):
        self.df = df
        self.scaler = StandardScaler()
        self.processed_df = None  # To store preprocessed data

    def preprocess_data(self):
        # Handle missing values by filling them with the mean, only for numeric columns
        self.processed_df = self.df.copy()  # Work with a copy of the original DataFrame
        numeric_cols = self.processed_df.select_dtypes(include=[np.number]).columns
        self.processed_df[numeric_cols] = self.processed_df[numeric_cols].fillna(self.processed_df[numeric_cols].mean())
    
    def compute_experience_metrics(self):
        # Add calculated columns for Average Throughput and TCP Retransmission to processed data
        self.processed_df['Average Throughput'] = (self.processed_df['Avg Bearer TP DL (kbps)'] + 
                                                   self.processed_df['Avg Bearer TP UL (kbps)']) / 2
        self.processed_df['TCP Retransmission'] = (self.processed_df['TCP DL Retrans. Vol (Bytes)'] + 
                                                   self.processed_df['TCP UL Retrans. Vol (Bytes)']) / 2
    
    def perform_clustering(self):
        metrics = self.processed_df[['Average Throughput', 'TCP Retransmission']]
        scaled_metrics = self.scaler.fit_transform(metrics)
        
        kmeans = KMeans(n_clusters=3, random_state=42)
        self.processed_df['Cluster'] = kmeans.fit_predict(scaled_metrics)
        
        # Identify the less engaged cluster (lowest average throughput)
        cluster_centers = pd.DataFrame(self.scaler.inverse_transform(kmeans.cluster_centers_), 
                                       columns=['Average Throughput', 'TCP Retransmission'])
        less_engaged_cluster_idx = cluster_centers['Average Throughput'].idxmin()
        less_engaged_cluster_center = cluster_centers.loc[less_engaged_cluster_idx]
        
        # Identify the worst experience cluster (highest average TCP retransmission)
        worst_experience_cluster_idx = cluster_centers['TCP Retransmission'].idxmax()
        worst_experience_cluster_center = cluster_centers.loc[worst_experience_cluster_idx]
        
        return less_engaged_cluster_center, worst_experience_cluster_center
    
    def calculate_scores(self, less_engaged_cluster_center, worst_experience_cluster_center):
        # Calculate the Euclidean distance for Engagement Score using processed data
        def engagement_score(row):
            user_point = row[['Average Throughput', 'TCP Retransmission']].values
            return euclidean(user_point, less_engaged_cluster_center)
        
        # Calculate the Euclidean distance for Experience Score using processed data
        def experience_score(row):
            user_point = row[['Average Throughput', 'TCP Retransmission']].values
            return euclidean(user_point, worst_experience_cluster_center)
        
        self.processed_df['Engagement Score'] = self.processed_df.apply(engagement_score, axis=1)
        self.processed_df['Experience Score'] = self.processed_df.apply(experience_score, axis=1)
    
    def calculate_satisfaction_score(self):
        # Calculate the average of Engagement Score and Experience Score as Satisfaction Score
        self.processed_df['Satisfaction Score'] = (self.processed_df['Engagement Score'] + 
                                                   self.processed_df['Experience Score']) / 2
    
    def run(self):
        self.preprocess_data()  # First preprocess the data
        self.compute_experience_metrics()  # Compute experience-related metrics
        less_engaged_cluster_center, worst_experience_cluster_center = self.perform_clustering()  # Perform clustering
        self.calculate_scores(less_engaged_cluster_center, worst_experience_cluster_center)  # Calculate scores
        self.calculate_satisfaction_score()  # Calculate satisfaction score
        return self.processed_df[['MSISDN/Number', 'Engagement Score', 'Experience Score', 'Satisfaction Score']]  # Return the final DataFrame

    def get_top_10_satisfied_customers(self):
        # Sort by Satisfaction Score and get the top 10
        top_10_satisfied = self.processed_df.sort_values(by='Satisfaction Score', ascending=False).head(10)
        return top_10_satisfied[['MSISDN/Number', 'Satisfaction Score']]
    
    def run_kmeans_clustering(self):
        # Extract the relevant columns
        result_df = self.run()
        self.df['Engagement Score'] = result_df['Engagement Score']
        self.df['Experience Score'] = result_df['Experience Score']
        self.df['Satisfaction Score'] = result_df['Satisfaction Score']
        
        data = self.df[['Engagement Score', 'Experience Score']].dropna()
        
        # Fit K-means with k=2
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(data)
        
        # Add the cluster labels to the DataFrame
        self.df['Cluster'] = kmeans.labels_

        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.scatter(self.df['Engagement Score'], self.df['Experience Score'], c=self.df['Cluster'], cmap='viridis', marker='o')
        plt.xlabel('Engagement Score')
        plt.ylabel('Experience Score')
        plt.title('K-means Clustering (k=2) of Engagement and Experience Scores')
        plt.colorbar(label='Cluster')
        plt.show()

        # Aggregate the average Satisfaction Score and Experience Score per cluster
        cluster_summary = self.df.groupby('Cluster').agg({
            'Satisfaction Score': 'mean',
            'Experience Score': 'mean'
        }).reset_index()
        
        print("Cluster Summary:")
        print(cluster_summary)

