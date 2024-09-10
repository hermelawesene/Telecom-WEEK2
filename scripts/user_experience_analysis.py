import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class TelecomUserExperience:
    def __init__(self, df):
        self.df = df
    
    def handle_missing_values(self):
        # Replace missing values with the mean for numerical columns
        numeric_cols = ['Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)', 
                        'Avg Bearer TP UL (kbps)', 'TCP DL Retrans. Vol (Bytes)', 
                        'TCP UL Retrans. Vol (Bytes)']
        self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
        
        # Replace missing values with the mode for categorical columns
        self.df['Handset Type'] = self.df['Handset Type'].fillna(self.df['Handset Type'].mode()[0])
        
    def handle_outliers(self):
        # Define thresholds for outliers or replace with mean
        for col in ['Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)', 
                    'Avg Bearer TP UL (kbps)', 'TCP DL Retrans. Vol (Bytes)', 
                    'TCP UL Retrans. Vol (Bytes)']:
            mean_val = self.df[col].mean()
            std_dev = self.df[col].std()
            upper_limit = mean_val + 3 * std_dev
            lower_limit = mean_val - 3 * std_dev
            
            # Capping the outliers
            self.df[col] = np.where(self.df[col] > upper_limit, mean_val, self.df[col])
            self.df[col] = np.where(self.df[col] < lower_limit, mean_val, self.df[col])
    
    def aggregate_per_customer(self):
        # Handle missing values
        self.handle_missing_values()
        
        # Handle outliers
        self.handle_outliers()
        
        # Aggregate information per customer
        self.df['Average TCP Retransmission'] = (self.df['TCP DL Retrans. Vol (Bytes)'] + 
                                                 self.df['TCP UL Retrans. Vol (Bytes)']) / 2
        self.df['Average RTT'] = (self.df['Avg RTT DL (ms)'] + self.df['Avg RTT UL (ms)']) / 2
        self.df['Average Throughput'] = (self.df['Avg Bearer TP DL (kbps)'] + 
                                         self.df['Avg Bearer TP UL (kbps)']) / 2
        
        # Group by 'MSISDN/Number'
        customer_agg = self.df.groupby('MSISDN/Number').agg({
            'Average TCP Retransmission': 'mean',
            'Average RTT': 'mean',
            'Average Throughput': 'mean',
            'Handset Type': lambda x: x.mode()[0]  # most common handset type
        }).reset_index()
        
        return customer_agg
    
    def top_bottom_frequent(self, series):
        # Top 10 values
        top_10 = series.sort_values(ascending=False).head(10)
        
        # Bottom 10 values
        bottom_10 = series.sort_values(ascending=True).head(10)
        
        # Most frequent 10 values
        most_frequent_10 = series.value_counts().head(10)
        
        return top_10, bottom_10, most_frequent_10
    
    def analyze_metrics(self):
        # Aggregate the data
        customer_agg = self.aggregate_per_customer()
        
        # Analyze TCP
        tcp_values = customer_agg['Average TCP Retransmission']
        tcp_top_10, tcp_bottom_10, tcp_most_frequent_10 = self.top_bottom_frequent(tcp_values)
        print("TCP - Top 10:\n", tcp_top_10)
        print("TCP - Bottom 10:\n", tcp_bottom_10)
        print("TCP - Most Frequent 10:\n", tcp_most_frequent_10)
        
        # Analyze RTT
        rtt_values = customer_agg['Average RTT']
        rtt_top_10, rtt_bottom_10, rtt_most_frequent_10 = self.top_bottom_frequent(rtt_values)
        print("\nRTT - Top 10:\n", rtt_top_10)
        print("RTT - Bottom 10:\n", rtt_bottom_10)
        print("RTT - Most Frequent 10:\n", rtt_most_frequent_10)
        
        # Analyze Throughput
        throughput_values = customer_agg['Average Throughput']
        throughput_top_10, throughput_bottom_10, throughput_most_frequent_10 = self.top_bottom_frequent(throughput_values)
        print("\nThroughput - Top 10:\n", throughput_top_10)
        print("Throughput - Bottom 10:\n", throughput_bottom_10)
        print("Throughput - Most Frequent 10:\n", throughput_most_frequent_10)

    def compute_distribution(self):
        # Compute the distribution of average throughput per handset type
        throughput_distribution = self.df.groupby('Handset Type')['Average Throughput'].mean().sort_values(ascending=False)
        print("Average Throughput per Handset Type:")
        print(throughput_distribution)
        
        # Compute the average TCP retransmission per handset type
        tcp_retransmission_distribution = self.df.groupby('Handset Type')['Average TCP Retransmission'].mean().sort_values(ascending=False)
        print("\nAverage TCP Retransmission per Handset Type:")
        print(tcp_retransmission_distribution)
        
    def report_distribution(self):
        # Compute the aggregated data
        customer_agg = self.aggregate_per_customer()
        
        # Compute the distribution for the aggregated dataset
        throughput_distribution = customer_agg.groupby('Handset Type')['Average Throughput'].mean().sort_values(ascending=False)
        tcp_retransmission_distribution = customer_agg.groupby('Handset Type')['Average TCP Retransmission'].mean().sort_values(ascending=False)
        
        # Print the results
        print("Distribution of Average Throughput per Handset Type:")
        print(throughput_distribution)
        
        print("\nAverage TCP Retransmission per Handset Type:")
        print(tcp_retransmission_distribution)

    def perform_clustering(self, k=3):
            # Aggregate the data
            customer_agg = self.aggregate_per_customer()

            # Prepare the data for clustering
            features = customer_agg[['Average TCP Retransmission', 'Average Throughput']]
            
            # Standardize the features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Perform K-Means clustering
            kmeans = KMeans(n_clusters=k, random_state=42)
            customer_agg['Cluster'] = kmeans.fit_predict(features_scaled)
            
            # Get cluster centers in original scale
            cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
            
            # Describe clusters
            cluster_descriptions = pd.DataFrame(cluster_centers, columns=['Average TCP Retransmission', 'Average Throughput'])
            cluster_descriptions['Cluster'] = range(k)

            # Print cluster centers
            print("\nCluster Centers (Original Scale):")
            print(cluster_descriptions)
            
            # Print average metrics per cluster
            for cluster in range(k):
                cluster_data = customer_agg[customer_agg['Cluster'] == cluster]
                avg_tcp = cluster_data['Average TCP Retransmission'].mean()
                avg_throughput = cluster_data['Average Throughput'].mean()
                print(f"\nCluster {cluster}:")
                print(f"Average TCP Retransmission: {avg_tcp:.2f}")
                print(f"Average Throughput: {avg_throughput:.2f}")

            # Plotting clusters
            plt.figure(figsize=(10, 6))
            plt.scatter(customer_agg['Average Throughput'], customer_agg['Average TCP Retransmission'], 
                        c=customer_agg['Cluster'], cmap='viridis', marker='o')
            plt.scatter(cluster_centers[:, 1], cluster_centers[:, 0], c='red', marker='x', s=200, label='Centroids')
            plt.xlabel('Average Throughput (kbps)')
            plt.ylabel('Average TCP Retransmission (Bytes)')
            plt.title('K-Means Clustering of User Experience')
            plt.legend()
            plt.show()
            