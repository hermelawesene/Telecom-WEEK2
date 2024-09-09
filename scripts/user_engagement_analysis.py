import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

class TelecomUserEngagement:
    def __init__(self, df):
        self.df = df
        self.aggregated_data = None

    def aggregate_metrics(self):
        # Aggregate the metrics per customer id (MSISDN)
        self.df['Total Duration (s)'] = self.df['Dur. (ms)'] / 1000
        self.df['Total Traffic (Bytes)'] = self.df['Total UL (Bytes)'] + self.df['Total DL (Bytes)']
        self.aggregated_data = self.df.groupby('MSISDN/Number').agg({
            'Total Duration (s)': 'sum',
            'Bearer Id': 'count',
            'Total Traffic (Bytes)': 'sum'
        }).rename(columns={'Bearer Id': 'Session Frequency'})
        return self.aggregated_data

    def top_customers(self, metric, top_n=10):
        # Return the top N customers based on the specified metric
        if self.aggregated_data is None:
            self.aggregate_metrics()
        return self.aggregated_data.nlargest(top_n, metric)

    def normalize_metrics(self):
        if self.aggregated_data is None:
            self.aggregate_metrics()
        # Select relevant columns for normalization
        metrics = self.aggregated_data[['Total Duration (s)', 'Session Frequency', 'Total Traffic (Bytes)']]
        
        # Apply Min-Max normalization
        scaler = MinMaxScaler()
        normalized_metrics = scaler.fit_transform(metrics)
        
        # Store the normalized metrics back in the DataFrame
        self.aggregated_data[['Normalized Duration', 'Normalized Frequency', 'Normalized Traffic']] = normalized_metrics
    
    def run_kmeans(self, n_clusters=3):
        if self.aggregated_data is None:
            self.aggregate_metrics()
        # Apply K-means clustering on the normalized metrics
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.aggregated_data['Engagement Cluster'] = kmeans.fit_predict(
            self.aggregated_data[['Normalized Duration', 'Normalized Frequency', 'Normalized Traffic']]
        )
    
    def compute_cluster_statistics(self):
        # Compute min, max, mean, and sum for each cluster
        cluster_stats = self.aggregated_data.groupby('Engagement Cluster').agg({
            'Total Duration (s)': ['min', 'max', 'mean', 'sum'],
            'Session Frequency': ['min', 'max', 'mean', 'sum'],
            'Total Traffic (Bytes)': ['min', 'max', 'mean', 'sum']
        })
        return cluster_stats
    
    def plot_clusters(self):
        if self.aggregated_data is None or 'Engagement Cluster' not in self.aggregated_data.columns:
            raise ValueError("Clusters have not been computed. Run the 'run_kmeans' method first.")
        
        # Plotting the clusters to visualize
        plt.figure(figsize=(8, 6))
        plt.scatter(
            self.aggregated_data['Normalized Duration'], 
            self.aggregated_data['Normalized Traffic'], 
            c=self.aggregated_data['Engagement Cluster'], cmap='viridis'
        )
        plt.xlabel('Normalized Duration')
        plt.ylabel('Normalized Traffic')
        plt.title('Customer Engagement Clusters')
        plt.colorbar(label='Cluster')
        plt.show()

    def plot_cluster_statistics(self, cluster_stats):
        # Plotting the statistics
        metrics = ['Total Duration (s)', 'Session Frequency', 'Total Traffic (Bytes)']
        stat_functions = ['min', 'max', 'mean', 'sum']

        for metric in metrics:
            plt.figure(figsize=(12, 8))
            for stat in stat_functions:
                plt.plot(cluster_stats.index, cluster_stats[(metric, stat)], label=f'{metric} - {stat}')
            plt.title(f'{metric} Statistics per Engagement Cluster')
            plt.xlabel('Engagement Cluster')
            plt.ylabel(f'{metric} Value')
            plt.legend()
            plt.grid(True)
            plt.show()

    def analyze_clusters(self):
        if self.aggregated_data is None or 'Engagement Cluster' not in self.aggregated_data.columns:
            raise ValueError("Clusters have not been computed. Run the 'run_kmeans' method first.")
        # Analyze the clusters to understand the engagement levels
        return self.aggregated_data.groupby('Engagement Cluster').mean()[['Total Duration (s)', 'Session Frequency', 'Total Traffic (Bytes)']]
    
    def aggregate_traffic_per_application(self):
        # Define the application columns
        application_columns = {
            'Social Media': 'Social Media DL (Bytes)',
            'Google': 'Google DL (Bytes)',
            'Email': 'Email DL (Bytes)',
            'Youtube': 'Youtube DL (Bytes)',
            'Netflix': 'Netflix DL (Bytes)',
            'Gaming': 'Gaming DL (Bytes)',
            'Other': 'Other DL (Bytes)'
        }
        
        # Create an empty DataFrame to store aggregated traffic per application
        application_traffic = pd.DataFrame()
        
        # Aggregate traffic data for each application
        for app, column in application_columns.items():
            app_traffic = self.df.groupby('MSISDN/Number').agg({
                column: 'sum'
            }).rename(columns={column: f'Total Traffic {app} (Bytes)'})
            app_traffic['Application'] = app
            application_traffic = pd.concat([application_traffic, app_traffic], axis=0)

        # Reset index to turn MSISDN/Number into a column
        application_traffic = application_traffic.reset_index()
        
        return application_traffic

    def top_users_per_application(self, top_n=10):
        application_traffic = self.aggregate_traffic_per_application()
        
        if 'Application' not in application_traffic.columns:
            raise ValueError("The 'Application' column is missing in the DataFrame.")
        
        top_users = pd.DataFrame()
        for app in application_traffic['Application'].dropna().unique():  # Ensure unique() is used on Series
            app_data = application_traffic[application_traffic['Application'] == app]
            top_users_app = app_data.nlargest(top_n, f'Total Traffic {app} (Bytes)')
            top_users = pd.concat([top_users, top_users_app], axis=0)
        
        return top_users
    
    # def plot_top_applications(self):
        application_traffic = self.aggregate_traffic_per_application()
        # Aggregate traffic data for each application
        total_traffic_per_app = application_traffic.groupby('Application').agg({
            'Total Traffic Social Media (Bytes)': 'sum',
            'Total Traffic Google (Bytes)': 'sum',
            'Total Traffic Email (Bytes)': 'sum',
            'Total Traffic Youtube (Bytes)': 'sum',
            'Total Traffic Netflix (Bytes)': 'sum',
            'Total Traffic Gaming (Bytes)': 'sum',
            'Total Traffic Other (Bytes)': 'sum'
        }).sum()

        # Find the top 3 applications with the highest total traffic
        top_3_apps = total_traffic_per_app.nlargest(3).index

        # Filter data for the top 3 applications
        top_3_data = application_traffic[application_traffic['Application'].isin(top_3_apps)]

        # Plot the top 3 applications
        for app in top_3_apps:
            plt.figure(figsize=(12, 6))
            app_data = top_3_data[top_3_data['Application'] == app]
            sns.barplot(x='MSISDN/Number', y=f'Total Traffic {app} (Bytes)', data=app_data, ci=None)
            plt.xticks(rotation=90)
            plt.title(f'Total Traffic for {app}')
            plt.xlabel('User (MSISDN/Number)')
            plt.ylabel('Total Traffic (Bytes)')
            plt.tight_layout()
            plt.show()

    def plot_top_applications(self):
        top_3_data = self.top_users_per_application(top_n=10)
        
        # Print the column names to confirm
        print(top_3_data.columns)
        
        # Plot the top 3 most used applications
        top_apps = top_3_data['Application'].value_counts().head(3).index
        for app in top_apps:
            plt.figure(figsize=(12, 6))
            
            # Make sure to correctly reference the columns
            app_col = f'Total Traffic {app} (Bytes)'
            
            if app_col in top_3_data.columns:
                app_data = top_3_data[top_3_data['Application'] == app]
                
                # Plot the data
                plt.bar(app_data['MSISDN/Number'], app_data[app_col], color='skyblue')
                plt.xlabel('MSISDN/Number')
                plt.ylabel(f'Total Traffic {app} (Bytes)')
                plt.title(f'Total Traffic for {app}')
                plt.xticks(rotation=90)
                plt.show()
            else:
                print(f"Column '{app_col}' does not exist in the DataFrame.")
