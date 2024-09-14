# telecom_analysis.py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class TelecomEDA:
    def __init__(self, df):
        self.df = df

    def describe_data(self):
        """Describe all relevant variables and data types."""
        return self.df.info(), self.df.describe()

    def handle_missing_values(self):
        """Identify and treat missing values."""
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_columns] = self.df[numeric_columns].fillna(self.df[numeric_columns].mean())
        #self.df.fillna(self.df.mean(), inplace=True)

    def handle_outliers(self):
        """Identify and treat outliers."""
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        z_scores = np.abs(stats.zscore(self.df[numeric_columns]))
        self.df = self.df[(z_scores < 3).all(axis=1)]

    def variable_transformation(self):
        """Segment users into deciles based on total duration."""
        self.df.loc[:, 'Total_Duration'] = self.df['Activity Duration DL (ms)'] + self.df['Activity Duration UL (ms)']
        self.df.loc[:, 'Decile'] = pd.qcut(self.df['Total_Duration'], 5, labels=False)
        decile_summary = self.df.groupby('Decile').agg({'Total_Duration': 'sum'}).reset_index()
        return decile_summary
    

    def basic_metrics(self):
        """Analyze basic metrics."""
        return self.df.mean(), self.df.median()

    def correlation_analysis(self):
        """Compute the correlation matrix."""
        return self.df[['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']].corr()

    def perform_pca(self):
        """Perform Principal Component Analysis (PCA)."""
        features = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']
        x = self.df[features]
        x = StandardScaler().fit_transform(x)

        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(x)
        return pca.explained_variance_ratio_

if __name__ == "__main__":
    analysis = TelecomEDA("telecom_data.csv")
    analysis.handle_missing_values()
    analysis.handle_outliers()
    print(analysis.describe_data())
    print(analysis.variable_transformation())
    print(analysis.basic_metrics())
    print(analysis.correlation_analysis())
    print(analysis.perform_pca())
