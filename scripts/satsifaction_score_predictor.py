from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import sys
import os

# Silence warnings and set environment variables
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Adjust to your CPU count
sys.path.append(os.path.abspath('../scripts'))

# Get the current directory
current_dir = os.getcwd()
print(current_dir)

# Get the parent directory
parent_dir = os.path.dirname(current_dir)
print(parent_dir)

# Insert the path to the parent directory
sys.path.insert(0, parent_dir)

from scripts.user_satsifaction_analysis import EngagementScoreCalculator

class SatisfactionScorePredictor:
    def __init__(self, df):
        self.df = df
        self.model = LinearRegression()
    
    def prepare_data(self):
        # Use 'Average Throughput', 'TCP Retransmission', 'Engagement Score', and 'Experience Score' as features
        self.df['Average Throughput'] = (self.df['Avg Bearer TP DL (kbps)'] + self.df['Avg Bearer TP UL (kbps)']) / 2
        self.df['TCP Retransmission'] = self.df['TCP DL Retrans. Vol (Bytes)'] + self.df['TCP UL Retrans. Vol (Bytes)']

        # Calculate Engagement Score and Experience Score
        calculator = EngagementScoreCalculator(self.df)
        result_df = calculator.run()
        self.df['Engagement Score'] = result_df['Engagement Score']
        self.df['Experience Score'] = result_df['Experience Score']
        self.df['Satisfaction Score'] = result_df['Satisfaction Score']

        # Drop rows with missing target values
        df_clean = self.df.dropna(subset=['Satisfaction Score'])

        # Extract numeric columns only
        numeric_df = df_clean.select_dtypes(include=['number'])

        # Fill missing values with the median of the respective columns
        numeric_df.fillna(numeric_df.median(), inplace=True)

        # Update the original DataFrame with imputed values
        self.df.update(numeric_df)

        # Prepare features and target
        features = self.df[['Average Throughput', 'TCP Retransmission', 'Engagement Score', 'Experience Score']]
        target = self.df['Satisfaction Score']

        # Drop rows with missing values in features
        df_final = features.copy()
        df_final['Satisfaction Score'] = target
        df_final = df_final.dropna()

        # Separate features and target
        features = df_final[['Average Throughput', 'TCP Retransmission', 'Engagement Score', 'Experience Score']]
        target = df_final['Satisfaction Score']

        # Split the data into training (80%) and testing sets (20%)
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test


    def train_model(self, X_train, y_train):
        # Train the Linear Regression model
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        # Predict Satisfaction Score on the test set
        y_pred = self.model.predict(X_test)
        
        # Calculate Mean Squared Error and R-squared
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Mean Squared Error: {mse}")
        print(f"R-Squared: {r2}")
    
    def predict(self, new_data):
        # Predict Satisfaction Score for new data
        return self.model.predict(new_data)

    def run(self):
        # Prepare the data and split into training and testing sets
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        # Train the model
        self.train_model(X_train, y_train)
        
        # Evaluate the model
        self.evaluate_model(X_test, y_test)

    def run(self):
        # Start an MLFlow run
        with mlflow.start_run():
            # Prepare the data and split into training and testing sets
            X_train, X_test, y_train, y_test = self.prepare_data()

            # Log parameters
            mlflow.log_param("model_type", "Linear Regression")
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)

            # Train the model
            self.train_model(X_train, y_train)

            # Log model
            mlflow.sklearn.log_model(self.model, "model")

            # Evaluate the model
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Log metrics
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2", r2)

            print(f"Mean Squared Error: {mse}")
            print(f"R-Squared: {r2}")

            # Save artifacts like the dataset or the final DataFrame
            self.df.to_csv("data.csv", index=False)
            mlflow.log_artifact("data.csv")


