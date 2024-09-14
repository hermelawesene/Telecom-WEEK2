import pandas as pd
import numpy as np
import scipy.stats 
from scipy.stats import zscore


class OverviewAnalyzer:
    def __init__(self, data, bytes_data) :
        self.data = data
        self.bytes_data = bytes_data

    def missing_values_table(self, df):
        # Total missing values
        mis_val = df.isnull().sum()

        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)

        # dtype of missing values
        mis_val_dtype = df.dtypes

        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent, mis_val_dtype], axis=1)

        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values', 2: 'Dtype'})

        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
            '% of Total Values', ascending=False).round(1)

        # Print some summary information
        print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
            " columns that have missing values.")

        # Return the dataframe with missing information
        return mis_val_table_ren_columns

    def convert_bytes_to_megabytes(self, df, bytes_data):
        megabyte = 1 * 10e+5
        df[bytes_data] = df[bytes_data] / megabyte
        return df[bytes_data]


    def fix_outlier(df, column):
        df[column] = np.where(df[column] > df[column].quantile(0.95), df[column].median(), df[column])
        return df[column]

    def remove_outliers(df, column_to_process, z_threshold=3):
        # Apply outlier removal to the specified column
        z_scores = zscore(df[column_to_process])
        outlier_column = column_to_process + '_Outlier'
        df[outlier_column] = (np.abs(z_scores) > z_threshold).astype(int)
        df = df[df[outlier_column] == 0]  # Keep rows without outliers

        # Drop the outlier column as it's no longer needed
        df = df.drop(columns=[outlier_column], errors='ignore')

        return df
    

    def aggregate_user_behavior(self, df):
        # Group by 'MSISDN/Number' to aggregate data per user
        user_behavior = df.groupby('MSISDN/Number').agg(
            num_xdr_sessions=('Bearer Id', 'count'),
            total_session_duration=('Dur. (ms)', 'sum'),
            total_download_data=('Total DL (Bytes)', 'sum'),
            total_upload_data=('Total UL (Bytes)', 'sum'),
            social_media_dl=('Social Media DL (Bytes)', 'sum'),
            social_media_ul=('Social Media UL (Bytes)', 'sum'),
            youtube_dl=('Youtube DL (Bytes)', 'sum'),  # Use 'Youtube' instead of 'YouTube'
            youtube_ul=('Youtube UL (Bytes)', 'sum'),
            netflix_dl=('Netflix DL (Bytes)', 'sum'),
            netflix_ul=('Netflix UL (Bytes)', 'sum'),
            google_dl=('Google DL (Bytes)', 'sum'),
            google_ul=('Google UL (Bytes)', 'sum'),
            email_dl=('Email DL (Bytes)', 'sum'),
            email_ul=('Email UL (Bytes)', 'sum'),
            gaming_dl=('Gaming DL (Bytes)', 'sum'),
            gaming_ul=('Gaming UL (Bytes)', 'sum'),
            other_dl=('Other DL (Bytes)', 'sum'),
            other_ul=('Other UL (Bytes)', 'sum')
        ).reset_index()

        # Add a total data volume column
        user_behavior['total_data_volume'] = (
            user_behavior['total_download_data'] + user_behavior['total_upload_data']
        )

        # Add a total application data volume column
        user_behavior['total_application_data_volume'] = (
            user_behavior['social_media_dl'] + user_behavior['social_media_ul'] +
            user_behavior['youtube_dl'] + user_behavior['youtube_ul'] +
            user_behavior['netflix_dl'] + user_behavior['netflix_ul'] +
            user_behavior['google_dl'] + user_behavior['google_ul'] +
            user_behavior['email_dl'] + user_behavior['email_ul'] +
            user_behavior['gaming_dl'] + user_behavior['gaming_dl'] +
            user_behavior['other_dl'] + user_behavior['other_ul']
        )

        return user_behavior