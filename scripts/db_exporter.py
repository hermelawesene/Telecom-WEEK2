from sqlalchemy import create_engine
import pandas as pd
import pymysql

def export_to_mysql(df, db_user, db_password, db_host, db_name, table_name):
    """
    Exports a DataFrame to a MySQL database with connection pooling, chunking, and error handling.

    :param df: DataFrame containing the data to export
    :param db_user: MySQL database username
    :param db_password: MySQL database password
    :param db_host: MySQL database host (e.g., 'localhost')
    :param db_name: Name of the MySQL database
    :param table_name: Name of the table where data will be inserted
    """
    try:
        # Create SQLAlchemy engine for MySQL connection with pool_recycle and pool_pre_ping
        engine = create_engine(
            f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",
            pool_recycle=18000,  # Recycle connections every 5 hours
            pool_pre_ping=True   # Check if connection is alive before using
        )

        # Export the dataframe to MySQL in chunks to avoid timeout issues
        df[['MSISDN/Number', 'Engagement Score', 'Experience Score', 'Satisfaction Score']].to_sql(
            table_name,
            con=engine,
            if_exists='replace',  # Options: 'fail', 'replace', 'append'
            index=False,
            chunksize=500  # Exporting the data in smaller chunks of 500 rows
        )

        print(f"Data successfully exported to {table_name} in the {db_name} database.")

    except pymysql.MySQLError as e:
        print(f"Error occurred while connecting to MySQL: {e}")
    except Exception as e:
        print(f"General error occurred: {e}")

# Example call to the function
# export_to_mysql(result_df, db_user, db_password, db_host, db_name, table_name)
