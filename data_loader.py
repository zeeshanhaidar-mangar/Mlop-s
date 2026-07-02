import pandas as pd
from sqlalchemy import create_engine
from config import DB_CONFIG

def load_data_from_db():
    """
    Connects to the XAMPP MySQL database and loads the iris table.
    """
    try:
        # Create connection string for SQLAlchemy
        # Format: mysql+mysqlconnector://user:password@host/database
        user = DB_CONFIG["user"]
        password = DB_CONFIG["password"]
        host = DB_CONFIG["host"]
        database = DB_CONFIG["database"]
        table = DB_CONFIG["table"]

        connection_string = f"mysql+mysqlconnector://{user}:{password}@{host}/{database}"
        engine = create_engine(connection_string)

        # Load data using pandas
        query = f"SELECT * FROM {table}"
        df = pd.read_sql(query, engine)
        
        print(f"Successfully loaded {len(df)} rows from table '{table}'")
        return df

    except Exception as e:
        print(f"Error loading data from database: {e}")
        return None

if __name__ == "__main__":
    data = load_data_from_db()
    if data is not None:
        print(data.head())
