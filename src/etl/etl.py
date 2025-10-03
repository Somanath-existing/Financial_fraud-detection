"""
ETL module for Financial Fraud Detection
- Loads, cleans, and stores transaction data in SQLite
"""
import sqlite3
import pandas as pd

def run_etl(csv_path, db_path="fraud_detection_db.sqlite"):
    import os
    print(f"Checking if CSV exists: {csv_path} -> {os.path.exists(csv_path)}")
    if not os.path.exists(csv_path):
        print("CSV file not found! Aborting ETL.")
        return
    df = pd.read_csv(csv_path)
    print(f"Loaded CSV with shape: {df.shape}")
    print("First 5 rows:")
    print(df.head())

    # Handle missing values
    df = df.fillna(method='ffill').fillna(method='bfill')

    # Rename columns to match expected schema
    # Kaggle columns: Time, V1...V28, Amount, Class
    df = df.rename(columns={
        'Time': 'timestamp',
        'Amount': 'amount',
        'Class': 'is_fraud'
    })
    df['transaction_id'] = df.index
    df['customer_id'] = 'unknown'  # Placeholder, as dataset has no customer_id
    df['location'] = 'unknown'     # Placeholder
    df['device'] = 'unknown'       # Placeholder

    # Data normalization
    df['amount_norm'] = (df['amount'] - df['amount'].min()) / (df['amount'].max() - df['amount'].min())

    # Feature engineering: extract hour from timestamp (Time is seconds elapsed)
    df['hour'] = (df['timestamp'] // 3600).astype(int)

    # Example anomaly feature: transaction amount above 99th percentile
    threshold = df['amount'].quantile(0.99)
    df['is_anomaly'] = (df['amount'] > threshold).astype(int)

    # Select columns for DB
    db_cols = [
        'transaction_id', 'customer_id', 'amount', 'timestamp', 'location', 'device',
        'is_fraud', 'amount_norm', 'hour', 'is_anomaly'
    ]
    df_db = df[db_cols]

    # Use absolute path for database
    abs_db_path = "d:/fraud_detection/src/etl/fraud_detection_db.sqlite"
    print(f"Using database path: {abs_db_path}")
    conn = sqlite3.connect(abs_db_path)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        transaction_id INTEGER,
        customer_id TEXT,
        amount REAL,
        timestamp TEXT,
        location TEXT,
        device TEXT,
        is_fraud INTEGER,
        amount_norm REAL,
        hour INTEGER,
        is_anomaly INTEGER
    )
    """)
    # Insert the cleaned data
    df_db.to_sql("transactions", conn, if_exists="replace", index=False)
    conn.commit()
    # Print tables for verification
    conn2 = sqlite3.connect(db_path)
    cursor2 = conn2.cursor()
    cursor2.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor2.fetchall()
    print("Tables in database:", tables)
    conn2.close()
    print("ETL process completed.")

if __name__ == "__main__":
    run_etl("d:/fraud_detection/data/creditcard.csv")
