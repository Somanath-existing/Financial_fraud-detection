"""
Real-time Transaction Monitoring with Fraud Detection
- Simulates streaming using file monitoring
- Integrates with trained ML models for real-time fraud detection
- Triggers alerts for suspicious transactions
"""
import pandas as pd
import numpy as np
import time
import os
import sqlite3
import joblib
from datetime import datetime
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class RealTimeMonitor:
    def __init__(self, 
                 input_dir="d:/fraud_detection/streaming_input/",
                 processed_dir="d:/fraud_detection/streaming_processed/",
                 model_dir="d:/fraud_detection/models/saved/",
                 db_path="d:/fraud_detection/src/etl/fraud_detection_db.sqlite"):
        self.input_dir = input_dir
        self.processed_dir = processed_dir
        self.model_dir = model_dir
        self.db_path = db_path
        self.models = {}
        self.scaler = None
        
        # Create directories
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        # Load trained models
        self.load_models()
    
    def load_models(self):
        """Load pre-trained fraud detection models"""
        try:
            # Load scaler
            scaler_path = os.path.join(self.model_dir, "scaler.pkl")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                print("✓ Loaded feature scaler")
            
            # Load models
            for file in os.listdir(self.model_dir):
                if file.endswith('_model.pkl'):
                    model_name = file.replace('_model.pkl', '')
                    model_path = os.path.join(self.model_dir, file)
                    self.models[model_name] = joblib.load(model_path)
                    print(f"✓ Loaded {model_name} model")
                    
            if not self.models:
                print("⚠ No trained models found. Please train models first.")
                
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def preprocess_transaction(self, transaction):
        """Preprocess a single transaction for prediction"""
        try:
            # Convert to DataFrame if it's a dict
            if isinstance(transaction, dict):
                df = pd.DataFrame([transaction])
            else:
                df = transaction.copy()
            
            # Feature engineering
            if 'timestamp' in df.columns:
                df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            else:
                df['hour'] = datetime.now().hour
            
            if 'amount' in df.columns:
                # Normalize amount (simple min-max scaling)
                df['amount_norm'] = df['amount'] / df['amount'].max() if df['amount'].max() > 0 else 0
            
            # Select features for prediction
            feature_cols = [col for col in df.columns if col.startswith('V')] + ['amount_norm', 'hour']
            available_cols = [col for col in feature_cols if col in df.columns]
            
            if not available_cols:
                # Create dummy features if none available
                for i in range(28):  # V1 to V28 from Kaggle dataset
                    df[f'V{i+1}'] = 0
                df['amount_norm'] = df.get('amount_norm', 0)
                df['hour'] = df.get('hour', 0)
                available_cols = [f'V{i+1}' for i in range(28)] + ['amount_norm', 'hour']
            
            X = df[available_cols].fillna(0)
            return X
            
        except Exception as e:
            print(f"Error preprocessing transaction: {e}")
            return None
    
    def predict_fraud(self, transaction, model_name='random_forest'):
        """Predict fraud probability for a transaction"""
        try:
            X = self.preprocess_transaction(transaction)
            if X is None:
                return None
            
            if model_name not in self.models:
                print(f"Model {model_name} not available. Using first available model.")
                if self.models:
                    model_name = list(self.models.keys())[0]
                else:
                    return None
            
            model = self.models[model_name]
            
            # Scale features if needed
            if model_name in ['logistic_regression'] and self.scaler:
                X_scaled = self.scaler.transform(X)
                prediction = model.predict(X_scaled)[0]
                prob = model.predict_proba(X_scaled)[0][1] if hasattr(model, 'predict_proba') else prediction
            elif model_name in ['isolation_forest', 'one_class_svm']:
                # Unsupervised models
                X_scaled = self.scaler.transform(X) if self.scaler else X
                prediction = 1 if model.predict(X_scaled)[0] == -1 else 0  # -1 means anomaly
                prob = prediction
            else:
                # Tree-based models
                prediction = model.predict(X)[0]
                prob = model.predict_proba(X)[0][1] if hasattr(model, 'predict_proba') else prediction
            
            return {
                'prediction': int(prediction),
                'probability': float(prob),
                'model_used': model_name
            }
            
        except Exception as e:
            print(f"Error predicting fraud: {e}")
            return None
    
    def trigger_alert(self, transaction, prediction_result):
        """Trigger alert for suspicious transactions"""
        try:
            from alerting.alert import send_alert
            
            transaction_details = f"""
            Transaction ID: {transaction.get('transaction_id', 'N/A')}
            Amount: ${transaction.get('amount', 'N/A')}
            Time: {transaction.get('timestamp', datetime.now())}
            Location: {transaction.get('location', 'N/A')}
            Device: {transaction.get('device', 'N/A')}
            Fraud Probability: {prediction_result['probability']:.2%}
            Model: {prediction_result['model_used']}
            """
            
            # Use your email credentials
            sender_email = "somanathks711@gmail.com"
            sender_password = "rwpsibtrmznlukat"
            recipient_email = "somanathks7111@gmail.com"
            
            send_alert(transaction_details, recipient_email, sender_email, sender_password)
            print(f"🚨 Alert sent for suspicious transaction: {transaction.get('transaction_id', 'N/A')}")
            
        except Exception as e:
            print(f"Error sending alert: {e}")
    
    def process_transaction_file(self, file_path):
        """Process a single transaction file"""
        try:
            # Read transaction data
            df = pd.read_csv(file_path)
            print(f"📄 Processing {len(df)} transactions from {os.path.basename(file_path)}")
            
            suspicious_count = 0
            
            for idx, transaction in df.iterrows():
                # Predict fraud
                result = self.predict_fraud(transaction.to_dict())
                
                if result and result['prediction'] == 1:
                    suspicious_count += 1
                    print(f"🚨 Suspicious transaction detected: ID {transaction.get('transaction_id', idx)} "
                          f"(Probability: {result['probability']:.2%})")
                    
                    # Trigger alert for high-probability fraud
                    if result['probability'] > 0.5:
                        self.trigger_alert(transaction.to_dict(), result)
                
                # Store result in database (optional)
                self.store_prediction(transaction.to_dict(), result)
            
            print(f"✅ Processed {len(df)} transactions, found {suspicious_count} suspicious")
            
            # Move processed file
            processed_path = os.path.join(self.processed_dir, os.path.basename(file_path))
            os.rename(file_path, processed_path)
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    
    def store_prediction(self, transaction, prediction_result):
        """Store prediction results in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create predictions table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transaction_id TEXT,
                    amount REAL,
                    timestamp TEXT,
                    predicted_fraud INTEGER,
                    fraud_probability REAL,
                    model_used TEXT,
                    processed_at TEXT
                )
            """)
            
            # Insert prediction
            cursor.execute("""
                INSERT INTO predictions 
                (transaction_id, amount, timestamp, predicted_fraud, fraud_probability, model_used, processed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                transaction.get('transaction_id', ''),
                transaction.get('amount', 0),
                transaction.get('timestamp', ''),
                prediction_result['prediction'] if prediction_result else 0,
                prediction_result['probability'] if prediction_result else 0,
                prediction_result['model_used'] if prediction_result else '',
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error storing prediction: {e}")
    
    def start_monitoring(self, check_interval=5):
        """Start real-time monitoring of input directory"""
        print(f"🔍 Starting real-time fraud monitoring...")
        print(f"📁 Watching directory: {self.input_dir}")
        print(f"⏱ Check interval: {check_interval} seconds")
        print(f"🤖 Loaded models: {list(self.models.keys())}")
        print("Press Ctrl+C to stop monitoring\n")
        
        try:
            while True:
                # Check for new CSV files
                csv_files = [f for f in os.listdir(self.input_dir) if f.endswith('.csv')]
                
                for file in csv_files:
                    file_path = os.path.join(self.input_dir, file)
                    print(f"📥 New file detected: {file}")
                    self.process_transaction_file(file_path)
                
                # Wait before next check
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
        except Exception as e:
            print(f"❌ Error during monitoring: {e}")

def create_sample_transactions(output_file="d:/fraud_detection/streaming_input/sample_transactions.csv"):
    """Create sample transaction data for testing"""
    import random
    
    # Generate sample transactions
    transactions = []
    for i in range(100):
        transaction = {
            'transaction_id': f'TXN_{i:06d}',
            'amount': random.uniform(1, 10000),
            'timestamp': datetime.now().isoformat(),
            'location': random.choice(['NY', 'CA', 'TX', 'FL', 'IL']),
            'device': random.choice(['Mobile', 'Desktop', 'ATM']),
            'customer_id': f'CUST_{random.randint(1000, 9999)}'
        }
        
        # Add some V features (simulated PCA components)
        for j in range(1, 29):
            transaction[f'V{j}'] = random.uniform(-3, 3)
        
        # Add fraud label (simulate 1% fraud rate)
        transaction['is_fraud'] = 1 if random.random() < 0.01 else 0
        
        transactions.append(transaction)
    
    # Save to CSV
    df = pd.DataFrame(transactions)
    df.to_csv(output_file, index=False)
    print(f"📊 Created sample transactions file: {output_file}")
    return output_file

def main():
    """Main function for testing streaming"""
    monitor = RealTimeMonitor()
    
    # Create sample data if needed
    sample_file = create_sample_transactions()
    
    print("Options:")
    print("1. Start real-time monitoring")
    print("2. Process sample file once")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        monitor.start_monitoring()
    elif choice == "2":
        monitor.process_transaction_file(sample_file)
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
