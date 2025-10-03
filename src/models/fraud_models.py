"""
Fraud Detection Models
- Implements ML algorithms for fraud detection
- Supports Logistic Regression, Random Forest, XGBoost, Isolation Forest
"""
import pandas as pd
import numpy as np
import sqlite3
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionModel:
    def __init__(self, db_path="d:/fraud_detection/src/etl/fraud_detection_db.sqlite"):
        self.db_path = db_path
        self.models = {}
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load data from SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query("SELECT * FROM transactions", conn)
            conn.close()
            print(f"Loaded {len(df)} transactions from database")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def prepare_features(self, df):
        """Prepare features for model training"""
        # Use all V columns (PCA features from Kaggle dataset) plus engineered features
        feature_cols = [col for col in df.columns if col.startswith('V')] + ['amount_norm', 'hour']
        X = df[feature_cols].fillna(0)
        y = df['is_fraud'] if 'is_fraud' in df.columns else None
        return X, y
    
    def train_supervised_models(self, df):
        """Train supervised learning models"""
        print("Training supervised models...")
        X, y = self.prepare_features(df)
        
        if y is None:
            print("No fraud labels found for supervised training.")
            return None
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models
        models_to_train = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        }
        
        results = {}
        for name, model in models_to_train.items():
            print(f"\nTraining {name}...")
            
            # Use scaled data for logistic regression, original for tree-based
            train_X = X_train_scaled if name == 'logistic_regression' else X_train
            test_X = X_test_scaled if name == 'logistic_regression' else X_test
            
            model.fit(train_X, y_train)
            predictions = model.predict(test_X)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, predictions)
            roc_auc = roc_auc_score(y_test, model.predict_proba(test_X)[:, 1])
            
            print(f"{name} - Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")
            print(f"Classification Report:\n{classification_report(y_test, predictions)}")
            
            self.models[name] = model
            results[name] = {'accuracy': accuracy, 'roc_auc': roc_auc}
            
        return results
    
    def train_unsupervised_models(self, df):
        """Train unsupervised anomaly detection models"""
        print("\nTraining unsupervised models...")
        X, _ = self.prepare_features(df)
        
        # Scale features for unsupervised learning
        X_scaled = self.scaler.fit_transform(X)
        
        models_to_train = {
            'isolation_forest': IsolationForest(contamination=0.001, random_state=42),
            'one_class_svm': OneClassSVM(nu=0.001)
        }
        
        for name, model in models_to_train.items():
            print(f"\nTraining {name}...")
            model.fit(X_scaled)
            anomaly_predictions = model.predict(X_scaled)
            
            # Convert to binary (1 for normal, 0 for anomaly)
            anomaly_binary = (anomaly_predictions == 1).astype(int)
            anomaly_count = len(anomaly_predictions) - anomaly_binary.sum()
            
            print(f"{name} detected {anomaly_count} anomalies out of {len(X)} transactions")
            print(f"Anomaly rate: {anomaly_count/len(X)*100:.2f}%")
            
            self.models[name] = model
    
    def predict_fraud(self, transaction_data, model_name='random_forest'):
        """Predict fraud for new transaction data"""
        if model_name not in self.models:
            print(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
            return None
            
        model = self.models[model_name]
        
        # Prepare features
        if isinstance(transaction_data, dict):
            # Convert single transaction to DataFrame
            transaction_data = pd.DataFrame([transaction_data])
        
        X, _ = self.prepare_features(transaction_data)
        
        # Scale if needed
        if model_name in ['logistic_regression', 'isolation_forest', 'one_class_svm']:
            X = self.scaler.transform(X)
            
        if model_name in ['isolation_forest', 'one_class_svm']:
            # Unsupervised models return -1 for anomalies, 1 for normal
            predictions = model.predict(X)
            fraud_predictions = (predictions == -1).astype(int)
        else:
            # Supervised models
            fraud_predictions = model.predict(X)
            
        return fraud_predictions
    
    def save_models(self, save_dir="d:/fraud_detection/models/saved/"):
        """Save trained models to disk"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for name, model in self.models.items():
            model_path = os.path.join(save_dir, f"{name}_model.pkl")
            joblib.dump(model, model_path)
            print(f"Saved {name} model to {model_path}")
        
        # Save scaler
        scaler_path = os.path.join(save_dir, "scaler.pkl")
        joblib.dump(self.scaler, scaler_path)
        print(f"Saved scaler to {scaler_path}")
    
    def load_models(self, save_dir="d:/fraud_detection/models/saved/"):
        """Load trained models from disk"""
        import os
        if not os.path.exists(save_dir):
            print(f"Model directory {save_dir} not found")
            return
            
        for file in os.listdir(save_dir):
            if file.endswith('_model.pkl'):
                model_name = file.replace('_model.pkl', '')
                model_path = os.path.join(save_dir, file)
                self.models[model_name] = joblib.load(model_path)
                print(f"Loaded {model_name} model")
        
        # Load scaler
        scaler_path = os.path.join(save_dir, "scaler.pkl")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print("Loaded scaler")

def main():
    """Main function to train and evaluate models"""
    detector = FraudDetectionModel()
    
    # Load data
    df = detector.load_data()
    if df is None:
        return
    
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud rate: {df['is_fraud'].mean()*100:.2f}%")
    
    # Train models
    supervised_results = detector.train_supervised_models(df)
    detector.train_unsupervised_models(df)
    
    # Save models
    detector.save_models()
    
    print("\nModel training completed!")
    return detector

if __name__ == "__main__":
    detector = main()
