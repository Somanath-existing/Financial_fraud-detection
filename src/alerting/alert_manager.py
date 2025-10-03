"""
Integrated Alert Manager
- Manages different types of alerts for fraud detection
- Integrates with ML models and dashboard
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from alerting.alert import send_alert
import sqlite3
from datetime import datetime
import json

class AlertManager:
    def __init__(self, 
                 sender_email="somanathks711@gmail.com",
                 sender_password="rwpsibtrmznlukat",
                 default_recipient="somanathks7111@gmail.com",
                 db_path="d:/fraud_detection/src/etl/fraud_detection_db.sqlite"):
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.default_recipient = default_recipient
        self.db_path = db_path
        self.setup_alerts_table()
    
    def setup_alerts_table(self):
        """Create alerts table in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transaction_id TEXT,
                    alert_type TEXT,
                    severity TEXT,
                    message TEXT,
                    recipient_email TEXT,
                    sent_at TEXT,
                    status TEXT
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error setting up alerts table: {e}")
    
    def send_fraud_alert(self, transaction_data, prediction_result, recipient=None):
        """Send fraud detection alert"""
        try:
            recipient = recipient or self.default_recipient
            
            # Determine severity based on probability
            probability = prediction_result.get('probability', 0)
            if probability > 0.8:
                severity = "HIGH"
                alert_type = "CRITICAL_FRAUD"
            elif probability > 0.5:
                severity = "MEDIUM"
                alert_type = "SUSPECTED_FRAUD"
            else:
                severity = "LOW"
                alert_type = "ANOMALY_DETECTED"
            
            # Create detailed message
            message = f"""
🚨 FRAUD ALERT - {severity} PRIORITY 🚨

Transaction Details:
- Transaction ID: {transaction_data.get('transaction_id', 'N/A')}
- Amount: ${transaction_data.get('amount', 'N/A'):,.2f}
- Time: {transaction_data.get('timestamp', datetime.now())}
- Location: {transaction_data.get('location', 'N/A')}
- Device: {transaction_data.get('device', 'N/A')}
- Customer ID: {transaction_data.get('customer_id', 'N/A')}

Fraud Analysis:
- Fraud Probability: {probability:.1%}
- Model Used: {prediction_result.get('model_used', 'N/A')}
- Risk Level: {severity}

Immediate Actions Required:
1. Review transaction details
2. Contact customer for verification
3. Consider temporary account restrictions
4. Investigate related transactions

This is an automated alert from the Fraud Detection System.
Please take appropriate action immediately.
            """
            
            # Send email alert
            send_alert(message, recipient, self.sender_email, self.sender_password)
            
            # Log alert in database
            self.log_alert(
                transaction_data.get('transaction_id', ''),
                alert_type,
                severity,
                message,
                recipient,
                "SENT"
            )
            
            print(f"✅ {severity} priority fraud alert sent for transaction {transaction_data.get('transaction_id', 'N/A')}")
            return True
            
        except Exception as e:
            print(f"❌ Error sending fraud alert: {e}")
            self.log_alert(
                transaction_data.get('transaction_id', ''),
                alert_type,
                severity,
                str(e),
                recipient,
                "FAILED"
            )
            return False
    
    def send_system_alert(self, message, alert_type="SYSTEM", recipient=None):
        """Send system-level alerts"""
        try:
            recipient = recipient or self.default_recipient
            
            full_message = f"""
🔧 SYSTEM ALERT - {alert_type} 🔧

{message}

System Details:
- Timestamp: {datetime.now().isoformat()}
- Alert Type: {alert_type}

This is an automated system alert from the Fraud Detection System.
            """
            
            send_alert(full_message, recipient, self.sender_email, self.sender_password)
            self.log_alert("SYSTEM", alert_type, "INFO", full_message, recipient, "SENT")
            print(f"✅ System alert sent: {alert_type}")
            return True
            
        except Exception as e:
            print(f"❌ Error sending system alert: {e}")
            return False
    
    def send_daily_summary(self, recipient=None):
        """Send daily fraud detection summary"""
        try:
            recipient = recipient or self.default_recipient
            
            # Get daily stats from database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get today's predictions
            today = datetime.now().date().isoformat()
            cursor.execute("""
                SELECT COUNT(*) as total, 
                       SUM(predicted_fraud) as fraud_detected,
                       AVG(fraud_probability) as avg_prob
                FROM predictions 
                WHERE DATE(processed_at) = ?
            """, (today,))
            
            stats = cursor.fetchone()
            total_transactions = stats[0] or 0
            fraud_detected = stats[1] or 0
            avg_probability = stats[2] or 0
            
            # Get alerts sent today
            cursor.execute("""
                SELECT COUNT(*) as alerts_sent
                FROM alerts 
                WHERE DATE(sent_at) = ? AND status = 'SENT'
            """, (today,))
            
            alerts_sent = cursor.fetchone()[0] or 0
            conn.close()
            
            fraud_rate = (fraud_detected / total_transactions * 100) if total_transactions > 0 else 0
            
            message = f"""
📊 DAILY FRAUD DETECTION SUMMARY 📊
Date: {today}

Transaction Statistics:
- Total Transactions Processed: {total_transactions:,}
- Fraudulent Transactions Detected: {fraud_detected:,}
- Fraud Detection Rate: {fraud_rate:.2f}%
- Average Fraud Probability: {avg_probability:.1%}

Alert Statistics:
- Total Alerts Sent: {alerts_sent:,}

System Status: ✅ OPERATIONAL

This is your daily summary from the Fraud Detection System.
            """
            
            send_alert(message, recipient, self.sender_email, self.sender_password)
            self.log_alert("SYSTEM", "DAILY_SUMMARY", "INFO", message, recipient, "SENT")
            print("✅ Daily summary sent")
            return True
            
        except Exception as e:
            print(f"❌ Error sending daily summary: {e}")
            return False
    
    def log_alert(self, transaction_id, alert_type, severity, message, recipient, status):
        """Log alert in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO alerts (transaction_id, alert_type, severity, message, recipient_email, sent_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (transaction_id, alert_type, severity, message, recipient, datetime.now().isoformat(), status))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error logging alert: {e}")
    
    def get_alert_history(self, days=7):
        """Get alert history from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM alerts 
                WHERE sent_at >= date('now', '-{} days')
                ORDER BY sent_at DESC
            """.format(days))
            
            alerts = cursor.fetchall()
            conn.close()
            return alerts
        except Exception as e:
            print(f"Error getting alert history: {e}")
            return []
    
    def test_alert_system(self):
        """Test the alert system"""
        print("🧪 Testing alert system...")
        
        # Test transaction data
        test_transaction = {
            'transaction_id': 'TEST_001',
            'amount': 5000.0,
            'timestamp': datetime.now().isoformat(),
            'location': 'TEST_LOCATION',
            'device': 'TEST_DEVICE',
            'customer_id': 'TEST_CUSTOMER'
        }
        
        test_prediction = {
            'prediction': 1,
            'probability': 0.85,
            'model_used': 'test_model'
        }
        
        # Send test fraud alert
        success = self.send_fraud_alert(test_transaction, test_prediction)
        if success:
            print("✅ Test fraud alert sent successfully")
        else:
            print("❌ Test fraud alert failed")
        
        # Send test system alert
        success = self.send_system_alert("This is a test system alert", "TEST")
        if success:
            print("✅ Test system alert sent successfully")
        else:
            print("❌ Test system alert failed")

def main():
    """Test the alert manager"""
    alert_manager = AlertManager()
    alert_manager.test_alert_system()

if __name__ == "__main__":
    main()