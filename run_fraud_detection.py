"""
Main Runner Script for Financial Fraud Detection System
- Provides a unified interface to run all components
- Includes setup, training, monitoring, and dashboard options
"""
import os
import sys
import subprocess
import time
from datetime import datetime

def print_banner():
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║         🔍 FINANCIAL FRAUD DETECTION SYSTEM 🔍               ║
║                                                               ║
║         Advanced ML-Powered Fraud Detection                   ║
║         Real-time Monitoring & Intelligent Alerts            ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """)

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'pandas', 'scikit-learn', 'numpy', 'joblib', 
        'streamlit', 'plotly', 'altair', 'pyspark'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("📦 Installing missing packages...")
        for package in missing_packages:
            subprocess.run([sys.executable, '-m', 'pip', 'install', package])
        print("✅ All packages installed!")
    else:
        print("✅ All dependencies satisfied!")

def check_data_files():
    """Check if required data files exist"""
    data_file = "d:/fraud_detection/data/creditcard.csv"
    if not os.path.exists(data_file):
        print(f"❌ Data file missing: {data_file}")
        print("📥 Please download the Kaggle Credit Card Fraud Dataset:")
        print("   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("   Place creditcard.csv in the data/ directory")
        return False
    
    print("✅ Data files found!")
    return True

def setup_project():
    """Setup the project structure and run initial ETL"""
    print("\n🚀 Setting up Financial Fraud Detection System...")
    
    # Check dependencies
    check_dependencies()
    
    # Check data files
    if not check_data_files():
        return False
    
    # Create necessary directories
    dirs = [
        "d:/fraud_detection/models/saved",
        "d:/fraud_detection/streaming_input",
        "d:/fraud_detection/streaming_processed"
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    print("📁 Project directories created!")
    
    # Run ETL process
    print("🔄 Running ETL process...")
    try:
        subprocess.run([sys.executable, "src/etl/etl.py"], cwd="d:/fraud_detection")
        print("✅ ETL process completed!")
    except Exception as e:
        print(f"❌ ETL process failed: {e}")
        return False
    
    return True

def train_models():
    """Train machine learning models"""
    print("\n🤖 Training ML models...")
    try:
        subprocess.run([sys.executable, "src/models/fraud_models.py"], cwd="d:/fraud_detection")
        print("✅ Model training completed!")
        return True
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        return False

def start_dashboard():
    """Start the Streamlit dashboard"""
    print("\n📊 Starting dashboard...")
    print("🌐 Dashboard will be available at: http://localhost:8501")
    print("🔐 Login credentials: admin / admin123")
    print("⏹️  Press Ctrl+C to stop the dashboard")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard/dashboard.py"], 
                      cwd="d:/fraud_detection")
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")

def start_monitoring():
    """Start real-time monitoring"""
    print("\n🔍 Starting real-time monitoring...")
    print("📁 Monitoring directory: d:/fraud_detection/streaming_input/")
    print("📧 Alerts will be sent for suspicious transactions")
    print("⏹️  Press Ctrl+C to stop monitoring")
    
    try:
        subprocess.run([sys.executable, "src/streaming/streaming.py"], cwd="d:/fraud_detection")
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")

def test_alerts():
    """Test the alert system"""
    print("\n📧 Testing alert system...")
    try:
        subprocess.run([sys.executable, "src/alerting/alert_manager.py"], cwd="d:/fraud_detection")
        print("✅ Alert test completed!")
    except Exception as e:
        print(f"❌ Alert test failed: {e}")

def show_status():
    """Show system status"""
    print("\n📊 System Status:")
    print("=" * 50)
    
    # Check database
    db_path = "d:/fraud_detection/src/etl/fraud_detection_db.sqlite"
    if os.path.exists(db_path):
        size = os.path.getsize(db_path) / (1024 * 1024)  # MB
        print(f"💾 Database: ✅ Connected ({size:.1f} MB)")
    else:
        print("💾 Database: ❌ Not found")
    
    # Check models
    model_dir = "d:/fraud_detection/models/saved"
    if os.path.exists(model_dir):
        models = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
        print(f"🤖 ML Models: ✅ {len(models)} models loaded")
    else:
        print("🤖 ML Models: ❌ No models found")
    
    # Check data
    data_file = "d:/fraud_detection/data/creditcard.csv"
    if os.path.exists(data_file):
        size = os.path.getsize(data_file) / (1024 * 1024)  # MB
        print(f"📊 Data File: ✅ Available ({size:.1f} MB)")
    else:
        print("📊 Data File: ❌ Missing")
    
    print("=" * 50)

def show_menu():
    """Display the main menu"""
    print("\n" + "=" * 60)
    print("🎛️  FRAUD DETECTION SYSTEM - MAIN MENU")
    print("=" * 60)
    print("1. 🚀 Setup Project (First-time setup)")
    print("2. 🤖 Train ML Models")
    print("3. 📊 Launch Dashboard")
    print("4. 🔍 Start Real-time Monitoring")
    print("5. 📧 Test Alert System")
    print("6. 📈 Show System Status")
    print("7. 🆘 Help & Documentation")
    print("8. 🚪 Exit")
    print("=" * 60)

def show_help():
    """Show help and documentation"""
    print("\n📚 Help & Documentation:")
    print("=" * 50)
    print("🚀 First-time Setup:")
    print("   1. Download Kaggle dataset to data/creditcard.csv")
    print("   2. Run option 1 (Setup Project)")
    print("   3. Run option 2 (Train ML Models)")
    print("   4. Launch dashboard with option 3")
    print()
    print("🔍 Usage:")
    print("   - Dashboard: View fraud analytics and system status")
    print("   - Monitoring: Real-time fraud detection on new transactions")
    print("   - Alerts: Email notifications for suspicious activity")
    print()
    print("🛠️  Troubleshooting:")
    print("   - Check system status (option 6) for issues")
    print("   - Ensure all files are in correct locations")
    print("   - Verify email credentials for alerts")
    print()
    print("📧 Email Setup:")
    print("   - Gmail: Requires App Password with 2FA enabled")
    print("   - Update credentials in src/alerting/alert.py")
    print("=" * 50)

def main():
    """Main menu loop"""
    print_banner()
    
    while True:
        show_menu()
        
        try:
            choice = input("\n🎯 Enter your choice (1-8): ").strip()
            
            if choice == '1':
                if setup_project():
                    print("✅ Project setup completed successfully!")
                else:
                    print("❌ Project setup failed. Please check the errors above.")
            
            elif choice == '2':
                if train_models():
                    print("✅ Model training completed successfully!")
                else:
                    print("❌ Model training failed. Please check the errors above.")
            
            elif choice == '3':
                start_dashboard()
            
            elif choice == '4':
                start_monitoring()
            
            elif choice == '5':
                test_alerts()
            
            elif choice == '6':
                show_status()
            
            elif choice == '7':
                show_help()
            
            elif choice == '8':
                print("\n👋 Thank you for using the Financial Fraud Detection System!")
                print("🔒 Stay secure and keep fighting fraud!")
                break
            
            else:
                print("❌ Invalid choice. Please enter a number between 1-8.")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")
        
        # Pause before showing menu again
        input("\n⏸️  Press Enter to continue...")

if __name__ == "__main__":
    main()