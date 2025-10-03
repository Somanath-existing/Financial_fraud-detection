# 🔍 Financial Fraud Detection System

A comprehensive machine learning-powered fraud detection system with real-time monitoring, intelligent alerting, and interactive dashboard.

## 🚀 Features

### 🧠 Advanced ML Models
- **Supervised Learning**: Logistic Regression, Random Forest, XGBoost
- **Unsupervised Learning**: Isolation Forest, One-Class SVM
- **Feature Engineering**: PCA components, temporal features, anomaly scores
- **Model Persistence**: Save/load trained models for production use

### ⚡ Real-Time Monitoring
- **Streaming Processing**: File-based transaction monitoring
- **Live Predictions**: Real-time fraud scoring for incoming transactions
- **Adaptive Thresholds**: Dynamic risk scoring based on transaction patterns
- **Performance Metrics**: Model accuracy tracking and evaluation

### 📧 Intelligent Alerting
- **Multi-Level Alerts**: High, Medium, Low priority based on fraud probability
- **Email Notifications**: Automated alerts via Gmail/Outlook SMTP
- **Alert Management**: Centralized alert history and status tracking
- **Daily Summaries**: Automated reporting of fraud detection statistics

### 📊 Interactive Dashboard
- **Real-Time Visualization**: Live transaction monitoring and fraud detection
- **Advanced Analytics**: Heatmaps, scatter plots, distribution charts
- **Model Management**: Switch between different ML models on-the-fly
- **System Monitoring**: Health checks and performance metrics

## 🏗️ Architecture

```
fraud_detection/
├── src/
│   ├── etl/                    # Data processing
│   │   ├── etl.py             # Extract, Transform, Load pipeline
│   │   └── fraud_detection_db.sqlite
│   ├── models/                 # Machine Learning
│   │   ├── fraud_models.py    # ML model training and prediction
│   │   └── saved/             # Trained model storage
│   ├── streaming/              # Real-time processing
│   │   ├── streaming.py       # Real-time monitoring system
│   │   └── processed/         # Processed transaction files
│   └── alerting/              # Alert management
│       ├── alert.py           # Basic email alerting
│       └── alert_manager.py   # Advanced alert management
├── dashboard/                  # Web interface
│   └── dashboard.py           # Streamlit dashboard
├── data/                      # Data storage
│   └── creditcard.csv         # Kaggle fraud dataset
├── streaming_input/           # Real-time data input
└── README.md
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Git
- Internet connection for package installation

### Setup Instructions

1. **Clone or navigate to the project directory**
```bash
cd d:/fraud_detection
```

2. **Install required packages**
```bash
pip install pandas scikit-learn numpy joblib xgboost lightgbm streamlit plotly altair pyspark
```

3. **Download the Kaggle Credit Card Fraud Dataset**
   - Visit: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
   - Download `creditcard.csv`
   - Place it in `d:/fraud_detection/data/creditcard.csv`

4. **Run the ETL process**
```bash
python src/etl/etl.py
```

5. **Train ML models**
```bash
python src/models/fraud_models.py
```

## 🚀 Usage

### 1. Launch the Dashboard
```bash
streamlit run dashboard/dashboard.py
```
- Open your browser to `http://localhost:8501`
- Login credentials: `admin` / `admin123`

### 2. Start Real-Time Monitoring
```bash
python src/streaming/streaming.py
```
- Creates sample transactions for testing
- Monitors `streaming_input/` directory for new CSV files
- Sends alerts for suspicious transactions

### 3. Test Alert System
```bash
python src/alerting/alert_manager.py
```

## 📊 Dashboard Features

### 🔐 Authentication
- Secure login system
- Session management
- User access control

### 📈 Overview Tab
- **Key Metrics**: Total transactions, fraud detected, alerts sent
- **Fraud by Hour**: Temporal fraud pattern analysis
- **Amount Distribution**: Transaction value analysis
- **Risk Heatmap**: Interactive fraud risk visualization

### 🔍 Real-Time Monitoring Tab
- **Live Transaction Feed**: Real-time transaction processing
- **Risk Scoring**: Dynamic fraud probability calculation
- **Volume Tracking**: Transaction throughput monitoring
- **Alert Triggers**: Configurable threshold-based alerts

### 🤖 ML Predictions Tab
- **Model Performance**: Accuracy metrics and statistics
- **Prediction History**: Historical fraud predictions
- **Probability Analysis**: Fraud likelihood distributions
- **Model Comparison**: Performance across different algorithms

### 📧 Alerts Tab
- **Alert Management**: View and manage fraud alerts
- **Delivery Status**: Track email delivery success/failure
- **Alert History**: Complete audit trail
- **Test Functions**: Validate alert system functionality

### ⚙️ System Tab
- **Health Monitoring**: System component status
- **Model Management**: Train and deploy new models
- **Cache Management**: Performance optimization
- **Future Roadmap**: Planned enhancements

## 🔧 Configuration

### Email Alerts
Update `src/alerting/alert.py` with your credentials:
```python
sender_email = "your_email@gmail.com"
sender_password = "your_app_password"  # Gmail App Password
recipient_email = "recipient@gmail.com"
```

### Model Parameters
Modify `src/models/fraud_models.py` for custom model settings:
```python
models_to_train = {
    'logistic_regression': LogisticRegression(max_iter=1000),
    'random_forest': RandomForestClassifier(n_estimators=100),
    'xgboost': XGBClassifier(n_estimators=100)
}
```

### Database Settings
Configure database path in modules:
```python
db_path = "d:/fraud_detection/src/etl/fraud_detection_db.sqlite"
```

## 📊 Data Schema

### Transactions Table
```sql
CREATE TABLE transactions (
    transaction_id INTEGER,
    customer_id TEXT,
    amount REAL,
    timestamp TEXT,
    location TEXT,
    device TEXT,
    is_fraud INTEGER,
    amount_norm REAL,
    hour INTEGER,
    is_anomaly INTEGER,
    V1-V28 REAL  -- PCA components from Kaggle dataset
);
```

### Predictions Table
```sql
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY,
    transaction_id TEXT,
    amount REAL,
    predicted_fraud INTEGER,
    fraud_probability REAL,
    model_used TEXT,
    processed_at TEXT
);
```

### Alerts Table
```sql
CREATE TABLE alerts (
    id INTEGER PRIMARY KEY,
    transaction_id TEXT,
    alert_type TEXT,
    severity TEXT,
    message TEXT,
    recipient_email TEXT,
    sent_at TEXT,
    status TEXT
);
```

## 🤖 Machine Learning Models

### Supervised Models
- **Logistic Regression**: Linear classification with probability scores
- **Random Forest**: Ensemble method with feature importance
- **XGBoost**: Gradient boosting for high performance

### Unsupervised Models
- **Isolation Forest**: Anomaly detection for outlier identification
- **One-Class SVM**: Support vector machine for novelty detection

### Feature Engineering
- **Temporal Features**: Hour of day, day of week patterns
- **Amount Normalization**: Min-max scaling for transaction amounts
- **Anomaly Scores**: Statistical outlier detection
- **PCA Components**: Dimensionality reduction features (V1-V28)

## 📧 Alert System

### Alert Types
- **CRITICAL_FRAUD**: Probability > 80%
- **SUSPECTED_FRAUD**: Probability > 50%
- **ANOMALY_DETECTED**: Unusual patterns detected
- **SYSTEM_ALERT**: System status notifications
- **DAILY_SUMMARY**: Automated daily reports

### Email Configuration
Supports multiple email providers:
- **Gmail**: Requires App Password with 2FA
- **Outlook**: Standard SMTP authentication
- **Custom SMTP**: Configure any email provider

## 🔄 Real-Time Processing

### Streaming Architecture
1. **File Monitoring**: Watches `streaming_input/` directory
2. **Data Processing**: Loads and preprocesses new transactions
3. **Model Prediction**: Applies trained ML models
4. **Alert Triggering**: Sends notifications for high-risk transactions
5. **Data Storage**: Saves predictions to database

### Sample Transaction Format
```csv
transaction_id,amount,timestamp,location,device,customer_id,V1,V2,...,V28,is_fraud
TXN_000001,1500.50,2025-01-01T10:30:00,NY,Mobile,CUST_1001,0.5,-1.2,...,0.8,0
```

## 🚀 Future Enhancements

### 🧠 Advanced AI Features
- **Deep Learning**: Neural networks for complex pattern recognition
- **Graph Analytics**: Network analysis for fraud ring detection
- **NLP Processing**: Text analysis for transaction descriptions
- **Reinforcement Learning**: Adaptive fraud detection strategies

### 🔗 Integration Capabilities
- **API Development**: RESTful APIs for third-party integration
- **Blockchain**: Immutable fraud prevention ledger
- **Apache Kafka**: True real-time streaming at scale
- **Cloud Deployment**: AWS/Azure/GCP cloud architecture

### 📱 User Experience
- **Mobile App**: iOS/Android fraud monitoring application
- **Voice Alerts**: Phone call notifications for critical fraud
- **Dashboard Customization**: Personalized user interfaces
- **Multi-language Support**: Internationalization features

## 🐛 Troubleshooting

### Common Issues

1. **Database Connection Error**
   - Ensure SQLite file exists: `d:/fraud_detection/src/etl/fraud_detection_db.sqlite`
   - Run ETL process: `python src/etl/etl.py`

2. **Import Errors**
   - Install missing packages: `pip install package_name`
   - Check Python path configuration

3. **Email Alerts Not Working**
   - Verify Gmail App Password setup
   - Check SMTP server settings
   - Test network connectivity

4. **Dashboard Loading Issues**
   - Clear Streamlit cache: Click "Clear Cache" in dashboard
   - Restart Streamlit: `Ctrl+C` and rerun command
   - Check port availability: Default is 8501

5. **Model Training Failures**
   - Ensure sufficient data in database
   - Check memory availability for large datasets
   - Verify scikit-learn installation

### Performance Optimization

1. **Database Optimization**
   - Create indexes on frequently queried columns
   - Use LIMIT clauses for large datasets
   - Regular database maintenance

2. **Model Performance**
   - Use feature selection to reduce dimensionality
   - Implement model caching for repeated predictions
   - Consider model ensemble techniques

3. **Dashboard Performance**
   - Enable Streamlit caching with appropriate TTL
   - Limit data visualization to recent records
   - Use asynchronous data loading where possible

## 📞 Support

For technical support and questions:
- Review the troubleshooting section above
- Check the GitHub issues (if applicable)
- Ensure all dependencies are properly installed
- Verify data files are in correct locations

## 📄 License

This project is for educational and demonstration purposes. Please ensure compliance with data privacy regulations when using with real financial data.

## 🙏 Acknowledgments

- **Kaggle**: Credit Card Fraud Detection Dataset
- **Scikit-learn**: Machine learning library
- **Streamlit**: Dashboard framework
- **Plotly**: Interactive visualizations
- **PySpark**: Big data processing

---

**Built with ❤️ for financial security and fraud prevention**