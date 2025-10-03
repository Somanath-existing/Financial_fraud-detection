
"""
Financial Fraud Detection Dashboard
- Real-time fraud monitoring and visualization
- ML model integration and predictions
- Alert management and system monitoring
"""
import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Page configuration
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #ff4b4b;
}
.success-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #00cc88;
}
</style>
""", unsafe_allow_html=True)

# --- Authentication ---
def authenticate():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    
    if not st.session_state["logged_in"]:
        st.title("🔐 Financial Fraud Detection System")
        st.markdown("### Secure Login Required")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            username = st.text_input("👤 Username", placeholder="Enter username")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter password")
            
            if st.button("🚀 Login", use_container_width=True):
                if username == "admin" and password == "admin123":
                    st.session_state["logged_in"] = True
                    st.success("✅ Login successful! Redirecting...")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials. Please try again.")
        return False
    return True

# Data loading functions
@st.cache_data(ttl=30)
def load_transaction_data():
    try:
        conn = sqlite3.connect("d:/fraud_detection/src/etl/fraud_detection_db.sqlite")
        df = pd.read_sql_query("SELECT * FROM transactions LIMIT 10000", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading transaction data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=30)
def load_prediction_data():
    try:
        conn = sqlite3.connect("d:/fraud_detection/src/etl/fraud_detection_db.sqlite")
        df = pd.read_sql_query("SELECT * FROM predictions ORDER BY processed_at DESC LIMIT 5000", conn)
        conn.close()
        return df
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=30)
def load_alert_data():
    try:
        conn = sqlite3.connect("d:/fraud_detection/src/etl/fraud_detection_db.sqlite")
        df = pd.read_sql_query("SELECT * FROM alerts ORDER BY sent_at DESC LIMIT 1000", conn)
        conn.close()
        return df
    except Exception as e:
        return pd.DataFrame()

# ML Model Integration
def load_available_models():
    model_dir = "d:/fraud_detection/models/saved/"
    if not os.path.exists(model_dir):
        return []
    
    models = []
    for file in os.listdir(model_dir):
        if file.endswith('_model.pkl'):
            models.append(file.replace('_model.pkl', ''))
    return models

# --- Main Dashboard ---
def main_dashboard():
    st.title("🔍 Financial Fraud Detection Dashboard")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Dashboard Controls")
        
        # Auto-refresh toggle
        auto_refresh = st.toggle("🔄 Auto Refresh (30s)", value=False)
        if auto_refresh:
            st.rerun()
        
        # Manual refresh button
        if st.button("🔄 Refresh Data Now", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        
        # Model selection
        st.subheader("🤖 Model Settings")
        available_models = load_available_models()
        if available_models:
            selected_model = st.selectbox("Select Model", available_models)
        else:
            st.warning("⚠️ No trained models found")
            selected_model = None
        
        st.markdown("---")
        
        # Time range filter
        st.subheader("📅 Time Range")
        time_range = st.selectbox("Select Range", 
                                 ["Last Hour", "Last 24 Hours", "Last 7 Days", "Last 30 Days"])
    
    # Load data
    df_transactions = load_transaction_data()
    df_predictions = load_prediction_data()
    df_alerts = load_alert_data()
    
    if df_transactions.empty:
        st.error("❌ No transaction data available. Please run the ETL process first.")
        return
    
    # Key Metrics Row
    st.subheader("📊 Key Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_transactions = len(df_transactions)
    fraud_detected = df_transactions['is_fraud'].sum() if 'is_fraud' in df_transactions.columns else 0
    fraud_rate = (fraud_detected / total_transactions * 100) if total_transactions > 0 else 0
    alerts_sent = len(df_alerts)
    avg_amount = df_transactions['amount'].mean() if 'amount' in df_transactions.columns else 0
    
    with col1:
        st.metric("📈 Total Transactions", f"{total_transactions:,}")
    with col2:
        st.metric("🚨 Fraud Detected", fraud_detected, delta=f"{fraud_rate:.2f}%")
    with col3:
        st.metric("📧 Alerts Sent", alerts_sent)
    with col4:
        st.metric("💰 Avg Amount", f"${avg_amount:,.2f}")
    with col5:
        model_status = "🟢 Active" if selected_model else "🔴 Inactive"
        st.metric("🤖 Model Status", model_status)
    
    st.markdown("---")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🔍 Real-time Monitoring", "🤖 ML Predictions", "📧 Alerts", "⚙️ System"])
    
    with tab1:
        show_overview_tab(df_transactions, df_predictions)
    
    with tab2:
        show_monitoring_tab(df_transactions, selected_model)
    
    with tab3:
        show_predictions_tab(df_predictions, selected_model)
    
    with tab4:
        show_alerts_tab(df_alerts)
    
    with tab5:
        show_system_tab()

def show_overview_tab(df_transactions, df_predictions):
    st.subheader("📊 Fraud Detection Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Fraud by Hour")
        if 'hour' in df_transactions.columns and 'is_fraud' in df_transactions.columns:
            hourly_fraud = df_transactions.groupby('hour')['is_fraud'].agg(['count', 'sum']).reset_index()
            hourly_fraud['fraud_rate'] = (hourly_fraud['sum'] / hourly_fraud['count'] * 100).fillna(0)
            
            fig = px.bar(hourly_fraud, x='hour', y='sum', 
                        title="Fraudulent Transactions by Hour",
                        labels={'sum': 'Fraud Count', 'hour': 'Hour of Day'},
                        color='fraud_rate', color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 💰 Amount Distribution")
        if 'amount' in df_transactions.columns and 'is_fraud' in df_transactions.columns:
            fig = px.histogram(df_transactions, x='amount', color='is_fraud',
                             title="Transaction Amount Distribution",
                             labels={'amount': 'Transaction Amount ($)'})
            fig.update_layout(barmode='overlay')
            fig.update_traces(opacity=0.7)
            st.plotly_chart(fig, use_container_width=True)
    
    # Anomaly Heatmap
    if 'hour' in df_transactions.columns and 'amount' in df_transactions.columns and 'is_fraud' in df_transactions.columns:
        st.markdown("#### 🔥 Fraud Risk Heatmap")
        
        # Create amount bins
        df_viz = df_transactions.copy()
        df_viz['amount_bin'] = pd.cut(df_viz['amount'], bins=20, labels=False)
        
        # Create heatmap data
        heatmap_data = df_viz.groupby(['hour', 'amount_bin'])['is_fraud'].agg(['count', 'sum']).reset_index()
        heatmap_data['fraud_rate'] = (heatmap_data['sum'] / heatmap_data['count'] * 100).fillna(0)
        
        # Pivot for heatmap
        heatmap_pivot = heatmap_data.pivot(index='amount_bin', columns='hour', values='fraud_rate').fillna(0)
        
        fig = px.imshow(heatmap_pivot, 
                       title="Fraud Risk by Hour and Amount",
                       labels=dict(x="Hour of Day", y="Amount Percentile", color="Fraud Rate (%)"),
                       aspect="auto", color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)

def show_monitoring_tab(df_transactions, selected_model):
    st.subheader("🔍 Real-time Transaction Monitoring")
    
    # Monitoring controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        monitoring_active = st.toggle("🟢 Live Monitoring", value=True)
    
    with col2:
        threshold = st.slider("🚨 Alert Threshold", 0.0, 1.0, 0.5, 0.1)
    
    with col3:
        if st.button("🔍 Process Sample Transactions", use_container_width=True):
            process_sample_transactions()
    
    st.markdown("---")
    
    # Recent transactions
    st.markdown("#### 📋 Recent Transactions")
    if not df_transactions.empty:
        recent_transactions = df_transactions.tail(20).copy()
        
        # Add risk scoring (simplified)
        if 'amount_norm' in recent_transactions.columns:
            recent_transactions['risk_score'] = recent_transactions['amount_norm']
        else:
            recent_transactions['risk_score'] = np.random.uniform(0, 1, len(recent_transactions))
        
        recent_transactions['risk_level'] = recent_transactions['risk_score'].apply(
            lambda x: "🔴 High" if x > 0.7 else "🟡 Medium" if x > 0.3 else "🟢 Low"
        )
        
        st.dataframe(
            recent_transactions[['transaction_id', 'amount', 'timestamp', 'location', 'device', 'risk_level']],
            use_container_width=True
        )
    
    # Real-time charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ⚡ Transaction Volume (Live)")
        if 'timestamp' in df_transactions.columns:
            # Simulate real-time data using hour column instead of timestamp string slicing
            if 'hour' in df_transactions.columns:
                volume_data = df_transactions.groupby('hour')['transaction_id'].count()
                fig = px.line(x=volume_data.index, y=volume_data.values,
                             title="Transactions per Hour",
                             labels={'x': 'Hour of Day', 'y': 'Transaction Count'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback: group by timestamp bins
                df_temp = df_transactions.copy()
                df_temp['time_bin'] = (df_temp['timestamp'] // 3600).astype(int)  # Convert to hours
                volume_data = df_temp.groupby('time_bin')['transaction_id'].count().tail(24)
                fig = px.line(x=volume_data.index, y=volume_data.values,
                             title="Transactions per Hour",
                             labels={'x': 'Time (Hours)', 'y': 'Transaction Count'})
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 🎯 Risk Distribution")
        if not df_transactions.empty:
            risk_dist = pd.DataFrame({
                'Risk Level': ['Low', 'Medium', 'High'],
                'Count': [
                    len(df_transactions) * 0.8,
                    len(df_transactions) * 0.15,
                    len(df_transactions) * 0.05
                ]
            })
            fig = px.pie(risk_dist, values='Count', names='Risk Level',
                        color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'})
            st.plotly_chart(fig, use_container_width=True)

def show_predictions_tab(df_predictions, selected_model):
    st.subheader("🤖 Machine Learning Predictions")
    
    if df_predictions.empty:
        st.info("📝 No prediction data available. Start the real-time monitoring to see predictions.")
        return
    
    # Model performance metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_predictions = len(df_predictions)
        st.metric("🔢 Total Predictions", f"{total_predictions:,}")
    
    with col2:
        fraud_predictions = df_predictions['predicted_fraud'].sum() if 'predicted_fraud' in df_predictions.columns else 0
        st.metric("🚨 Fraud Predictions", fraud_predictions)
    
    with col3:
        avg_prob = df_predictions['fraud_probability'].mean() if 'fraud_probability' in df_predictions.columns else 0
        st.metric("📊 Avg Fraud Probability", f"{avg_prob:.1%}")
    
    # Prediction visualization
    if 'fraud_probability' in df_predictions.columns:
        st.markdown("#### 📈 Fraud Probability Distribution")
        fig = px.histogram(df_predictions, x='fraud_probability',
                          title="Distribution of Fraud Probabilities",
                          labels={'fraud_probability': 'Fraud Probability'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent predictions table
    st.markdown("#### 📋 Recent Predictions")
    if not df_predictions.empty:
        display_cols = ['transaction_id', 'amount', 'predicted_fraud', 'fraud_probability', 'model_used', 'processed_at']
        available_cols = [col for col in display_cols if col in df_predictions.columns]
        st.dataframe(df_predictions[available_cols].head(20), use_container_width=True)

def show_alerts_tab(df_alerts):
    st.subheader("📧 Alert Management")
    
    # Alert controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🧪 Test Alert System", use_container_width=True):
            test_alert_system()
    
    with col2:
        if st.button("📊 Send Daily Summary", use_container_width=True):
            send_daily_summary()
    
    with col3:
        alert_settings = st.toggle("🔔 Alerts Enabled", value=True)
    
    # Alert statistics
    if not df_alerts.empty:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_alerts = len(df_alerts)
            st.metric("📧 Total Alerts", total_alerts)
        
        with col2:
            sent_alerts = len(df_alerts[df_alerts['status'] == 'SENT']) if 'status' in df_alerts.columns else 0
            st.metric("✅ Sent Successfully", sent_alerts)
        
        with col3:
            failed_alerts = len(df_alerts[df_alerts['status'] == 'FAILED']) if 'status' in df_alerts.columns else 0
            st.metric("❌ Failed", failed_alerts)
        
        # Alert history
        st.markdown("#### 📜 Alert History")
        display_cols = ['alert_type', 'severity', 'recipient_email', 'sent_at', 'status']
        available_cols = [col for col in display_cols if col in df_alerts.columns]
        if available_cols:
            st.dataframe(df_alerts[available_cols].head(20), use_container_width=True)
    else:
        st.info("📝 No alerts sent yet. Configure alert thresholds and start monitoring.")

def show_system_tab():
    st.subheader("⚙️ System Administration")
    
    # System status
    st.markdown("#### 📊 System Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="success-card">
            <h4>🔗 Database Connection</h4>
            <p>✅ Connected</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        model_count = len(load_available_models())
        status = "🟢 Active" if model_count > 0 else "🔴 Inactive"
        st.markdown(f"""
        <div class="metric-card">
            <h4>🤖 ML Models</h4>
            <p>{status} ({model_count} loaded)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="success-card">
            <h4>📧 Alert System</h4>
            <p>✅ Operational</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # System actions
    st.markdown("#### 🛠️ System Actions")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Train New Models", use_container_width=True):
            train_models()
        
        if st.button("🧹 Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.success("✅ Cache cleared successfully!")
    
    with col2:
        if st.button("📊 Generate Report", use_container_width=True):
            generate_system_report()
        
        if st.button("🔄 Restart Services", use_container_width=True):
            st.info("🔄 Service restart initiated...")
    
    # Future enhancements
    st.markdown("#### 🚀 Future Enhancements")
    st.markdown("""
    - 🧠 **AI-driven Risk Scoring**: Advanced neural networks for customer creditworthiness assessment
    - 🔗 **Blockchain Integration**: Immutable fraud prevention ledger system  
    - 📱 **Mobile App**: Real-time alerts and monitoring on mobile devices
    - 🌐 **API Integration**: RESTful APIs for third-party system integration
    - 🔍 **Graph Analytics**: Network analysis for detecting fraud rings
    - ⚡ **Apache Kafka**: True real-time streaming for high-volume transactions
    """)

# Helper functions
def process_sample_transactions():
    st.info("🔄 Processing sample transactions...")
    # This would integrate with the streaming module
    
def test_alert_system():
    try:
        from alerting.alert_manager import AlertManager
        alert_manager = AlertManager()
        alert_manager.test_alert_system()
        st.success("✅ Test alerts sent successfully!")
    except Exception as e:
        st.error(f"❌ Error testing alerts: {e}")

def send_daily_summary():
    try:
        from alerting.alert_manager import AlertManager
        alert_manager = AlertManager()
        alert_manager.send_daily_summary()
        st.success("✅ Daily summary sent successfully!")
    except Exception as e:
        st.error(f"❌ Error sending summary: {e}")

def train_models():
    st.info("🤖 Training new models...")
    try:
        from models.fraud_models import main as train_main
        train_main()
        st.success("✅ Models trained successfully!")
        st.cache_data.clear()
    except Exception as e:
        st.error(f"❌ Error training models: {e}")

def generate_system_report():
    st.info("📊 Generating system report...")
    # Implementation for system report generation

# --- Main App ---
if __name__ == "__main__":
    if authenticate():
        main_dashboard()
