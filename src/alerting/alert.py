"""
Alerting system for high-risk transactions
- Sends email notifications
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_alert(transaction_details, recipient_email, sender_email, sender_password):
    subject = "Fraud Alert: Suspicious Transaction Detected"
    body = f"🚨 Alert! Suspicious transaction detected:\n\n{transaction_details}"

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = recipient_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    try:
        # --- First try TLS (port 587) ---
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_email, message.as_string())
        server.quit()
        print("✅ Alert email sent successfully via TLS (587).")

    except Exception as e1:
        print(f"⚠️ TLS failed: {e1}")
        try:
            # --- If TLS fails, try SSL (port 465) ---
            server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, message.as_string())
            server.quit()
            print("✅ Alert email sent successfully via SSL (465).")
        except Exception as e2:
            print(f"❌ Both TLS and SSL failed. Error: {e2}")

# Example usage
if __name__ == "__main__":
    transaction_details = "Transaction ID: 12345, Amount: $5000, Location: NY, Device: Mobile"
    sender_email = "somanathks711@gmail.com"
    sender_password = "keroypktdekqbone"  # 16-char App Password (no spaces)
    recipient_email = "somanathks711@gmail.com"

    send_alert(transaction_details, recipient_email, sender_email, sender_password)
