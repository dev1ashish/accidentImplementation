from twilio.rest import Client
import os
from datetime import datetime

class TwilioHandler:
    def __init__(self):
        self.account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        self.auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        self.whatsapp_number = f"whatsapp:{os.getenv('TWILIO_WHATSAPP_NUMBER')}"
        self.recipient_whatsapp = f"whatsapp:{os.getenv('RECIPIENT_WHATSAPP_NUMBER')}"
        
        self.client = Client(self.account_sid, self.auth_token)

    def send_crash_alert(self, camera_id=None, city=None, district_no=None):
        whatsapp_message = (
            f"🚨 *Accident Detected!*\n"
        )

    try:
        message = self.client.messages.create(
            body=whatsapp_message,
            from_=self.whatsapp_number,
            to=self.recipient_whatsapp
        )
        print(f"WhatsApp alert sent successfully! SID: {message.sid}")
    except Exception as e:
        print(f"Failed to send WhatsApp alert: {str(e)}")

