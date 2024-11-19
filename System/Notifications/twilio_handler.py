from twilio.rest import Client
import os
from datetime import datetime
import cv2
import numpy as np

class TwilioHandler:
    def __init__(self):
        self.account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        self.auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        self.from_number = os.getenv('TWILIO_PHONE_NUMBER')
        self.whatsapp_number = os.getenv('TWILIO_WHATSAPP_NUMBER')
        self.recipient_number = os.getenv('RECIPIENT_PHONE_NUMBER')
        self.recipient_whatsapp = os.getenv('RECIPIENT_WHATSAPP_NUMBER')
        
        if self.account_sid and self.auth_token:
            self.client = Client(self.account_sid, self.auth_token)
        else:
            print("Warning: Twilio credentials not found in environment variables")
            self.client = None

    def _save_temp_image(self, image_data):
        """Save numpy array image to temporary file"""
        if isinstance(image_data, np.ndarray):
            temp_path = "temp_crash.jpg"
            cv2.imwrite(temp_path, image_data)
            return temp_path
        return None

    def send_crash_alert(self, camera_id, city, district_no, crash_pic=None):
        if not self.client:
            print("Twilio client not initialized. Skipping notifications.")
            return False

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message_body = (
            f"🚨 CRASH ALERT!\n"
            f"Time:{timestamp}\n"
            f"Location: https://maps.app.goo.gl/9p7A22yZfDc6s4EV6 \n"
            f"Camera ID: 001"
        )

        # Handle crash_pic if it's a numpy array
        media_url = None
        if crash_pic is not None:
            temp_image_path = self._save_temp_image(crash_pic)
            if temp_image_path:
                # In production, you'd want to upload this to a cloud service
                # and use the URL. For testing, we'll skip the image.
                print(f"Image saved to {temp_image_path}")
                # media_url = "https://your-cloud-storage/temp_crash.jpg"

        sms_sent = False
        whatsapp_sent = False

        # Send SMS
        if self.from_number and self.recipient_number:
            try:
                message = self.client.messages.create(
                    body=message_body,
                    from_=self.from_number,
                    to=self.recipient_number
                )
                print(f"Crash alert SMS sent successfully! SID: {message.sid}")
                sms_sent = True
            except Exception as e:
                print(f"Failed to send crash alert SMS: {str(e)}")

        # Send WhatsApp
        if self.whatsapp_number and self.recipient_whatsapp:
            try:
                message = self.client.messages.create(
                    body=message_body,
                    from_=self.whatsapp_number,
                    to=self.recipient_whatsapp
                )
                print(f"Crash alert WhatsApp sent successfully! SID: {message.sid}")
                whatsapp_sent = True
            except Exception as e:
                print(f"Failed to send crash alert WhatsApp: {str(e)}")