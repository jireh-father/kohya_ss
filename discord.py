import os
import requests


def send_message_to_discord(message):
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        return
    data = {"content": message}
    requests.post(webhook_url, json=data)
