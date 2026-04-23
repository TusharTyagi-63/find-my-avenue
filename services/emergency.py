import re

from config import (
    ALERT_STATUS_BY_SEVERITY,
    INCIDENT_SERVICE_PROFILES,
    TWILIO_ACCOUNT_SID,
    TWILIO_AUTH_TOKEN,
    TWILIO_SMS_FROM_NUMBER,
    TWILIO_VOICE_FROM_NUMBER,
)


def parse_recipient_numbers(raw_value):
    if not raw_value:
        return []
    recipients = []
    for part in re.split(r"[,\n;]+", raw_value):
        cleaned = re.sub(r"[^\d+]", "", part.strip())
        if not cleaned:
            continue
        if cleaned.startswith("00"):
            cleaned = f"+{cleaned[2:]}"
        if cleaned.startswith("+") and re.fullmatch(r"\+[1-9]\d{7,14}", cleaned):
            recipients.append(cleaned)
            continue
        if re.fullmatch(r"[1-9]\d{7,14}", cleaned):
            recipients.append(f"+{cleaned}")
            continue
        raise ValueError(
            f"Use full mobile numbers with country code, like +919876543210. Invalid value: {part.strip()}"
        )
    return list(dict.fromkeys(recipients))


def get_twilio_client():
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        raise RuntimeError("Twilio is not configured. Add TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN.")
    try:
        from twilio.rest import Client
    except ImportError as exc:
        raise RuntimeError(
            "Twilio dependency is missing on the server. Redeploy after installing requirements."
        ) from exc
    return Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)


def build_alert_message_text(road_location, source_location, incident_type, severity, notes):
    message = (
        f"Find My Avenue alert: {severity.title()} {incident_type} reported near "
        f"{road_location}. Source: {source_location}."
    )
    if notes:
        message = f"{message} Notes: {notes[:120]}"
    return message


def build_call_twiml(alert_message):
    try:
        from twilio.twiml.voice_response import VoiceResponse
    except ImportError as exc:
        raise RuntimeError(
            "Twilio voice helper is missing on the server. Redeploy after installing requirements."
        ) from exc
    response = VoiceResponse()
    response.pause(length=1)
    response.say("This is a Find My Avenue emergency test alert.", voice="alice")
    response.say(alert_message, voice="alice")
    response.pause(length=1)
    response.say("This was a manual test call from the project prototype.", voice="alice")
    return str(response)


def build_emergency_services(incident_type, severity):
    services = []
    for index, service_name in enumerate(
        INCIDENT_SERVICE_PROFILES.get(incident_type, INCIDENT_SERVICE_PROFILES["accident"]),
        start=1,
    ):
        services.append(
            {
                "service": service_name,
                "channel": f"Simulated API channel #{index}",
                "status": "notified",
                "priority": severity,
            }
        )
    return services


def deliver_real_notifications(alert_message, recipient_numbers, send_sms, send_call):
    client = get_twilio_client()
    results = []
    if send_sms and not TWILIO_SMS_FROM_NUMBER:
        raise RuntimeError("Twilio SMS sender is missing. Add TWILIO_SMS_FROM_NUMBER.")
    if send_call and not TWILIO_VOICE_FROM_NUMBER:
        raise RuntimeError("Twilio voice sender is missing. Add TWILIO_VOICE_FROM_NUMBER or TWILIO_SMS_FROM_NUMBER.")
    call_twiml = build_call_twiml(alert_message) if send_call else None
    for number in recipient_numbers:
        if send_sms:
            try:
                message = client.messages.create(body=alert_message, from_=TWILIO_SMS_FROM_NUMBER, to=number)
                results.append(
                    {
                        "channel": "sms",
                        "to": number,
                        "provider": "twilio",
                        "status": message.status or "queued",
                        "sid": message.sid,
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "channel": "sms",
                        "to": number,
                        "provider": "twilio",
                        "status": "failed",
                        "error": str(exc)[:240],
                    }
                )
        if send_call:
            try:
                call = client.calls.create(to=number, from_=TWILIO_VOICE_FROM_NUMBER, twiml=call_twiml)
                results.append(
                    {
                        "channel": "call",
                        "to": number,
                        "provider": "twilio",
                        "status": call.status or "queued",
                        "sid": call.sid,
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "channel": "call",
                        "to": number,
                        "provider": "twilio",
                        "status": "failed",
                        "error": str(exc)[:240],
                    }
                )
    return results


def summarize_notification_results(results, send_sms, send_call):
    if not send_sms and not send_call:
        return "Simulation only. No real SMS or call was requested."
    if not results:
        return "No real notifications were attempted."
    delivered = [item for item in results if item.get("status") not in {"failed", "canceled"}]
    failed = [item for item in results if item.get("status") == "failed"]
    channels = []
    if send_sms:
        channels.append(f"SMS attempted: {sum(1 for item in delivered if item['channel'] == 'sms')}")
    if send_call:
        channels.append(f"Calls attempted: {sum(1 for item in delivered if item['channel'] == 'call')}")
    if failed:
        channels.append(f"Failures: {len(failed)}")
    return " | ".join(channels)


def build_dispatch_status(severity):
    return ALERT_STATUS_BY_SEVERITY[severity]
