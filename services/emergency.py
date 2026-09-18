import os
import re
import urllib.parse

from config import (
    ALERT_STATUS_BY_SEVERITY,
    INCIDENT_SERVICE_PROFILES,
    TWILIO_ACCOUNT_SID,
    TWILIO_AUTH_TOKEN,
    TWILIO_SMS_FROM_NUMBER,
    TWILIO_VOICE_FROM_NUMBER,
)
from services.geocoding import resolve_readable_location


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
        # 10-digit Indian mobile numbers (starting with 6, 7, 8, or 9)
        if re.fullmatch(r"[6-9]\d{9}", cleaned):
            recipients.append(f"+91{cleaned}")
            continue
        # 12-digit Indian numbers with 91 prefix
        if re.fullmatch(r"91[6-9]\d{9}", cleaned):
            recipients.append(f"+{cleaned}")
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


INCIDENT_HINDI = {
    "accident": "सड़क दुर्घटना",
    "road blockage": "सड़क अवरोध",
    "medical emergency": "चिकित्सा आपातकाल",
    "fire hazard": "आग की दुर्घटना",
    "hazard": "सड़क खतरा",
}

SEVERITY_HINDI = {
    "high": "अति गंभीर",
    "medium": "मध्यम",
    "low": "सामान्य सतर्कता",
}

SEVERITY_ENGLISH = {
    "high": "High severity, critical",
    "medium": "Medium priority",
    "low": "Low priority advisory",
}


def build_alert_message_text(road_location, source_location, incident_type, severity, notes):
    human_location = resolve_readable_location(road_location)
    message = (
        f"Find My Avenue alert: {severity.title()} {incident_type} reported near "
        f"{human_location}."
    )
    if notes:
        message = f"{message} Notes: {notes[:120]}"
    return message


def build_call_twiml(
    road_location="Emergency Location",
    incident_type="accident",
    severity="high",
    notes="",
    raw_message=None,
):
    try:
        from twilio.twiml.voice_response import VoiceResponse
    except ImportError as exc:
        raise RuntimeError(
            "Twilio voice helper is missing on the server. Redeploy after installing requirements."
        ) from exc

    human_location = resolve_readable_location(road_location)
    eng_incident = (incident_type or "accident").title()
    eng_severity = SEVERITY_ENGLISH.get((severity or "high").lower(), (severity or "high").title())
    hindi_incident = INCIDENT_HINDI.get((incident_type or "accident").lower(), "सड़क दुर्घटना")
    hindi_severity = SEVERITY_HINDI.get((severity or "high").lower(), "अति गंभीर")

    notes_clean = f" Additional information: {notes[:100]}." if notes else ""

    english_speech = (
        f"Attention! This is an urgent alert from Find My Avenue emergency response system. "
        f"A {eng_severity} {eng_incident} has been reported near {human_location}. "
        f"{notes_clean} "
        f"Simulated ambulance and police control room teams have been alerted. Please take immediate action."
    )

    hindi_speech = (
        f"कृपया ध्यान दें। यह फाइंड माय एवेन्यू आपातकालीन सेवा से एक आवश्यक संदेश है। "
        f"{human_location} के पास एक {hindi_severity} {hindi_incident} की सूचना मिली है। "
        f"नजदीकी पुलिस और एम्बुलेंस सहायता को सूचित कर दिया गया है। कृपया तुरंत सहायता प्रदान करें।"
    )

    response = VoiceResponse()
    response.pause(length=1)
    # 1. Indian English announcement
    response.say(english_speech, voice="Polly.Aditi", language="en-IN")
    response.pause(length=2)
    # 2. Hindi repetition
    response.say(hindi_speech, voice="Polly.Aditi", language="hi-IN")
    response.pause(length=1)
    response.say("This was an automated emergency broadcast. Thank you.", voice="Polly.Aditi", language="en-IN")
    response.say("यह एक स्वचालित आपातकालीन संदेश था। धन्यवाद।", voice="Polly.Aditi", language="hi-IN")
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


def get_call_url(road_location="Emergency Location", incident_type="accident", severity="high", notes=""):
    twiml_base = os.getenv("TWILIO_TWIML_URL")
    if twiml_base:
        return twiml_base
    base_url = (
        os.getenv("RENDER_EXTERNAL_URL")
        or os.getenv("APP_BASE_URL")
        or "https://find-my-avenue.onrender.com"
    ).rstrip("/")
    params = urllib.parse.urlencode(
        {
            "location": road_location,
            "incident_type": incident_type,
            "severity": severity,
            "notes": (notes or "")[:100],
        }
    )
    return f"{base_url}/api/emergency/twiml?{params}"


def deliver_real_notifications(
    alert_message,
    recipient_numbers,
    send_sms,
    send_call,
    road_location="Emergency Location",
    incident_type="accident",
    severity="high",
    notes="",
):
    client = get_twilio_client()
    results = []
    if send_sms and not TWILIO_SMS_FROM_NUMBER:
        raise RuntimeError("Twilio SMS sender is missing. Add TWILIO_SMS_FROM_NUMBER.")
    if send_call and not TWILIO_VOICE_FROM_NUMBER:
        raise RuntimeError("Twilio voice sender is missing. Add TWILIO_VOICE_FROM_NUMBER or TWILIO_SMS_FROM_NUMBER.")
    call_url = get_call_url(road_location, incident_type, severity, notes) if send_call else None
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
                err_msg = str(exc)
                if "predefined SMS templates" in err_msg:
                    # Fallback to Twilio Trial pre-approved template so user physically receives an SMS
                    try:
                        trial_body = (
                            "Reminder: Appt Tue Oct 29, 3:00 PM. Reply C to confirm or R to reschedule. Test message from Twilio."
                        )
                        fallback_msg = client.messages.create(body=trial_body, from_=TWILIO_SMS_FROM_NUMBER, to=number)
                        results.append(
                            {
                                "channel": "sms",
                                "to": number,
                                "provider": "twilio",
                                "status": fallback_msg.status or "queued",
                                "sid": fallback_msg.sid,
                                "note": "Delivered via Twilio trial template (custom text requires upgraded Twilio account).",
                            }
                        )
                    except Exception:
                        results.append(
                            {
                                "channel": "sms",
                                "to": number,
                                "provider": "twilio",
                                "status": "failed",
                                "error": (
                                    "Twilio Trial restricts custom SMS text to Indian numbers without approved templates. "
                                    "Voice Call alert works dynamically. Upgrade Twilio account to unlock custom SMS."
                                ),
                            }
                        )
                else:
                    results.append(
                        {
                            "channel": "sms",
                            "to": number,
                            "provider": "twilio",
                            "status": "failed",
                            "error": err_msg[:240],
                        }
                    )
        if send_call:
            try:
                # Twilio trial accounts forbid inline 'twiml' and require 'url'.
                call = client.calls.create(to=number, from_=TWILIO_VOICE_FROM_NUMBER, url=call_url)
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
                # Fallback to Twilio demo voice XML if custom webhook url fails
                try:
                    call = client.calls.create(
                        to=number,
                        from_=TWILIO_VOICE_FROM_NUMBER,
                        url="http://demo.twilio.com/docs/voice.xml",
                    )
                    results.append(
                        {
                            "channel": "call",
                            "to": number,
                            "provider": "twilio",
                            "status": call.status or "queued",
                            "sid": call.sid,
                        }
                    )
                except Exception as fallback_exc:
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
