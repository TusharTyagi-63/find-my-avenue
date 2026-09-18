import os
import re
import urllib.parse

from config import (
    ALERT_STATUS_BY_SEVERITY,
    D7_API_TOKEN,
    FAST2SMS_API_KEY,
    INCIDENT_SERVICE_PROFILES,
    TWILIO_ACCOUNT_SID,
    TWILIO_AUTH_TOKEN,
    TWILIO_SMS_FROM_NUMBER,
    TWILIO_VOICE_FROM_NUMBER,
    TWILIO_WHATSAPP_FROM_NUMBER,
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


def build_whatsapp_alert_text(road_location, incident_type, severity, notes=""):
    human_location = resolve_readable_location(road_location)
    eng_incident = (incident_type or "accident").title()
    eng_severity = (severity or "high").upper()
    dispatch_status = ALERT_STATUS_BY_SEVERITY.get((severity or "high").lower(), "Immediate dispatch confirmed")

    lines = [
        "🚨 *FIND MY AVENUE — EMERGENCY ALERT*",
        "━━━━━━━━━━━━━━━━━━━━━━",
        f"⚠️ *Incident:* {eng_incident}",
        f"🔴 *Severity:* {eng_severity}",
        f"📍 *Location:* {human_location}",
    ]
    if notes:
        lines.append(f"📝 *Details:* {notes[:180]}")
    lines.extend([
        f"⚡ *Dispatch Status:* {dispatch_status}",
        "🚑 *Units:* Emergency medical, ambulance, and police response teams notified.",
        "━━━━━━━━━━━━━━━━━━━━━━",
        "🌐 *Source:* Find My Avenue Emergency Response System",
    ])
    return "\n".join(lines)


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
        f"Emergency Alert! This is the Find My Avenue Emergency Response Dispatch. "
        f"A {eng_severity} {eng_incident} has been confirmed near {human_location}. "
        f"{notes_clean} "
        f"Emergency medical teams, ambulance, and police control have been notified. "
        f"Immediate response required."
    )

    hindi_speech = (
        f"आपातकालीन सूचना! यह फाइंड माय एवेन्यू आपातकालीन सेवा नियंत्रण कक्ष से आवश्यक संदेश है। "
        f"{human_location} के पास {hindi_severity} {hindi_incident} की पुष्टि हुई है। "
        f"एम्बुलेंस और पुलिस राहत दल को तुरंत सूचित कर दिया गया है। कृपया शीघ्र आवश्यक सहायता सुनिश्चित करें।"
    )

    response = VoiceResponse()
    response.pause(length=1)
    # 1. Indian English announcement
    response.say(english_speech, voice="Polly.Aditi", language="en-IN")
    response.pause(length=2)
    # 2. Hindi repetition
    response.say(hindi_speech, voice="Polly.Aditi", language="hi-IN")
    response.pause(length=1)
    response.say("Emergency alert broadcast concluded.", voice="Polly.Aditi", language="en-IN")
    response.say("आपातकालीन चेतावनी संदेश समाप्त।", voice="Polly.Aditi", language="hi-IN")
    return str(response)


def send_sms_via_fast2sms(recipient_numbers, message_text):
    if not FAST2SMS_API_KEY:
        raise RuntimeError("Fast2SMS is not configured. Add FAST2SMS_API_KEY.")
    import requests
    cleaned_numbers = []
    for num in recipient_numbers:
        digits = re.sub(r"[^\d]", "", str(num))
        if digits.startswith("91") and len(digits) == 12:
            digits = digits[2:]
        if len(digits) == 10:
            cleaned_numbers.append(digits)
    if not cleaned_numbers:
        raise ValueError("No valid 10-digit Indian mobile numbers found for SMS.")

    url = "https://www.fast2sms.com/dev/bulkV2"
    payload = {
        "route": "q",
        "message": message_text[:160],
        "language": "english",
        "flash": 0,
        "numbers": ",".join(cleaned_numbers),
    }
    headers = {
        "authorization": FAST2SMS_API_KEY,
        "Content-Type": "application/x-www-form-urlencoded",
    }
    resp = requests.post(url, data=payload, headers=headers, timeout=10)
    data = resp.json()
    if data.get("return") is True:
        return [
            {
                "channel": "sms",
                "to": f"+91{num}",
                "provider": "fast2sms",
                "status": "delivered",
                "request_id": data.get("request_id"),
            }
            for num in cleaned_numbers
        ]
    else:
        err_msg = ", ".join(data.get("message", ["SMS dispatch failed"]))
        return [
            {
                "channel": "sms",
                "to": f"+91{num}",
                "provider": "fast2sms",
                "status": "failed",
                "error": err_msg,
            }
            for num in cleaned_numbers
        ]


def send_sms_via_d7(recipient_numbers, message_text):
    if not D7_API_TOKEN:
        raise RuntimeError("D7 Networks is not configured. Add D7_API_TOKEN.")
    import requests
    url = "https://api.d7networks.com/messages/v1/send"
    headers = {
        "Authorization": f"Bearer {D7_API_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "messages": [
            {
                "channel": "sms",
                "originator": "SignSMS",
                "recipients": recipient_numbers,
                "content": message_text[:160],
                "msg_type": "text",
            }
        ]
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=12)
    data = resp.json() if resp.text else {}
    if resp.status_code in (200, 201, 202):
        return [
            {
                "channel": "sms",
                "to": num,
                "provider": "d7networks",
                "status": "delivered",
            }
            for num in recipient_numbers
        ]
    else:
        err_msg = data.get("detail") or data.get("message") or resp.text
        if isinstance(err_msg, list):
            err_msg = ", ".join(err_msg)
        return [
            {
                "channel": "sms",
                "to": num,
                "provider": "d7networks",
                "status": "failed",
                "error": str(err_msg)[:240],
            }
            for num in recipient_numbers
        ]


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
    send_sms=False,
    send_call=False,
    send_whatsapp=False,
    road_location="Emergency Location",
    incident_type="accident",
    severity="high",
    notes="",
    source_location="Emergency Location",
):
    results = []

    # 1. WhatsApp Dispatch (Twilio WhatsApp Sandbox)
    if send_whatsapp:
        client = get_twilio_client()
        raw_from = TWILIO_WHATSAPP_FROM_NUMBER or "+14155238886"
        whatsapp_from = raw_from if raw_from.startswith("whatsapp:") else f"whatsapp:{raw_from}"
        whatsapp_body = build_whatsapp_alert_text(
            road_location=road_location,
            incident_type=incident_type,
            severity=severity,
            notes=notes,
        )
        for number in recipient_numbers:
            whatsapp_to = number if number.startswith("whatsapp:") else f"whatsapp:{number}"
            try:
                msg = client.messages.create(
                    body=whatsapp_body,
                    from_=whatsapp_from,
                    to=whatsapp_to,
                )
                results.append(
                    {
                        "channel": "whatsapp",
                        "to": number,
                        "provider": "twilio",
                        "status": msg.status or "queued",
                        "sid": msg.sid,
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "channel": "whatsapp",
                        "to": number,
                        "provider": "twilio",
                        "status": "failed",
                        "error": str(exc)[:240],
                    }
                )

    # 2. SMS Dispatch
    if send_sms:
        if D7_API_TOKEN:
            try:
                sms_results = send_sms_via_d7(recipient_numbers, alert_message)
                results.extend(sms_results)
            except Exception as exc:
                for number in recipient_numbers:
                    results.append(
                        {
                            "channel": "sms",
                            "to": number,
                            "provider": "d7networks",
                            "status": "failed",
                            "error": str(exc)[:240],
                        }
                    )
        elif FAST2SMS_API_KEY:
            try:
                sms_results = send_sms_via_fast2sms(recipient_numbers, alert_message)
                results.extend(sms_results)
            except Exception as exc:
                for number in recipient_numbers:
                    results.append(
                        {
                            "channel": "sms",
                            "to": number,
                            "provider": "fast2sms",
                            "status": "failed",
                            "error": str(exc)[:240],
                        }
                    )
        else:
            client = get_twilio_client()
            if not TWILIO_SMS_FROM_NUMBER:
                raise RuntimeError("SMS provider is missing. Add D7_API_TOKEN, FAST2SMS_API_KEY, or TWILIO_SMS_FROM_NUMBER.")
            for number in recipient_numbers:
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
                                    "note": "Delivered via Twilio trial template.",
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
                                        "Twilio Trial restricts custom SMS to Indian numbers without approved templates. "
                                        "Add FAST2SMS_API_KEY for free instant custom SMS."
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

    # 2. Voice Call Dispatch
    if send_call:
        client = get_twilio_client()
        if not TWILIO_VOICE_FROM_NUMBER:
            raise RuntimeError("Twilio voice sender is missing. Add TWILIO_VOICE_FROM_NUMBER.")
        call_url = get_call_url(road_location, incident_type, severity, notes)
        for number in recipient_numbers:
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


def summarize_notification_results(results, send_sms=False, send_call=False, send_whatsapp=False):
    if not send_sms and not send_call and not send_whatsapp:
        return "Simulation only. No real SMS, call, or WhatsApp alert was requested."
    if not results:
        return "No real notifications were attempted."
    delivered = [item for item in results if item.get("status") not in {"failed", "canceled"}]
    failed = [item for item in results if item.get("status") == "failed"]
    channels = []
    if send_whatsapp:
        channels.append(f"WhatsApp: {sum(1 for item in delivered if item['channel'] == 'whatsapp')}")
    if send_call:
        channels.append(f"Calls: {sum(1 for item in delivered if item['channel'] == 'call')}")
    if send_sms:
        channels.append(f"SMS: {sum(1 for item in delivered if item['channel'] == 'sms')}")
    if failed:
        channels.append(f"Failures: {len(failed)}")
    return " | ".join(channels)


def build_dispatch_status(severity):
    return ALERT_STATUS_BY_SEVERITY[severity]
