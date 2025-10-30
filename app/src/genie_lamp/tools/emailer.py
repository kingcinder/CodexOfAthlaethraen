import os, ssl, smtplib
from email.message import EmailMessage
def send_email(smtp_host, smtp_port, user, pass_env, to, subject, body):
    pwd = os.getenv(pass_env, "")
    msg = EmailMessage(); msg["From"]=user; msg["To"]=to; msg["Subject"]=subject; msg.set_content(body)
    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_host, smtp_port, context=ctx) as s:
        s.login(user, pwd); s.send_message(msg)
    return {"ok": True}
