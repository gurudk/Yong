import smtplib
import os
from email.message import EmailMessage
from email.header import Header


def sendMail(SUBJECT, TEXT):
    msg = EmailMessage()
    msg.set_content(TEXT)
    msg['Subject'] = SUBJECT
    msg['From'] = "soccervit@126.com"
    msg['To'] = "soccervit@126.com"

    s = smtplib.SMTP('smtp.126.com')
    s.login('soccervit@126.com', os.environ["auth_code"])
    s.send_message(msg)
    s.quit()


content = "Epoch 2/1000 loss: 2.0951 duration:56.31"

sendMail(content, content)

# sendMail("soccervit@126.com", "soccervit@126.com", content, content, 'smtp.126.com')
