import smtplib
import os

# SERVER = "localhost"

FROM = 'xx@126.com'

TO = ["zxx@126.com"]  # must be a list

SUBJECT = "Hello!"

TEXT = "Training loss is 2.567, This message was sent with Python's smtplib."

# Prepare actual message

message = """\
From: %s
To: %s
Subject: %s

%s
""" % (FROM, ", ".join(TO), SUBJECT, TEXT)

# Send the mail

auth_code = os.environ["netease_auth_code"]

server = smtplib.SMTP('smtp.126.com')
server.login("xx@126.com", auth_code)

server.sendmail(FROM, TO, message)
server.quit()
