from lib2to3.pgen2 import token
from google.oauth2 import id_token
import pyrebase

config = {
    'apiKey': "",
    'authDomain': "",
    'projectId': "",
    'storageBucket': "",
    'messagingSenderId': "",
    'appId': "",
    'measurementId': "",
    'databaseURL': ''
}

def login_user(email,password):
    firebase = pyrebase.initialize_app(config)
    auth = firebase.auth()
    try:
        user = auth.sign_in_with_email_and_password(email,password)
        id_token = user['localId']
        return id_token
    except:
        return 0

def google_login(id):
    firebase = pyrebase.initialize_app(config)
    auth = firebase.auth()
    try:
        user = auth.sign_in_with_custom_token(id)
        id_token = user['localId']
        return id_token
    except:
        return 0
