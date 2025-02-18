import pyrebase

'''For Retrieving the chordhseets'''

def get_chordsheet():
    global config
    config = {
    'apiKey': "",
    'authDomain': "",
    'projectId': "",
    'storageBucket': "",
    'messagingSenderId': "1",
    'appId': "",
    'measurementId': "",
    'databaseURL': ""
    }
    firebase = pyrebase.initialize_app(config)
    db = firebase.database()
    items = db.child('ChordSheets').order_by_key().limit_to_last(5).get().val()
    item_homepage = list(reversed(list(items.values())))
    return item_homepage
    
def get_all_chordsheet():
    global config
    config = {
    'apiKey': "",
    'authDomain': "",
    'projectId': "",
    'storageBucket': "",
    'messagingSenderId': "",
    'appId': "",
    'measurementId': "",
    'databaseURL': ""
    }
    firebase = pyrebase.initialize_app(config)
    db = firebase.database()
    items = db.child('ChordSheets').order_by_key().get().val()
    item_homepage = list(reversed(list(items.values())))
    return item_homepage

def get_realtime():
    global config
    config = {
    'apiKey': "",
    'authDomain': "",
    'projectId': "e",
    'storageBucket': "",
    'messagingSenderId': "1",
    'appId': "",
    'measurementId': "",
    'databaseURL': ""
    }
    firebase = pyrebase.initialize_app(config)
    db = firebase.database()
    items = db.child('Realtime_Chordsheets').order_by_key().get().val()
    item_homepage = list(reversed(list(items.values())))
    return item_homepage

def practice_chordsheet(link):
    global config
    config = {
    'apiKey': "",
    'authDomain': "",
    'projectId': "",
    'storageBucket': "",
    'messagingSenderId': "",
    'appId': "",
    'measurementId': "",
    'databaseURL': ""
    }
    firebase = pyrebase.initialize_app(config)
    db = firebase.database()
    items = db.child('ChordSheets').order_by_key().limit_to_last(5).get().val()
    item_homepage = list(reversed(list(items.values())))
    return item_homepage
    
