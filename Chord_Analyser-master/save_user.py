import firebase_admin 
from firebase_admin import credentials
from firebase_admin import firestore, App

def save_data(userID,name,email,password,location,level):
    cred = credentials.Certificate("ServiceAccountKey.json")
    firebase_admin.initialize_app(cred)
    db_firestore = firestore.client()
    try:
        user_data = db_firestore.collection('Users').document(userID)
        user_data.set({
            'name': name,
            'email': email,
            'password':password,
            'location':location,
            'level of playing guitar': level
            })
        return 1
    except:
        return 0
    
    