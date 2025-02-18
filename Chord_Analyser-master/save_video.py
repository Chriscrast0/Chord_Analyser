import pyrebase

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

#saving the video in the database and returning the url of the video
def save_video(file, name, genre,user_id):
    firebase = pyrebase.initialize_app(config)
    storage = firebase.storage()
    db = firebase.database()
    storage.child("Videos/"+file.filename).put(file)
    print("file saved in database successfully")
     # start the upload task in the background
    download_url = storage.child("Videos/"+file.filename).get_url(None)
    print(download_url)
    # store the download URL in Realtime Database
    data={"Name": name,"Genre":genre,"Url":download_url, 'user_id': user_id}
    db.child('Videos_Data').push(data)
    print("successfully entered the video details in realtime database")
    if download_url!=0:
        return download_url
    else:
        return 0
    
