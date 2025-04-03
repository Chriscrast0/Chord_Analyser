# Chord Analyzer - A revolutionary chord detection system
# Author: [Your Name]
# Date: [Current Date]
import os
import datetime
from flask import Flask, redirect, render_template, request, Response, session,url_for, jsonify
import firebase_admin 
from firebase_admin import credentials
from firebase_admin import firestore, auth
from login import *
from save_video import *
from video_to_chord import *
from realtime_camera import *
from test_chord import *
from chordsheet import *
import urllib.request
import requests
from google.oauth2 import id_token
from google_auth_oauthlib.flow import Flow
from pip._vendor import cachecontrol
import google.auth.transport.requests

#initializing the Flask app
app = Flask(__name__)

#initializing database of pyrebase
'''
config = {
    'apiKey': "put your apiKey",
    'authDomain': "put domain",
    'projectId': "put project id",
    'storageBucket': "put sorage bucket",
    'messagingSenderId': "put id",
    'appId': "put id",
    'measurementId': "put id",
    'databaseURL': ''
}

firebase = pyrebase.initialize_app(config)
auth = firebase.auth()
'''


''' Setting up the Firebase Database '''
cred = credentials.Certificate("ServiceAccountKey.json")
firebase_admin.initialize_app(cred, {
            'storageBucket': '',
            'databaseURL': ''
            })
db_firestore = firestore.client()


#initializing secret key for checking the session which will be used to track the user for registration
app.secret_key = 'lajibolala'
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"


''' Start of basic redirections or linking to page '''
#adding the paths to buttons
#main landing page
@app.route('/')
def index():
    chords = get_chordsheet()
    if('user' in session and 'id' in session):
        chords = get_chordsheet()
        return render_template('logout.html',t=chords)
    return render_template('index.html',t=chords)


#when clicked on the logo
@app.route('/home')
def home():
    return redirect('/')


#when clicked on get started on homepage
@app.route('/get_started')
def get_started():
    if('user' in session and 'id' in session):
        return render_template('get-started.html')
    else:
        return render_template('signin.html')


#when clicked on music theory on homepage
@app.route('/music_theory')
def music_theory():
    return render_template('music-theory.html')


#when clicked on contact us on anywhere
@app.route('/contact_us')
def contact_us():
    return render_template('contact-us.html')


#when clicked on faq page anywhere
@app.route('/faq')
def faq():
    return render_template('faq.html')


#when the user is already logged in
@app.route('/logout')
def logout():
    session.pop('user')
    session.pop('id')
    chords = get_chordsheet()
    return render_template('index.html',t=chords)


#for practicing chords
@app.route('/practice_chord')
def practice_chord():
    return render_template('practice_chords.html')


#for practice chordsheet
@app.route('/practice_chordsheet')
def practice_chordsheet():
    return render_template('practice_chordsheet.html')

''' End of Basic Redirections or linking to pages '''



#for listing all chordhseets saved
@app.route('/all_chordsheet')
def all_chordsheet():
    chords = get_chordsheet()
    realtime_chord = get_realtime()
    return render_template('chordsheet_list.html',t=chords, y=realtime_chord)



''' Start of Sign In '''
#when clicked on signin anywhere
@app.route('/signin',methods=['POST','GET'])
def signin():
    if('user' in session and 'id' in session):
        return render_template('logout.html')
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if email and len(password) != 0 and len(password) >= 6:
            try:
                user = login_user(email,password)
                print(user)
                if user!=0:
                    session['user'] = email
                    session['id'] = user
                    chords = get_chordsheet()
                    return render_template('logout.html',t=chords)
                else:
                    text = "Opps failed to login"
                    return render_template('error-msg.html',text=text)
            except:
                text = "OPPS!!! Failed to Login \n Try again later"
                return render_template('error-msg.html',text=text)
        else:
            text = "PLEASE ENTER CORRECT EMAIL AND PASSWORD WITH 8 OR MORE CHARACTERS"
            return render_template('error-msg.html',text=text)
    return render_template('signin.html')

''' End of Sign In '''    



'''Start of Sign In with Google'''

GOOGLE_CLIENT_ID = "your id"
flow = Flow.from_client_secrets_file(client_secrets_file='client_secret.json',
                                     scopes=["https://www.googleapis.com/auth/userinfo.profile", "https://www.googleapis.com/auth/userinfo.email", "openid"],
                                     redirect_uri="http://127.0.0.1:5000/callback"
                                     )


@app.route('/signin_google')
def signin_google():
    authorization_url, state = flow.authorization_url()
    session["state"] = state
    return redirect(authorization_url)

@app.route('/callback')
def callback():
    flow.fetch_token(authorization_response=request.url)

    if not session["state"] == request.args["state"]:
        text = "Oops!! Sorry couldn.t log you in"
        return render_template('error-msg.html',text=text)
        # State does not match!

    credentials = flow.credentials
    request_session = requests.session()
    cached_session = cachecontrol.CacheControl(request_session)
    token_request = google.auth.transport.requests.Request(session=cached_session)
    print(credentials._id_token)
    
    id_info = id_token.verify_oauth2_token(
        id_token=credentials._id_token,
        request=token_request,
        audience=GOOGLE_CLIENT_ID
    )

    session["google_id"] = id_info.get("sub")
    session["name"] = id_info.get("name")
    user_email = id_info.get('email')
    user_name = id_info.get('name')
    print(id_info)
    user_info = auth.create_user(email = user_email)
    if user_info:
        user_data = db_firestore.collection('Users').document(user_info.uid)
        user_data.set({
        'name': user_name,
        'email': user_email,
        'level':'beginner'
        })
        session['user'] = user_email
        session['id'] = user_info.uid
        chords = get_chordsheet()
        return render_template('logout.html',t=chords)
    else:
        text = "Oops There seems to be an issue!! Please try again later!!"
        return render_template('error-msg.html',text = text)

'''End of Sign In with Google'''



''' Start of Sign Up '''
#when clicked on sign up option on login page
@app.route('/register',methods=['POST', 'GET'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        location = request.form.get('location')
        level = request.form.get('level')
        print(name,email,password,location,level)
        if name and email and location and level and len(password)>=6:
            try:
                user = auth.create_user(email = email,
                                        email_verified = False,
                                        password =password,
                                        display_name = name,
                                        disabled=False)
                if user:
                    user_data = db_firestore.collection('Users').document(user.uid)
                    user_data.set({
                        'name': name,
                        'email': email,
                        'password':password,
                        'location':location,
                        'level':level
                    })
                    login = login_user(email,password)
                    if login!=0:
                        session['user'] = email
                        session['id'] = user.uid
                        chords = get_chordsheet()
                        return render_template('logout.html',t=chords)
                    else:
                        text = "Opps registered but failed to login you"
                        return render_template('error-msg.html',text=text)
                else:
                    text = "Opps !! Account created but failed to save details. Please try again later"
                    return render_template('error-msg.html',text=text)
            except:
                text = "OPPS!! Failed to register \n Try again later"
                return render_template('error-msg.html',text=text)
        else:
            text = "PLEASE ENTER ALL THE DETAILS AND PASSWORD SHOULD  NOT BE LESS THAN 6 CHARACHTERS"
            return render_template('error-msg.html',text=text)
    return render_template('signup.html')
''' End of Sign Up '''



''' Start of Profile page '''
#when clicked on profile on homepage
@app.route('/profile')
def profile():
    if('user' in session and 'id' in session):
        data_chord=[]
        data_realtime=[]
        data_test=[]
        id_token = session['id']
        doc_ref = db_firestore.collection('Users').document(id_token)
        chord_ref = db_firestore.collection(f'Users/{id_token}/ChordSheets')
        chord_ref_list = chord_ref.stream() 
        chord_realtime = db_firestore.collection(f'Users/{id_token}/Realtime_Chordsheets')
        chord_realtime_list = chord_realtime.stream()
        chord_test = db_firestore.collection(f'Users/{id_token}/Test_Series')
        chord_test_list = chord_test.stream()
        for chord in chord_ref_list:
            data_chord.append(chord.to_dict())
        for real_chord in chord_realtime_list:
            data_realtime.append(real_chord.to_dict())
        for test in chord_test_list:
            data_test.append(test.to_dict())
        doc = doc_ref.get()
        if doc.exists:
            details = doc.to_dict()
            name = details['name']
            email = details['email']
            level = details['level']
            return render_template('profile.html',name=name, email=email, level=level, chord = data_chord, real_chord=data_realtime, test = data_test)
        else:
            chords = get_chordsheet()
            return render_template('signin.html')
    else:
        chords = get_chordsheet()
        return render_template('signin.html')

''' End of profile page '''



''' Start of Generating chord Sheets'''
#for uploading video
@app.route('/upload',methods=['POST', 'GET'])
def upload():
    global save
    global name_video
    global genre_video
    global name_video_realtime
    global genre_video_realtime
    if request.method == 'POST':
        try:
            file = request.files.get('video')
            name_video = request.form.get('name_of_video')
            genre_video = request.form.get('genre_of_video')
            name_video_realtime = request.form.get('name_of_video_realtime')
            genre_video_realtime = request.form.get('genre_of_video_realtime')
            if name_video and genre_video and file:
                user_id = session['id']
                save = save_video(file, name_video, genre_video,user_id)
                print("completed")
                return redirect(url_for('video'))
            elif name_video_realtime and genre_video_realtime:
                return render_template('realtime_chordsheet.html')
            elif name_video and genre_video and name_video_realtime and genre_video_realtime and file:
                text = "You cannot enter both the values at same time Please select one Video to chordsheet or Realtime chordsheet"
                return render_template('error-msg.html',text=text)
            else:
                text = "Please enter the values"
                return render_template('error-msg.html',text=text)
        except:
            text = "Due to some issue your request couldn't be parsed"
            return render_template('error-msg.html',text=text)
    return render_template('get-started.html')

'''End of Generating Chord Sheets'''





''' Start of Video to Chord Sheet '''

@app.route('/video')
def video():
    return render_template('video_chordsheet.html')


#for video to chord sheet generation
@app.route('/video_to_chordsheet')
def video_to_chordsheet():
    print("inside video")
    print(save)
    return Response(process_video_for_chordsheet(save),mimetype='multipart/x-mixed-replace; boundary=frame')

#for saving chordsheet
@app.route('/save_chordsheet')
def save_chordsheet():
    user_id = session['id']
    url = chord_sheet_to_file_link_video(user_id,save,name_video,genre_video)
    try:
        print('inside save chordsheet video')
        data ={'Name':name_video, 'Genre':genre_video, 'ChordSheet Link':url}
        db_firestore.collection('Users').document(user_id).collection('ChordSheets').add(data)
        #save_chord_sheet(url,user_id,name_video,genre_video)
        return render_template('get-started.html')
    except:
        text = "OPPS!!! FAILED TO SAVE THE CHORD!! PLEASE TRY AGAIN LATER"
        return render_template('error-msg.html',text=text)

''' End of Video to Chord Sheet '''



''' Start to Realtime Chordsheets'''
#for realtime chordsheet generation
@app.route('/realtime_chordsheet')
def realtime_chordsheet():
    print('inside realtime chordsheet')
    return Response(process_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

#for saving the realtime chordsheets
@app.route('/save_realtime_chordsheet')
def save_realtime_chordsheet():
    user_id = session['id']
    url_realtime = chord_sheet_to_file_realtime(user_id,name_video_realtime,genre_video_realtime)
    try:
        data = {'Name': name_video_realtime, 'Genre':genre_video_realtime, 'Realtime ChordSheet Link': url_realtime}
        db_firestore.collection('Users').document(user_id).collection('Realtime_Chordsheets').add(data)
        return render_template('get-started.html')
    except:
        text = "OPPS!! FAILED TO SAVE THE CHORD!! PLEASE TRY AGAIN LATER"
        return render_template('error-msg.html',text=text)

''' End of Realtime Chordsheets'''



'''Start of Practice ChordSheets'''
@app.route('/chordsheet_list_practice')
def chordsheet_list_practice():
    chords = get_all_chordsheet()
    realtime_chord = get_realtime()
    return render_template('chordsheet_list_practice.html',t = chords, y=realtime_chord)

@app.route('/display_chords')
def display_chords():
    url = request.args.get('url')
    f = urllib.request.urlopen(url).read()
    return render_template('practice_chordsheet.html',chord = f)

@app.route('/stop_practice_chordsheet')
def stop_practice_chordsheet():
    cv2.destroyAllWindows()
    return render_template('index.html')

'''End of Practice ChordSheets'''




''' Start of Practice chords '''

#for practicing chords
@app.route('/practice_feed_route')
def practice_feed_route():
    return Response(process_video(),mimetype='multipart/x-mixed-replace; boundary=frame')

#for chord A
@app.route('/a_chord')
def a_chord():
    return render_template('chord_page_A.html')

#for chord C
@app.route('/c_chord')
def c_chord():
    return render_template('chord_page_C.html')

#for chord F
@app.route('/f_chord')
def f_chord():
    return render_template('chord_page_F.html')

#for chord G
@app.route('/g_chord')
def g_chord():
    return render_template('chord_page_G.html')

@app.route('/back_to_get_started')
def back_to_get_started():
    stop_camera_feed()
    return redirect(url_for('get_started'))

''' End of Practice Chords'''



''' Test Series Code '''
@app.route("/test_series")
def test_series():
    return render_template("test-chords.html")

'''A Chord Test'''
@app.route("/a_chord_test")
def a_chord_test():
    return render_template('test-series-A.html')

@app.route("/resut_a")
def result_a():
    cv2.destroyAllWindows()
    user_id = session['id']
    result,accuracy,score = generate_score("A")
    try:
        data = {'Chord':'A','Time':firestore.SERVER_TIMESTAMP , 'Accuracy of Chord': accuracy , 'Score of Chord':score}
        db_firestore.collection('Users').document(user_id).collection('Test_Series').add(data)
        return render_template('test-result-display.html',result_chord = result)
    except:
        text = "Opssss!!! There's some issue n saving and displaying your result"
        return render_template('error-msg.html',text=text)

'''C Chord Test'''
@app.route("/c_chord_test")
def c_chord_test():
    return render_template('test-series-C.html')

@app.route("/resut_c")
def result_c():
    cv2.destroyAllWindows()
    user_id = session['id']
    result,accuracy,score = generate_score("C")
    chord = 'C'
    try:
        data = {'Chord':chord, 'Time':firestore.SERVER_TIMESTAMP, 'Accuracy of Chord':accuracy , 'Score of Chord':score}
        db_firestore.collection('Users').document(user_id).collection('Test_Series').add(data)
        return render_template('test-result-display.html',result_chord = result)
    except:
        text = "Opssss!!! There's some issue n saving and displaying your result"
        return render_template('error-msg.html',text=text)


'''F Chord Test'''
@app.route("/f_chord_test")
def f_chord_test():
    return render_template('test-series-F.html')

@app.route("/resut_f")
def result_f():
    cv2.destroyAllWindows()
    user_id = session['id']
    result,accuracy,score = generate_score("F")
    try:
        data = {'Chord':'F','Time':firestore.SERVER_TIMESTAMP , 'Accuracy of Chord':accuracy , 'Score of Chord':score}
        db_firestore.collection('Users').document(user_id).collection('Test_Series').add(data)
        return render_template('test-result-display.html',result_chord = result)
    except:
        text = "Opssss!!! There's some issue n saving and displaying your result"
        return render_template('error-msg.html',text=text)

'''G Chord Test'''
@app.route("/g_chord_test")
def g_chord_test():
    return render_template('test-series-G.html')

@app.route("/resut_g")
def result_g():
    
    cv2.destroyAllWindows()
    user_id = session['id']
    result,accuracy,score = generate_score("G")
    try:
        time = firestore.SERVER_TIMESTAMP
        time_convert = custom_converter(time)
        data = {'Chord':'G','Time':time_convert , 'Accuracy of Chord':accuracy , 'Score of Chord':score}
        db_firestore.collection('Users').document(user_id).collection('Test_Series').add(data)
        return render_template('test-result-display.html',result_chord = result)
    except:
        text = "Opssss!!! There's some issue n saving and displaying your result"
        return render_template('error-msg.html',text=text)

#link for displaying video
@app.route("/test")
def test():
    return Response(process_video_test(),mimetype='multipart/x-mixed-replace; boundary=frame')

'''End of Test Series Feature'''
    


'''Some Extra functions'''
def custom_converter(value):
    if isinstance(value, firestore.SERVER_TIMESTAMP):
        return datetime.utcnow()
    else:
        return value
    

#initializing main function
if __name__ == '__main__':
    app.run(debug=True)
    

