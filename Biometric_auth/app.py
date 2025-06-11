from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_session import Session
import pandas as pd
from psycopg2 import OperationalError
from sqlalchemy import PickleType
from werkzeug.utils import secure_filename

import base64
import os
import uuid
import numpy as np
from pydub import AudioSegment
import io
from deepface import DeepFace
from resemblyzer import VoiceEncoder, preprocess_wav
from numpy import dot
from numpy.linalg import norm

from flask import jsonify
import joblib

# ---------- App Configuration ----------
app = Flask(__name__)
app.secret_key = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:Siva%40123@localhost:5432/biometrics'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SESSION_TYPE'] = 'filesystem'

Session(app)
db = SQLAlchemy(app)

# ---------- Resemblyzer Voice Encoder ----------
voice_encoder = VoiceEncoder()

# ---------- Database Model ----------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    face_embedding = db.Column(PickleType, nullable=True)
    voice_embedding = db.Column(PickleType, nullable=True)

# ---------- Helper Functions ----------
def extract_face_embedding(image_path):
    try:
        obj = DeepFace.represent(img_path=image_path, model_name='VGG-Face', enforce_detection=False)
        embedding = np.array(obj[0]['embedding'])
        return embedding
    except Exception as e:
        print(f"❌ Face embedding error: {e}")
        return None

def extract_voice_embedding(audio_path):
    try:
        wav = preprocess_wav(audio_path)
        embedding = voice_encoder.embed_utterance(wav)
        print(f"✅ Voice embedding generated. Length: {len(embedding)}")
        return embedding
    except Exception as e:
        print("❌ Resemblyzer voice processing error:", e)
        return None

def is_match(embedding1, embedding2, threshold=0.6):
    if embedding1 is None or embedding2 is None:
        return False

    embedding1 = np.array(embedding1).flatten()
    embedding2 = np.array(embedding2).flatten()

    if np.isnan(embedding1).any() or np.isnan(embedding2).any():
        return False
    
    # cosine similarity
    similarity = dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))
    print(f"🔍similarity: {similarity}")

    return similarity > (1 - threshold)

def save_image(data_url, folder='static/faces'):
    os.makedirs(folder, exist_ok=True)
    encoded_data = data_url.split(',')[1]
    img_data = base64.b64decode(encoded_data)
    filename = f"{uuid.uuid4()}.png"
    filepath = os.path.join(folder, filename)
    with open(filepath, 'wb') as f:
        f.write(img_data)
    return filepath

def save_audio_file(file_data, folder='static/voices'):
    os.makedirs(folder, exist_ok=True)
    filename = f"{uuid.uuid4()}.wav"
    filepath = os.path.join(folder, filename)

    try:
        audio = AudioSegment.from_file(io.BytesIO(file_data), format="webm")
        audio.export(filepath, format="wav")
    except Exception as e:
        print(f"❌ Audio error: {e}")
        return None

    return filepath

# ---------- Routes ----------
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login')
def login():
    return render_template('login.html')  # Page with face and voice login buttons

@app.route('/register')
def register():
    return render_template('register.html')  # Page with face and voice login buttons

@app.route('/game')
def game():
    return render_template('game.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/speech')
def speech():
    return render_template('speech.html')

# -------------------- REGISTER --------------------

@app.route('/register/face', methods=['GET', 'POST'])
def register_face():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        face_data = request.form['faceImage']

        if User.query.filter_by(email=email).first():
            flash('Email already exists!')
            print(f"❌ Registration failed: Email {email} already exists.")
            return redirect(url_for('register'))

        face_path = save_image(face_data)
        face_embedding = extract_face_embedding(face_path)
        os.remove(face_path)

        if face_embedding is None or np.any(np.isnan(face_embedding)):
            flash('Failed to extract face embedding.')
            print(f"❌ Registration failed: Face embedding extraction failed for {email}.")
            return redirect(url_for('register'))

        new_user = User(username=username, email=email, face_embedding=face_embedding, voice_embedding=[])
        db.session.add(new_user)
        db.session.commit()

        flash('Face registration successful!')
        print(f"✅ Face registration successful for {email}.")
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/register/voice', methods=['GET', 'POST'])
def register_voice():
    if request.method == 'POST':
        username = request.form['voiceUsername']
        email = request.form['voiceEmail']
        voice_data = request.form['voiceData']

        if User.query.filter_by(email=email).first():
            flash('Email already exists!')
            print(f"❌ Registration failed: Email {email} already exists.")
            return redirect(url_for('register'))

        try:
            voice_binary = base64.b64decode(voice_data.split(',')[1])
        except Exception:
            flash('Invalid voice data format.')
            print(f"❌ Registration failed: Invalid voice data format for {email}.")
            return redirect(url_for('register'))

        voice_path = save_audio_file(voice_binary)
        voice_embedding = extract_voice_embedding(voice_path)
        os.remove(voice_path)

        if voice_embedding is None or np.any(np.isnan(voice_embedding)):
            flash('Failed to extract voice embedding.')
            print(f"❌ Registration failed: Voice embedding extraction failed for {email}.")
            return redirect(url_for('register'))

        new_user = User(username=username, email=email, face_embedding=[], voice_embedding=voice_embedding)
        db.session.add(new_user)
        db.session.commit()

        flash('Voice registration successful!')
        print(f"✅ Voice registration successful for {email}.")
        return redirect(url_for('login'))

    return render_template('register.html')




# -------------------- LOGIN --------------------

@app.route('/login/face', methods=['GET', 'POST']) 
def login_face():
    if request.method == 'POST':
        email = request.form['email']
        face_data = request.form['loginFaceImage']
        print(f"📨 Received face login request for: {email}")

        user = User.query.filter_by(email=email).first()
        if not user or user.face_embedding is None or len(user.face_embedding) == 0:
            print("⚠️ User not found or face embedding not registered.")
            flash('User not found or face data not registered.')
            return redirect(url_for('login'))

        face_path = save_image(face_data)
        print(f"📸 Face image saved temporarily at: {face_path}")

        input_embedding = extract_face_embedding(face_path)
        os.remove(face_path)
        print("🧹 Temporary face image deleted.")

        if input_embedding is None:
            print("❌ Failed to extract face embedding.")
            flash('Failed to extract face embedding.')
            return redirect(url_for('login'))

        face_match = is_match(input_embedding, user.face_embedding)

        if face_match:
            print(f"✅ Face match successful for: {email}")
        else:
            print(f"❌ Face match failed for: {email}")

        if face_match:
            session['email'] = user.email
            flash('Face login successful!')
            return redirect(url_for('dashboard'))
        else:
            flash('Face login failed.')
            return redirect(url_for('login_face'))

    return render_template('login.html')


@app.route('/login/voice', methods=['GET', 'POST'])
def login_voice():
    if request.method == 'POST':
        email = request.form['voiceEmail']
        voice_data = request.form['voiceLoginData']
        print(f"📨 Received voice login request for: {email}")

        user = User.query.filter_by(email=email).first()
        if user is None or user.voice_embedding is None or len(user.voice_embedding) == 0:
            print("⚠️ User not found or voice embedding not registered.")
            flash('User not found or voice data not registered.')
            return redirect(url_for('login'))

        try:
            voice_binary = base64.b64decode(voice_data.split(',')[1])
            print("🎤 Voice data decoded successfully.")
        except Exception as e:
            print(f"❌ Error decoding voice data: {e}")
            flash('Invalid voice data format.')
            return redirect(url_for('login'))

        voice_path = save_audio_file(voice_binary)
        print(f"🎧 Voice audio saved temporarily at: {voice_path}")

        input_embedding = extract_voice_embedding(voice_path)
        os.remove(voice_path)
        print("🧹 Temporary voice file deleted.")

        if input_embedding is None:
            print("❌ Failed to extract voice embedding.")
            flash('Failed to extract voice embedding.')
            return redirect(url_for('login'))

        voice_match = is_match(input_embedding, user.voice_embedding, threshold=0.6)
        if voice_match:
            print(f"✅ Voice match successful for: {email}")
        else:
            print(f"❌ Voice match failed for: {email}")

        if voice_match:
            session['email'] = user.email
            flash('Voice login successful!')
            return redirect(url_for('dashboard'))
        else:
            flash('Voice login failed.')
            return redirect(url_for('login'))

    return render_template('login.html')


# ---------- game control -------------------

# Load the dataset for reference (if needed for columns)
data = pd.read_csv('dataset/mental-state.csv')  # Adjust the path to your dataset

# Load model and scaler
svm_model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

# Manually input data - assuming the format of X is the same
manual_input_1 = pd.DataFrame([data.drop('Label', axis=1).iloc[0].values], columns=data.drop('Label', axis=1).columns)
manual_input_2 = pd.DataFrame([data.drop('Label', axis=1).iloc[3].values], columns=data.drop('Label', axis=1).columns)
manual_input_3 = pd.DataFrame([data.drop('Label', axis=1).iloc[7].values], columns=data.drop('Label', axis=1).columns)

# Standardize the manual inputs using the scaler (fit on training data)
scaled_input_1 = scaler.transform(manual_input_1)
scaled_input_2 = scaler.transform(manual_input_2)
scaled_input_3 = scaler.transform(manual_input_3)

# Convert the scaled inputs to lists (flattening them)
freq_band_1 = scaled_input_1.flatten().tolist()
freq_band_2 = scaled_input_2.flatten().tolist()
freq_band_3 = scaled_input_3.flatten().tolist()

state_map = {2.0: "Concentrated", 1.0: "Stressed", 0.0: "Relaxed"}

#fre1 - concentrated fre2 - stress fre3 - relaxed
predefined_data = {
    'easy': {
        'moves': ['down','down','down','down','right','right','right','down','down','right','right','right','right','right','right','down','down','down','down','down','right'],
        'freq_bands': [freq_band_1, freq_band_1, freq_band_1, freq_band_1, freq_band_1, freq_band_1, freq_band_1, freq_band_1, freq_band_1, freq_band_1, freq_band_1, freq_band_1, freq_band_1, freq_band_1, freq_band_1, freq_band_1, freq_band_1, freq_band_1, freq_band_1, freq_band_1, freq_band_1]
    },
    'medium': {
        'moves': ['right','right','down','down','right','right','right','down','down','down','down','right','right','right','right','down','down','down','down','down','right'],
        'freq_bands': [freq_band_1, freq_band_1, freq_band_1, freq_band_1, freq_band_1, freq_band_1, freq_band_1, freq_band_1, freq_band_1, freq_band_1, freq_band_1, freq_band_2, freq_band_1, freq_band_1, freq_band_1, freq_band_1, freq_band_1, freq_band_1, freq_band_1, freq_band_1, freq_band_1]
    },
    'hard': {
        'moves': ['down','down','down','down','right','right','right','down','down','right','right','right','right','right','right','down','down','down','down','down','right'],
        'freq_bands': [freq_band_1, freq_band_1, freq_band_1, freq_band_1, freq_band_1, freq_band_1, freq_band_1, freq_band_1, freq_band_1, freq_band_1, freq_band_1, freq_band_3, freq_band_1, freq_band_1, freq_band_1, freq_band_1, freq_band_1, freq_band_1, freq_band_1, freq_band_1, freq_band_1]
    }
}

@app.route('/start_game', methods=['POST'])
def start_game():
    data = request.get_json()
    difficulty = data.get('difficulty')
    print(f"Starting game with difficulty: {difficulty}")

    if difficulty not in predefined_data:
        return jsonify({'error': 'Invalid difficulty'}), 400

    # Save data to session
    session['difficulty'] = difficulty
    session['current_move_index'] = 0

    return jsonify({'status': 'ready'})


@app.route('/next_move', methods=['POST'])
def next_move():
    difficulty = session.get('difficulty')
    move_index = session.get('current_move_index', 0)

    if not difficulty or difficulty not in predefined_data:
        return jsonify({'error': 'No game in progress'}), 400

    moves = predefined_data[difficulty]['moves']
    freq_bands = predefined_data[difficulty]['freq_bands']

    if move_index >= len(moves):
        return jsonify({'status': 'done'})

    move = moves[move_index]
    freq_input = freq_bands[move_index]

    # Don't scale again!
    input_array = np.array([freq_input])
    prediction = svm_model.predict(input_array)[0]
    mental_state = state_map[prediction]

    if mental_state == "Concentrated":
        session['current_move_index'] = move_index + 1

    print(f"Move: {move}, Mental State: {mental_state}, Move Index: {move_index}")

    return jsonify({
        'status': 'ok',
        'move': move,
        'mental_state': mental_state,
        'freq_band': freq_input
    })

#------------- thought-to-speech---------------
import random

@app.route('/get-thought', methods=['GET'])
def get_thought():
    # Randomly pick one frequency band input (simulating real-time signal)
    freq_input = random.choice([freq_band_1, freq_band_2, freq_band_3])

    # Convert input to array for prediction
    input_array = np.array([freq_input])

    # Predict mental state
    prediction = svm_model.predict(input_array)[0]
    mental_state = state_map[prediction]  # 'Concentrated', 'Stressed', 'Relaxed'

    # Map mental states to multiple possible thoughts
    thought_map = {
    'Concentrated': [
        "I'm laser-focused right now, completely immersed in what I'm doing without any distractions.",
        "All my attention is on this task, and I'm determined to finish it with precision and clarity.",
        "This task has all my concentration; every detail matters and I'm mentally locked in.",
        "I'm fully in the zone, thinking clearly and sharply as if everything else around me has faded away.",
        "My thoughts are aligned, and I feel mentally sharp, as though solving a puzzle piece by piece."
    ],
    'Stressed': [
        "I feel a bit pressured right now, like my mind is racing to keep up with everything happening at once.",
        "It's getting hard to stay calm; there's a weight pressing down on my thoughts and it's overwhelming.",
        "My thoughts are scattered and racing, making it difficult to find clarity or direction in this moment.",
        "This situation is stressing me out—I wish I could take a breath and escape for a while.",
        "My brain feels overworked, like I'm juggling too many things at once and dropping the ball."
    ],
    'Relaxed': [
        "I'm feeling peaceful and relaxed, as if I'm lying under a tree on a warm day with no worries at all.",
        "Everything is calm and easy right now, and my mind is just gently drifting without tension.",
        "My mind is at ease, thoughts flowing slowly like a quiet river, bringing comfort with every moment.",
        "This is such a chill moment—I feel grounded, present, and free from any kind of stress or urgency.",
        "I feel mentally light, like I'm floating in a bubble of calm where nothing can disturb me."
    ]
}

    # Get a random thought based on mental state
    thought_options = thought_map.get(mental_state, ["I'm feeling neutral."])
    thought = random.choice(thought_options)

    print(f"Mental State: {mental_state}, Thought: {thought}")

    return jsonify({
        'mental_state': mental_state,
        'thought': thought,
        'freq_band': freq_input  # Optional: useful for debugging
    })

#------------- Dashboard ---------------
@app.route('/dashboard')
def dashboard():
    if 'email' not in session:
        flash('Please login first.')
        print('please login first')
        return redirect(url_for('login'))

    user = User.query.filter_by(email=session['email']).first()
    if not user:
        flash('User not found.')
        print('user not found')
        return redirect(url_for('login'))

    return render_template('dashboard.html', username=user.username)

#------------ logout ---------
@app.route('/logout')
def logout():
    session.clear()
    flash('✅ You have been logged out successfully.')
    return redirect(url_for('login'))


# ---------- Run App ----------
if __name__ == '__main__':
    with app.app_context():
        try:
            db.create_all()
            print("✅ Database initialized successfully!")
        except OperationalError as e:
            print(f"❌ Database connection failed: {e}")

    app.run(debug=True, use_reloader=False)