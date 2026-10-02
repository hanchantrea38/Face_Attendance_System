from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import os
import sqlite3
import csv
import io
from contextlib import closing
from datetime import datetime
import base64

app = Flask(__name__)

# Anything that fails while the app starts is stored here instead of crashing
# the whole server, so the pages still load and /health shows the real error.
STARTUP_ERRORS = []

try:
    import cv2
except Exception as e:
    cv2 = None
    STARTUP_ERRORS.append(f'OpenCV failed to load: {e}')


# =========================
# Database connection
# =========================
# All data (SQLite database, face dataset, trained model, CSV exports)
# lives in folders next to this file: database/, dataset/, trained_data/, exports/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Serverless hosts like Vercel have a read-only file system except /tmp.
# Copy the bundled data there so the app can still write (data in /tmp is
# temporary and will be lost when the function restarts).
if os.environ.get('VERCEL'):
    import shutil
    TMP_DIR = '/tmp/face_attendance'
    for folder in ('database', 'dataset', 'trained_data'):
        src = os.path.join(BASE_DIR, folder)
        dst = os.path.join(TMP_DIR, folder)
        try:
            if os.path.isdir(src) and not os.path.exists(dst):
                shutil.copytree(src, dst)
        except Exception as e:
            STARTUP_ERRORS.append(f'Could not copy {folder} to /tmp: {e}')
    BASE_DIR = TMP_DIR

DATABASE_FOLDER = os.path.join(BASE_DIR, 'database')
DATABASE_NAME = 'attendance.db'
DB_PATH = os.path.join(DATABASE_FOLDER, DATABASE_NAME)
DATASET_FOLDER = os.path.join(BASE_DIR, 'dataset')
TRAINED_DATA_FOLDER = os.path.join(BASE_DIR, 'trained_data')


def get_db_connection():
    os.makedirs(DATABASE_FOLDER, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# Initialize database
def init_db():
    with closing(get_db_connection()) as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS attendance
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      name TEXT NOT NULL,
                      date TEXT NOT NULL,
                      time TEXT NOT NULL)''')
        c.execute('''CREATE TABLE IF NOT EXISTS students
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      name TEXT NOT NULL UNIQUE,
                      image_count INTEGER NOT NULL DEFAULT 0,
                      created_at TEXT NOT NULL,
                      updated_at TEXT NOT NULL)''')
        conn.commit()
        print(f"Connected to database: {DB_PATH}")

# Create necessary directories
try:
    os.makedirs(DATASET_FOLDER, exist_ok=True)
    os.makedirs(TRAINED_DATA_FOLDER, exist_ok=True)
    os.makedirs(DATABASE_FOLDER, exist_ok=True)
except Exception as e:
    STARTUP_ERRORS.append(f'Could not create data folders: {e}')

# Initialize face detector and recognizer
face_cascade = None
if cv2 is not None:
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    except Exception as e:
        STARTUP_ERRORS.append(f'Face detector failed to load: {e}')
using_simple_recognizer = False
SIMPLE_RECOGNIZER_THRESHOLD = 350.0
LBPH_RECOGNIZER_THRESHOLD = 100.0


def preprocess_face(face):
    face = cv2.resize(face, (100, 100))
    return cv2.equalizeHist(face)

# Try different recognizer options
try:
    # First try the contrib version
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    print("Using OpenCV contrib face recognizer")
except AttributeError:
    try:
        # Fallback to basic OpenCV
        recognizer = cv2.face.createLBPHFaceRecognizer()
        print("Using legacy OpenCV face recognizer")
    except AttributeError:
        # Final fallback - create a simple recognizer class
        class SimpleFaceRecognizer:
            def __init__(self):
                self.labels = []
                self.faces = []
                self.features = []

            def extract_features(self, face):
                face = preprocess_face(face)
                center = face[1:-1, 1:-1]
                codes = np.zeros_like(center, dtype=np.uint8)
                neighbors = [
                    face[:-2, :-2], face[:-2, 1:-1], face[:-2, 2:],
                    face[1:-1, 2:], face[2:, 2:], face[2:, 1:-1],
                    face[2:, :-2], face[1:-1, :-2]
                ]

                for bit, neighbor in enumerate(neighbors):
                    codes |= ((neighbor >= center).astype(np.uint8) << bit)

                grid_size = 8
                height, width = codes.shape
                features = []

                for row in range(grid_size):
                    for col in range(grid_size):
                        cell = codes[
                            row * height // grid_size:(row + 1) * height // grid_size,
                            col * width // grid_size:(col + 1) * width // grid_size
                        ]
                        hist, _ = np.histogram(cell, bins=256, range=(0, 256))
                        hist = hist.astype('float32')
                        hist /= hist.sum() + 1e-7
                        features.append(hist)

                return np.concatenate(features)
                
            def train(self, faces, labels):
                self.faces = [preprocess_face(face) for face in faces]
                self.labels = np.array(labels)
                self.features = np.array([self.extract_features(face) for face in self.faces])
                
            def predict(self, face):
                if len(self.features) == 0:
                    return (0, 1000)
                
                min_dist = float('inf')
                best_label = 0
                feature = self.extract_features(face)
                
                for i, trained_feature in enumerate(self.features):
                    dist = 0.5 * np.sum(
                        ((trained_feature - feature) ** 2) /
                        (trained_feature + feature + 1e-7)
                    )
                    if dist < min_dist:
                        min_dist = dist
                        best_label = self.labels[i]
                
                return (int(best_label), float(min_dist))
                
            def save(self, filename):
                # Save training data
                np.savez(filename, faces=np.array(self.faces), labels=self.labels, features=self.features)
                
            def read(self, filename):
                # Load training data
                if os.path.exists(filename):
                    data = np.load(filename, allow_pickle=True)
                    self.faces = list(data['faces'])
                    self.labels = data['labels']
                    if 'features' in data:
                        self.features = data['features']
                    else:
                        self.features = np.array([self.extract_features(face) for face in self.faces])
        
        recognizer = SimpleFaceRecognizer()
        using_simple_recognizer = True
        print("Using simple face recognizer")

try:
    init_db()
except Exception as e:
    STARTUP_ERRORS.append(f'Database setup failed: {e}')


def startup_error_response():
    """Return a JSON error if face features can't work, otherwise None."""
    if cv2 is None or face_cascade is None:
        return jsonify({'success': False, 'message': 'Server setup error: ' + '; '.join(STARTUP_ERRORS)})
    return None

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    if STARTUP_ERRORS:
        return jsonify({'success': False, 'status': 'error', 'errors': STARTUP_ERRORS})
    return jsonify({'success': True, 'status': 'ok'})

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/attendance')
def attendance():
    return render_template('attendance.html')

@app.route('/view_records')
def view_records():
    return render_template('view_records.html')

@app.route('/api/register_face', methods=['POST'])
def register_face():
    error = startup_error_response()
    if error:
        return error
    try:
        name = request.form['name'].strip()
        image_data = request.form['image']

        if not name:
            return jsonify({'success': False, 'message': 'Please enter a name'})
        
        # Convert base64 image to OpenCV format
        image_data = image_data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            return jsonify({'success': False, 'message': 'No face detected'})
        
        # Save face images for training
        face_dir = os.path.join(DATASET_FOLDER, name)
        os.makedirs(face_dir, exist_ok=True)
        
        count = len([f for f in os.listdir(face_dir) if f.endswith('.jpg')])
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            # Resize to standard size for better recognition
            face_roi = preprocess_face(face_roi)
            cv2.imwrite(os.path.join(face_dir, f'{count + 1}.jpg'), face_roi)
            count += 1

        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with closing(get_db_connection()) as conn:
            c = conn.cursor()
            c.execute('''INSERT INTO students (name, image_count, created_at, updated_at)
                         VALUES (?, ?, ?, ?)
                         ON CONFLICT(name) DO UPDATE SET
                            image_count = excluded.image_count,
                            updated_at = excluded.updated_at''',
                      (name, count, current_time, current_time))
            conn.commit()
        
        # Train the recognizer
        train_recognizer()
        
        return jsonify({'success': True, 'message': f'Face registered successfully for {name}. Captured {len(faces)} face(s).'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/mark_attendance', methods=['POST'])
def mark_attendance():
    error = startup_error_response()
    if error:
        return error
    try:
        image_data = request.form['image']
        
        # Convert base64 image to OpenCV format
        image_data = image_data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            return jsonify({'success': False, 'message': 'No face detected'})
        
        # Load trained recognizer
        trainer_path = os.path.join(TRAINED_DATA_FOLDER, 'trainer.yml')
        trainer_npz_path = os.path.join(TRAINED_DATA_FOLDER, 'trainer.npz')
        model_path = trainer_npz_path if using_simple_recognizer else trainer_path
        if os.path.exists(model_path):
            recognizer.read(model_path)
        else:
            return jsonify({'success': False, 'message': 'No trained face data found. Please register a face first.'})
        
        recognized_names = []
        already_marked_names = []
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            # Resize to match training size
            face_roi = preprocess_face(face_roi)
            
            # Recognize face
            try:
                label, confidence = recognizer.predict(face_roi)
                
                # Adjust confidence threshold based on recognizer type
                confidence_threshold = SIMPLE_RECOGNIZER_THRESHOLD if using_simple_recognizer else LBPH_RECOGNIZER_THRESHOLD
                print(f"Best match label={label}, confidence={confidence}, threshold={confidence_threshold}")
                
                if confidence < confidence_threshold:
                    name = get_name_from_label(label)
                    if name:
                        # Mark attendance
                        current_time = datetime.now()
                        date_str = current_time.strftime('%Y-%m-%d')
                        time_str = current_time.strftime('%H:%M:%S')
                        
                        with closing(get_db_connection()) as conn:
                            c = conn.cursor()
                            
                            # Check if already marked today
                            c.execute('''SELECT * FROM attendance 
                                        WHERE name = ? AND date = ?''', (name, date_str))
                            existing = c.fetchone()
                            
                            if not existing:
                                c.execute('''INSERT INTO attendance (name, date, time)
                                            VALUES (?, ?, ?)''', (name, date_str, time_str))
                                conn.commit()
                                recognized_names.append(name)
                                print(f"Attendance marked for {name} with confidence {confidence}")
                            else:
                                already_marked_names.append(name)
                                print(f"{name} was recognized, but attendance is already marked for today")
                else:
                    name = get_name_from_label(label) or 'Unknown'
                    print(f"Rejected match for {name}: confidence {confidence} is above threshold {confidence_threshold}")
            except Exception as e:
                print(f"Recognition error: {e}")
                continue
        
        if recognized_names:
            return jsonify({'success': True, 'message': f'Attendance marked for: {", ".join(recognized_names)}'})
        elif already_marked_names:
            return jsonify({'success': True, 'message': f'Already marked today for: {", ".join(already_marked_names)}'})
        else:
            return jsonify({'success': False, 'message': 'Face not recognized. Please register more face images with good lighting.'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/get_attendance')
def get_attendance():
    try:
        with closing(get_db_connection()) as conn:
            c = conn.cursor()
            c.execute('SELECT * FROM attendance ORDER BY date DESC, time DESC')
            records = c.fetchall()
        
        attendance_data = []
        for record in records:
            attendance_data.append({
                'id': record[0],
                'name': record[1],
                'date': record[2],
                'time': record[3]
            })
        
        return jsonify({'success': True, 'data': attendance_data})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/export_csv')
def export_csv():
    try:
        with closing(get_db_connection()) as conn:
            c = conn.cursor()
            c.execute('SELECT * FROM attendance ORDER BY date DESC, time DESC')
            records = c.fetchall()
        
        # Build the CSV in memory and send it directly - no temp file needed,
        # so this works even on read-only or ephemeral file systems
        csv_data = io.StringIO()
        writer = csv.writer(csv_data)
        writer.writerow(['ID', 'Name', 'Date', 'Time'])
        writer.writerows(records)

        export_name = f"attendance_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        return send_file(
            io.BytesIO(csv_data.getvalue().encode('utf-8-sig')),
            as_attachment=True,
            download_name=export_name,
            mimetype='text/csv'
        )
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

def train_recognizer():
    faces = []
    labels = []
    label_dict = {}
    current_label = 0
    
    # Collect face samples and labels
    for person_name in os.listdir(DATASET_FOLDER):
        person_dir = os.path.join(DATASET_FOLDER, person_name)
        if os.path.isdir(person_dir):
            label_dict[current_label] = person_name
            for image_name in os.listdir(person_dir):
                if image_name.endswith('.jpg'):
                    image_path = os.path.join(person_dir, image_name)
                    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    # Resize to standard size
                    img = preprocess_face(img)
                    faces.append(img)
                    labels.append(current_label)
            current_label += 1
    
    if faces and labels:
        try:
            recognizer.train(faces, np.array(labels))
            # Save based on recognizer type
            if using_simple_recognizer:
                recognizer.save(os.path.join(TRAINED_DATA_FOLDER, 'trainer.npz'))
            else:
                recognizer.save(os.path.join(TRAINED_DATA_FOLDER, 'trainer.yml'))
            
            # Save label mapping
            with open(os.path.join(TRAINED_DATA_FOLDER, 'labels.txt'), 'w') as f:
                for label, name in label_dict.items():
                    f.write(f'{label},{name}\n')
                    
            print(f"Trained recognizer with {len(faces)} faces from {len(label_dict)} people")
        except Exception as e:
            print(f"Training error: {e}")

def get_name_from_label(label):
    try:
        with open(os.path.join(TRAINED_DATA_FOLDER, 'labels.txt'), 'r') as f:
            for line in f:
                lbl, name = line.strip().split(',')
                if int(lbl) == label:
                    return name
    except:
        pass
    return None

if __name__ == '__main__':
    # Production-friendly: bind to all interfaces and use the PORT env var
    # (required by Render, Railway, Heroku and other cloud platforms)
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(host='0.0.0.0', port=port, debug=debug, use_reloader=False)
