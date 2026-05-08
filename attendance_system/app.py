from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file
import cv2
import numpy as np
import os
import sqlite3
import csv
from datetime import datetime
import base64

app = Flask(__name__)



# Initialize database
def init_db():
    conn = sqlite3.connect('database/attendance.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS attendance
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL,
                  date TEXT NOT NULL,
                  time TEXT NOT NULL)''')
    conn.commit()
    conn.close()

# Create necessary directories
os.makedirs('dataset', exist_ok=True)
os.makedirs('trained_data', exist_ok=True)
os.makedirs('database', exist_ok=True)

# Initialize face detector and recognizer
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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
                self.label_map = {}
                
            def train(self, faces, labels):
                self.faces = faces
                self.labels = labels
                # Simple training - just store the data
                
            def predict(self, face):
                # Simple distance-based recognition
                if not self.faces:
                    return (0, 1000)
                
                min_dist = float('inf')
                best_label = 0
                
                for i, trained_face in enumerate(self.faces):
                    if trained_face.shape == face.shape:
                        dist = np.sqrt(np.sum((trained_face - face) ** 2))
                        if dist < min_dist:
                            min_dist = dist
                            best_label = self.labels[i]
                
                confidence = min(100, max(0, min_dist / 10))
                return (best_label, confidence)
                
            def save(self, filename):
                # Save training data
                np.savez(filename, faces=self.faces, labels=self.labels, label_map=self.label_map)
                
            def read(self, filename):
                # Load training data
                if os.path.exists(filename):
                    data = np.load(filename, allow_pickle=True)
                    self.faces = data['faces']
                    self.labels = data['labels']
                    self.label_map = data['label_map'].item()
        
        recognizer = SimpleFaceRecognizer()
        print("Using simple face recognizer")

init_db()

@app.route('/')
def index():
    return render_template('index.html')

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
    try:
        name = request.form['name']
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
        
        # Save face images for training
        face_dir = f'dataset/{name}'
        os.makedirs(face_dir, exist_ok=True)
        
        count = len([f for f in os.listdir(face_dir) if f.endswith('.jpg')])
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            # Resize to standard size for better recognition
            face_roi = cv2.resize(face_roi, (100, 100))
            cv2.imwrite(f'{face_dir}/{count + 1}.jpg', face_roi)
            count += 1
        
        # Train the recognizer
        train_recognizer()
        
        return jsonify({'success': True, 'message': f'Face registered successfully for {name}. Captured {len(faces)} face(s).'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/mark_attendance', methods=['POST'])
def mark_attendance():
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
        if os.path.exists('trained_data/trainer.yml'):
            try:
                recognizer.read('trained_data/trainer.yml')
            except:
                # Try loading from NPZ if using simple recognizer
                if hasattr(recognizer, 'read'):
                    recognizer.read('trained_data/trainer.npz')
        
        recognized_names = []
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            # Resize to match training size
            face_roi = cv2.resize(face_roi, (100, 100))
            
            # Recognize face
            try:
                label, confidence = recognizer.predict(face_roi)
                
                # Adjust confidence threshold based on recognizer type
                confidence_threshold = 70 if hasattr(recognizer, '__class__') and 'Simple' in str(recognizer.__class__) else 100
                
                if confidence < confidence_threshold:
                    name = get_name_from_label(label)
                    if name:
                        # Mark attendance
                        current_time = datetime.now()
                        date_str = current_time.strftime('%Y-%m-%d')
                        time_str = current_time.strftime('%H:%M:%S')
                        
                        conn = sqlite3.connect('database/attendance.db')
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
                        
                        conn.close()
            except Exception as e:
                print(f"Recognition error: {e}")
                continue
        
        if recognized_names:
            return jsonify({'success': True, 'message': f'Attendance marked for: {", ".join(recognized_names)}'})
        else:
            return jsonify({'success': False, 'message': 'No recognized faces or attendance already marked'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/get_attendance')
def get_attendance():
    try:
        conn = sqlite3.connect('database/attendance.db')
        c = conn.cursor()
        c.execute('SELECT * FROM attendance ORDER BY date DESC, time DESC')
        records = c.fetchall()
        conn.close()
        
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
        conn = sqlite3.connect('database/attendance.db')
        c = conn.cursor()
        c.execute('SELECT * FROM attendance ORDER BY date DESC, time DESC')
        records = c.fetchall()
        conn.close()
        
        csv_filename = 'attendance_export.csv'
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['ID', 'Name', 'Date', 'Time'])
            writer.writerows(records)
        
        return send_file(csv_filename, as_attachment=True)
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

def train_recognizer():
    faces = []
    labels = []
    label_dict = {}
    current_label = 0
    
    # Collect face samples and labels
    for person_name in os.listdir('dataset'):
        person_dir = os.path.join('dataset', person_name)
        if os.path.isdir(person_dir):
            label_dict[current_label] = person_name
            for image_name in os.listdir(person_dir):
                if image_name.endswith('.jpg'):
                    image_path = os.path.join(person_dir, image_name)
                    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    # Resize to standard size
                    img = cv2.resize(img, (100, 100))
                    faces.append(img)
                    labels.append(current_label)
            current_label += 1
    
    if faces and labels:
        try:
            recognizer.train(faces, np.array(labels))
            # Save based on recognizer type
            if hasattr(recognizer, 'save'):
                recognizer.save('trained_data/trainer.yml')
            else:
                recognizer.save('trained_data/trainer.npz')
            
            # Save label mapping
            with open('trained_data/labels.txt', 'w') as f:
                for label, name in label_dict.items():
                    f.write(f'{label},{name}\n')
                    
            print(f"Trained recognizer with {len(faces)} faces from {len(label_dict)} people")
        except Exception as e:
            print(f"Training error: {e}")

def get_name_from_label(label):
    try:
        with open('trained_data/labels.txt', 'r') as f:
            for line in f:
                lbl, name = line.strip().split(',')
                if int(lbl) == label:
                    return name
    except:
        pass
    return None

if __name__ == '__main__':
    app.run(debug=True)