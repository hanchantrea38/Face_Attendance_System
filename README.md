# Face Recognition Attendance System

A complete web-based attendance system using facial recognition technology.

## Features

- **Face Registration** — register new faces from the webcam (multiple images)
- **Real-time Attendance** — mark attendance by capturing a face
- **Database Storage** — SQLite database for attendance records
- **Export Functionality** — download attendance records as CSV
- **Responsive Design** — mobile-friendly interface

## Project structure (one single location — no duplicates)

```
Face_Attendance_System/
├── app.py            # Flask application
├── templates/        # HTML pages
├── database/         # SQLite database (attendance.db)
├── dataset/          # Captured face images, one folder per person
├── trained_data/     # Trained model (trainer.yml) + labels.txt
├── Procfile          # Start command for Render/Railway/Heroku
├── runtime.txt       # Python version for Heroku
├── wsgi.py           # WSGI entry point (PythonAnywhere, gunicorn)
└── requirements.txt  # Python dependencies
```

## Local setup (Windows)

1. **Install Python 3.13** (or 3.12/3.11) from [python.org](https://python.org)

2. **Install dependencies**:
   ```bash
   py -m pip install -r requirements.txt
   ```

3. **Run the app**:
   ```bash
   py app.py
   ```
   Open http://localhost:5000 — allow camera access when the browser asks.

> Camera access in the browser requires **HTTPS or localhost**. That's automatic on
> localhost and on the hosting platforms below.

## Deploying online

### ❌ Why Vercel will never work
Vercel only hosts static sites and Node.js/Next.js serverless functions. It cannot
run a Flask + OpenCV + SQLite app: the `cv2` package alone exceeds Vercel's size
limit, and Vercel's file system is read-only, so the database and face dataset
can't be written at runtime. That is the cause of the 404 you saw.

### ✅ Recommended: Render (free)

1. Push this folder to a GitHub repository.
2. Go to [render.com](https://render.com) → **New** → **Web Service**.
3. Connect your repository. Render auto-detects Python.
4. Set:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
5. Click **Create Web Service**. You'll get an `https://...onrender.com` URL.

Your app needs Python **3.12 or 3.13** — Render's default is fine. The webcam will
work because Render gives you HTTPS.

### Railway (also good)

1. Push to GitHub, then create a new project at [railway.app](https://railway.app).
2. **New** → **Deploy from GitHub repo** → choose your repo.
3. Railway auto-detects the `Procfile` (`gunicorn app:app`) and builds automatically.

### PythonAnywhere

1. Upload the files (via the **Files** tab).
2. Create a Python 3.12 **web app** with **manual config**.
3. In the **WSGI configuration file**, replace the contents with:
   ```python
   import sys
   sys.path.insert(0, '/home/YOUR_USERNAME/mysite')
   from app import app as application
   ```
4. Open a Bash console and run `pip install -r requirements.txt`.

### Heroku

Heroku works with the included `Procfile` + `runtime.txt`. Free tier no longer
exists, so this is usually not worth it — use Render instead.

## ⚠️ Important deployment note

On **free hosting** (Render free tier, Railway trial, Heroku), the file system is
**ephemeral** — data saved at runtime (new registrations, attendance records) is
wiped on every redeploy. The trained model and dataset that are committed to git
ship with the app, but anything new is temporary.

If you need records to survive redeploys, add a **persistent disk** on Render
(mount it at the project directory) or move to a paid plan.

## How it works

- `dataset/` holds cropped face images, one folder per person.
- `train_recognizer()` trains an OpenCV **LBPH** model into `trained_data/trainer.yml`
  after every registration.
- `trained_data/labels.txt` maps model labels to names.
- `database/attendance.db` stores the attendance log and registered students.

## API endpoints

| Method | Path                  | Description                            |
|--------|-----------------------|----------------------------------------|
| GET    | `/`                   | Home page                              |
| GET    | `/register`           | Register a face                        |
| GET    | `/attendance`         | Mark attendance                        |
| GET    | `/view_records`       | View attendance records                |
| GET    | `/health`             | Health check (for hosting platforms)   |
| POST   | `/api/register_face`  | Upload a face capture + name           |
| POST   | `/api/mark_attendance`| Upload a face to mark attendance       |
| GET    | `/api/get_attendance` | JSON list of attendance records        |
| GET    | `/api/export_csv`     | Download records as CSV                |
