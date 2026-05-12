# Age Detection System (Pro)

High-level age detection platform with:

- OpenCV age estimation and tracking
- Streamlit dashboard with history and analytics
- FastAPI server with JWT auth and role-based access
- SQLite session persistence
- Docker and Docker Compose support

## 1) Setup

```powershell
Set-Location "D:\My Projects\Age Detection system"
python -m pip install -r requirements.txt
python download_weights.py
Copy-Item .env.example .env
```

Edit `.env` before production usage.

## 2) Run Dashboard

```powershell
$env:STREAMLIT_DASH_PASSWORD="your_dashboard_password"
streamlit run streamlit_app.py
```

## 3) Run API

```powershell
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

### API Auth

- `POST /auth/login` with form fields `username`, `password`
- Roles:
  - `admin`: full access including `/sessions`
  - `viewer`: analyze endpoints only

### Main Endpoints

- `GET /health`
- `GET /v1/features`
- `POST /auth/login`
- `GET /auth/me`
- `GET /sessions` (admin only)
- `GET /v1/sessions` (admin only)
- `GET /v1/sessions/{session_id}` (admin only)
- `GET /v1/overview` (admin/viewer)
- `POST /analyze/image` (admin/viewer)
- `POST /analyze/video` (admin/viewer)
- `POST /v1/analyze/image` (admin/viewer, enriched response)

### Web Frontend

Run the API server, then open:

- `http://localhost:8000/`

This serves a ready-to-use frontend with:

- login UI
- image upload and analysis
- direct live camera analysis in browser
- live interval-based auto analysis
- session history table
- analytics overview KPI cards

## 4) Docker

```powershell
Copy-Item .env.example .env
docker compose up --build
```

Services:

- API: `http://localhost:8000`
- Dashboard: `http://localhost:8501`

















# Age Detection Analytics

Enterprise-grade AI-powered real-time age detection and analytics platform.

Built using:
- OpenCV DNN
- FastAPI
- Streamlit
- SQLite
- JWT Authentication
- Docker
- HTML/CSS/JavaScript

---

# Features

## AI Features

- Real-time face detection
- AI-based age estimation
- Multi-face tracking
- Confidence calibration
- Temporal smoothing
- Analytics aggregation
- Crowd analysis
- Age bucket distribution
- GPU acceleration support

---

# Backend Features

- FastAPI REST API
- JWT Authentication
- Role-Based Access Control (RBAC)
- Session persistence
- Analytics APIs
- Modular architecture
- Docker deployment

---

# Frontend Features

- Live browser camera analysis
- Session dashboard
- Analytics KPIs
- Image upload analysis
- Real-time visualization

---

# Tech Stack

| Layer | Technology |
|---|---|
| AI Engine | OpenCV DNN |
| Backend | FastAPI |
| Dashboard | Streamlit |
| Frontend | HTML/CSS/JS |
| Database | SQLite |
| Deployment | Docker |
| Authentication | JWT |

---

# System Architecture

Frontend → FastAPI → AI Engine → SQLite Storage → Analytics Layer

---

# Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/age-detection-pro.git
cd age-detection-pro
