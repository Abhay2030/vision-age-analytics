import os
import tempfile
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional

import cv2
import numpy as np
import jwt
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse

from main import AgeDetectionSystem
from storage import aggregate_overview, get_session, list_sessions, save_session

app = FastAPI(title="Age Detection API", version="1.0.0")
security = HTTPBearer()
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"

JWT_SECRET = os.getenv("API_JWT_SECRET", "change-this-in-production")
JWT_ALG = "HS256"
ACCESS_TOKEN_MINUTES = int(os.getenv("API_ACCESS_TOKEN_MINUTES", "120"))

# Configure users via env vars for quick setup.
# API_ADMIN_PASSWORD and API_VIEWER_PASSWORD should be set in production.
API_USERS = {
    "admin": {
        "password": os.getenv("API_ADMIN_PASSWORD", "admin123"),
        "role": "admin",
    },
    "viewer": {
        "password": os.getenv("API_VIEWER_PASSWORD", "viewer123"),
        "role": "viewer",
    },
}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def make_detector(
    use_gpu: bool = False,
    conf_threshold: float = 0.55,
    min_face: int = 70,
    smooth_alpha: float = 0.30,
    track_distance: float = 90.0,
    weights_dir: Optional[str] = None,
) -> AgeDetectionSystem:
    return AgeDetectionSystem(
        weights_dir=weights_dir,
        use_gpu=use_gpu,
        confidence_threshold=conf_threshold,
        min_face_size=min_face,
        smooth_alpha=smooth_alpha,
        track_distance=track_distance,
    )


def create_access_token(username: str, role: str) -> str:
    expires = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_MINUTES)
    payload = {
        "sub": username,
        "role": role,
        "exp": expires,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
    except jwt.PyJWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {e}",
        ) from e
    username = str(payload.get("sub", ""))
    role = str(payload.get("role", ""))
    if not username or role not in {"admin", "viewer"}:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload.")
    return {"username": username, "role": role}


def require_role(*allowed_roles: str):
    def checker(user: dict = Depends(get_current_user)) -> dict:
        if user["role"] not in set(allowed_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{user['role']}' is not allowed for this endpoint.",
            )
        return user

    return checker


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/v1/features")
def features() -> dict:
    return {
        "app": "Age Detection Pro",
        "features": [
            "JWT auth with RBAC",
            "Image and video analysis",
            "Quality-aware confidence calibration",
            "Session persistence and history",
            "Interactive web dashboard",
            "Direct browser camera analysis",
            "Session analytics overview",
        ],
    }


@app.post("/auth/login")
async def login(
    request: Request,
) -> dict:
    # Accept login credentials from either JSON or form-urlencoded.
    username = ""
    password = ""
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        payload = await request.json()
        username = str(payload.get("username", "")).strip()
        password = str(payload.get("password", "")).strip()
    else:
        form = await request.form()
        username = str(form.get("username", "")).strip()
        password = str(form.get("password", "")).strip()

    if not username or not password:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="username and password are required.")

    user = API_USERS.get(username)
    if not user or user["password"] != password:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password.")
    token = create_access_token(username=username, role=user["role"])
    return {
        "access_token": token,
        "token_type": "bearer",
        "role": user["role"],
        "expires_in_minutes": ACCESS_TOKEN_MINUTES,
    }


@app.get("/auth/me")
def me(user: dict = Depends(get_current_user)) -> dict:
    return {"username": user["username"], "role": user["role"]}


@app.get("/sessions")
def sessions(limit: int = 20, _: dict = Depends(require_role("admin"))) -> dict:
    return {"items": list_sessions(limit=max(1, min(100, limit)))}


@app.get("/v1/sessions")
def sessions_v1(limit: int = 20, user: dict = Depends(require_role("admin"))) -> dict:
    return {"items": list_sessions(limit=max(1, min(100, limit))), "requested_by": user["username"]}


@app.get("/v1/sessions/{session_id}")
def session_detail(session_id: int, user: dict = Depends(require_role("admin"))) -> dict:
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"item": session, "requested_by": user["username"]}


@app.get("/v1/overview")
def overview(limit: int = 200, _: dict = Depends(require_role("admin", "viewer"))) -> dict:
    return {"overview": aggregate_overview(limit=max(1, min(1000, limit)))}


@app.post("/analyze/image")
async def analyze_image(
    file: UploadFile = File(...),
    use_gpu: bool = Form(False),
    conf_threshold: float = Form(0.55),
    min_face: int = Form(70),
    smooth_alpha: float = Form(0.30),
    track_distance: float = Form(90.0),
    _: dict = Depends(require_role("admin", "viewer")),
) -> JSONResponse:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Unsupported image format.")

    detector = make_detector(
        use_gpu=use_gpu,
        conf_threshold=conf_threshold,
        min_face=min_face,
        smooth_alpha=smooth_alpha,
        track_distance=track_distance,
    )
    _, detections = detector.annotate_frame(frame)
    summary = detector.analytics.summary_dict()
    session_id = save_session(summary, source_type="api_image", source_label=file.filename or "")

    return JSONResponse(
        {
            "session_id": session_id,
            "summary": summary,
            "detections": detections,
        }
    )


@app.post("/analyze/video")
async def analyze_video(
    file: UploadFile = File(...),
    max_frames: int = Form(300),
    _: dict = Depends(require_role("admin", "viewer")),
) -> JSONResponse:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(raw)
        video_path = tmp.name

    detector = make_detector()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        Path(video_path).unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Unable to decode video.")

    processed = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        detector.annotate_frame(frame)
        processed += 1
        if max_frames > 0 and processed >= max_frames:
            break
    cap.release()
    Path(video_path).unlink(missing_ok=True)

    summary = detector.analytics.summary_dict()
    session_id = save_session(summary, source_type="api_video", source_label=file.filename or "")
    return JSONResponse({"session_id": session_id, "summary": summary, "frames_used": processed})


@app.post("/v1/analyze/image")
async def analyze_image_v1(
    file: UploadFile = File(...),
    use_gpu: bool = Form(False),
    conf_threshold: float = Form(0.55),
    min_face: int = Form(70),
    smooth_alpha: float = Form(0.30),
    track_distance: float = Form(90.0),
    user: dict = Depends(require_role("admin", "viewer")),
) -> JSONResponse:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Unsupported image format.")

    detector = make_detector(
        use_gpu=use_gpu,
        conf_threshold=conf_threshold,
        min_face=min_face,
        smooth_alpha=smooth_alpha,
        track_distance=track_distance,
    )
    _, detections = detector.annotate_frame(frame)
    summary = detector.analytics.summary_dict()
    session_id = save_session(summary, source_type="api_image_v1", source_label=file.filename or "")
    return JSONResponse(
        {
            "session_id": session_id,
            "summary": summary,
            "detections": detections,
            "model_info": {
                "backend": "OpenCV DNN + Caffe",
                "quality_calibrated": True,
                "tta_horizontal_flip": True,
            },
            "requested_by": user["username"],
        }
    )


if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")


@app.get("/")
def ui_root() -> FileResponse:
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Frontend not found.")
