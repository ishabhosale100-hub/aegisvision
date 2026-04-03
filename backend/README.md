# AegisVision – DualShield Interface (OpenEnv Edition)
> AI-powered women's digital safety platform + OpenEnv training environment

---

## 🗂️ Project Structure

```
aegisvision/
├── backend/
│   ├── env.py          # AegisVisionEnv (OpenEnv class)
│   ├── main.py         # FastAPI app with all endpoints
│   └── requirements.txt
│
└── frontend/
    └── index.html      # Complete React SPA (CDN, no build step)
```

---

## ⚙️ Backend Setup & Run

### 1. Install Python dependencies

```bash
cd aegisvision/backend
pip install -r requirements.txt
```

### 2. Start the FastAPI server

```bash
python main.py
```

Or with uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Server runs at: **http://localhost:8000**
API docs (Swagger): **http://localhost:8000/docs**

---

## 🌐 Frontend Setup & Run

No build step required — pure HTML + React via CDN.

### Option A: Direct browser open
```bash
open aegisvision/frontend/index.html
```

### Option B: Serve with Python (recommended for CORS)
```bash
cd aegisvision/frontend
python -m http.server 3000
```
Then visit: **http://localhost:3000**

### Option C: VS Code Live Server
Right-click `index.html` → Open with Live Server

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/reset` | Reset OpenEnv, generate new scenario |
| GET | `/state` | Get current environment state |
| POST | `/step/{action}` | Execute action (5 actions available) |
| POST | `/protect` | Simulate watermark protection (JSON) |
| POST | `/protect/upload` | Protect uploaded image file |
| POST | `/compare` | Simulate forensic comparison (JSON) |
| POST | `/compare/upload` | Compare two uploaded image files |
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI |

### Valid Actions for `/step/{action}`:
- `MARK_REAL` — Mark image as genuine (+10 / -5)
- `MARK_FAKE` — Mark image as manipulated (+10 / -5)
- `APPLY_PROTECTION` — Apply watermark protection (+3)
- `GENERATE_REPORT` — Generate forensic report (+3)
- `IGNORE` — Ignore scenario (-10 / 0)

---

## 🎮 Usage Guide

### Login
- Any username + password (demo mode)
- Animated "Initializing Security System..." sequence

### PRE-SHIELD
1. Upload an image
2. Click "APPLY INVISIBLE WATERMARK"
3. View: certificate ID, method, risk reduction %, watermark strength

### POST-SHIELD
1. Upload original image
2. Upload suspected/modified image
3. Click "RUN FORENSIC COMPARISON"
4. View: manipulation score, confidence meter, flagged regions heatmap

### ACTION CENTRE
1. Complete the 3-step legal action checklist
2. Download forensic report as `.txt`
3. Copy complaint text for submission
4. Access helplines: 1930 (Cyber Crime) / 112 (Emergency)

### GAMIFICATION (OpenEnv)
1. Click "INITIALIZE ENV" → calls `GET /reset`
2. Review scenario: image type, platform, risk level, manipulation hints
3. Select action from 5 options
4. Click "EXECUTE ACTION" → calls `POST /step/{action}`
5. Watch: score, threat level, confidence meter update in real-time
6. Console shows detailed step-by-step feedback

---

## 🔌 Offline/Demo Mode

The frontend **auto-simulates** backend responses if the backend is offline.
All features work in demo mode — no backend required for UI testing.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.10+, FastAPI, Uvicorn |
| Frontend | React 18 (CDN), Babel standalone |
| AI Env | Custom OpenEnv-compatible class |
| Styling | Pure CSS (glassmorphism, neon, dark) |
| Fonts | Orbitron, Share Tech Mono, Exo 2 |

---

## 🎨 Design System

- **Theme**: Dark void (black/navy)
- **Accents**: Neon blue `#00d4ff`, Purple `#b200ff`, Cyan `#00ffe7`
- **Effects**: Glassmorphism, glowing borders, particle field, scanlines, CSS grid
- **Typography**: Orbitron (display), Share Tech Mono (data), Exo 2 (body)
- **Animations**: Fade/slide reveals, orbiting glows, confidence meter, threat banners

---

## 🛡️ Safety & Purpose

AegisVision is designed to:
1. **Protect** women's images before sharing (pre-shield watermarking)
2. **Detect** manipulation after the fact (post-shield forensics)
3. **Empower** victims with legal action steps and helplines
4. **Train** AI models via the OpenEnv reinforcement learning interface

---

*AegisVision DualShield Interface © 2025 — Shielding truth in every pixel*

