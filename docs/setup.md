# Setup & Installation Guide - ML RAG Agent

## Prerequisites

- **Python**: Version 3.12+ installed.
- **Git**: Installed for version control.
- **Google Gemini API Key**: Obtainable from [Google AI Studio](https://aistudio.google.com/).

---

## Environment Setup

### 1. Clone Repository & Navigate to Directory
```bash
git clone https://github.com/your-username/ml-rag-agent.git
cd ml-rag-agent
```

### 2. Create Virtual Environment
On Linux/macOS:
```bash
python3.12 -m venv venv
source venv/bin/activate
```

On Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Configuration

Copy `.env.example` to create your active `.env` file:
```bash
cp .env.example .env
```

Edit `.env` and set your Google Gemini API Key:
```env
GEMINI_API_KEY="AIzaSyYourActualGeminiApiKeyHere"
GEMINI_MODEL_NAME="gemini-1.5-flash"
LOG_LEVEL="INFO"
```

---

## Running the Application

### 1. Ingest Machine Learning Documents
Place your raw PDF papers, Scikit-Learn documentation, or KTU notes inside `data/raw/`, then run:

Via CLI:
```bash
python app/cli.py ingest --path data/raw
```

Via Helper Script:
```bash
python scripts/run_ingestion.py
```

### 2. Launch FastAPI Server
```bash
uvicorn api.main:app --reload --port 8000
```
Or run directly:
```bash
python api/main.py
```

### 3. Interactive OpenAPI Documentation
Open your browser and navigate to:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## Testing

Run unit and integration test suite:
```bash
pytest -v
```
