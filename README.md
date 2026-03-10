# AI Code Review & Rewrite Agent

AI Code Review & Rewrite Agent uses **Google Gemini 2.5 Flash** to deliver real-time bug detection, security checks, optimized code rewriting, **language-to-language code conversion**, and an **AI chatbot** for code Q&A — all via **FastAPI** and a modern web UI — improving code quality and developer speed.

---

## Features

- **Code Review** — Structured, severity-graded feedback (Critical / High / Medium / Low) with line-by-line analysis
- **Code Rewrite** — Automatically rewrites code to production quality with a list of improvements
- **Code Conversion** — Convert code between any of the 16 supported languages with side-by-side Monaco editors, swap button, and conversion notes
- **AI Chat** — Ask questions about your code in a conversational chatbot with context-aware answers, conversation history, and quick-suggestion chips
- **Code History** — Automatically saves all review, rewrite, and conversion operations to a persistent history dropdown (localStorage) with load & delete
- **Chat History** — Sidebar with saved chat sessions that persist across page reloads (localStorage)
- **16 Languages** — Python, JavaScript, TypeScript, Java, C, C++, C#, Go, Rust, Ruby, PHP, Swift, Kotlin, SQL, HTML, CSS
- **Modern Web UI** — Glass-morphism design with dark/light theme toggle and Monaco code editor with 140+ syntax-coloring rules
- **Onboarding Flow** — "How It Works" page shown on first sign-in with a "Get Started" button
- **Login Page** — Clean authentication interface with session management
- **Interactive API Docs** — Auto-generated Swagger UI at `/docs`
- **Retry Logic** — Built-in rate-limit handling with 5× retry and exponential backoff (10s–40s) for Gemini API calls
- **Keyboard Shortcuts** — `Ctrl+Enter` to run review/rewrite/convert, `Enter` to send chat messages

---

## Tech Stack

| Layer      | Technology                                        |
|------------|---------------------------------------------------|
| Backend    | Python, FastAPI, Uvicorn, Pydantic                |
| AI Model   | Google Gemini 2.5 Flash (`google-genai`)          |
| Frontend   | HTML, CSS, JavaScript, Monaco Editor 0.45.0       |
| Libraries  | Highlight.js 11.9.0, Marked.js 12.0.1, Font Awesome 6.5.1 |
| Storage    | localStorage (code history, chat history), sessionStorage (auth) |
| Fonts      | Inter, JetBrains Mono (Google Fonts)              |

---

## Project Structure

```
AI-Code-Review-Rewrite-Agent/
├── backend/
│   ├── main.py            # FastAPI server, Gemini integration, all API routes
│   ├── requirements.txt   # Python dependencies
│   ├── .env               # API key (not committed)
│   └── __init__.py
├── frontend/
│   ├── index.html         # Main app UI (review, rewrite, convert, chat, how-it-works tabs)
│   └── login.html         # Login page
├── .gitignore
└── README.md
```

---

## Getting Started

### Prerequisites

- **Python 3.10+**
- A **Google Gemini API key** — get one at [Google AI Studio](https://aistudio.google.com/app/apikey)

### 1. Clone the Repository

```bash
git clone https://github.com/Nagamanikanta2331/AI-Code-Review-Rewrite-Agent.git
cd AI-Code-Review-Rewrite-Agent
```

### 2. Install Dependencies

```bash
pip install -r backend/requirements.txt
```

### 3. Configure Environment Variables

Create a `backend/.env` file:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 4. Run the Server

```bash
cd backend
python main.py
```

The server starts at **http://localhost:8000**.

---

## Usage

| URL                          | Description              |
|------------------------------|--------------------------|
| http://localhost:8000/login   | Login page               |
| http://localhost:8000/app     | Main application UI      |
| http://localhost:8000/docs    | Swagger API documentation|

### API Endpoints

| Method | Endpoint        | Description                                      |
|--------|-----------------|--------------------------------------------------|
| GET    | `/`             | Health check & API status                        |
| POST   | `/review`       | Review code (returns severity-graded markdown)   |
| POST   | `/api/rewrite`  | Rewrite code (returns structured JSON)           |
| POST   | `/api/convert`  | Convert code between languages (returns JSON)    |
| POST   | `/api/chat`     | Chat about code (context-aware, with history)    |
| GET    | `/login`        | Serve login page                                 |
| GET    | `/app`          | Serve main application page                      |

### Example — Code Review Request

```bash
curl -X POST http://localhost:8000/review \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def add(a, b):\n    return a - b",
    "language": "python"
  }'
```

### Example — Code Rewrite Request

```bash
curl -X POST http://localhost:8000/api/rewrite \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def add(a, b):\n    return a - b",
    "language": "python"
  }'
```

### Example — Code Conversion Request

```bash
curl -X POST http://localhost:8000/api/convert \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def add(a, b):\n    return a + b\n\nprint(add(5, 7))",
    "from_language": "python",
    "to_language": "c"
  }'
```

### Example — Chat Request

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What does this function do?",
    "code": "def add(a, b):\n    return a + b",
    "language": "python",
    "history": []
  }'
```

---

## App Tabs

| Tab              | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| **Code Review**  | Paste code, select language & focus areas, get severity-ranked feedback      |
| **Rewrite Code** | Side-by-side Monaco editors — input your code and get the improved version  |
| **Convert Code** | Convert code between languages with source/target selectors and swap button |
| **How It Works** | Overview of the architecture, features, and tech stack (shown on first login)|
| **Chat**         | AI chatbot that answers questions about your code with conversation memory   |

---

## Gemini API Rate Limits (Free Tier)

| Limit                    | Value       |
|--------------------------|-------------|
| Requests per minute      | 10 RPM      |
| Tokens per minute        | 250,000 TPM |
| Requests per day         | 500 RPD     |

The backend automatically retries rate-limited requests up to 5 times with exponential backoff. If you need higher limits, upgrade to the [paid Gemini API tier](https://ai.google.dev/pricing).

---

## Screenshots

### Dark Theme
> Modern glass-morphism UI with Monaco code editor, severity-based review output, AI chat, code conversion, and one-click code rewriting.

### Light Theme
> Full light-mode support with smooth theme transitions.

---

## License

This project is open-source and available under the [MIT License](LICENSE).
