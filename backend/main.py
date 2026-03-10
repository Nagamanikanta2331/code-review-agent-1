"""
AI Code Review & Rewrite Agent — Backend API
=============================================
Production-grade FastAPI server with Google Gemini API integration.
Model: Gemini 2.5 Flash | Inference: Google AI
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ───────────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────────
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
MODEL_ID: str = "gemini-2.5-flash"
TEMPERATURE: float = 0.3
MAX_TOKENS: int = 8192
TOP_P: float = 0.9

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

if not GEMINI_API_KEY:
    print("⚠️  GEMINI_API_KEY is missing — set it in backend/.env")

# ───────────────────────────────────────────────────────────────
# Gemini Client
# ───────────────────────────────────────────────────────────────
gemini_client: Optional[genai.Client] = None
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ───────────────────────────────────────────────────────────────
# FastAPI Application
# ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Code Review & Rewrite Agent",
    description="Reviews, rewrites and improves code with Google Gemini 2.5 Flash.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ───────────────────────────────────────────────────────────────
# Pydantic Models
# ───────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    model: str
    gemini_connected: bool


class ReviewRequest(BaseModel):
    code: str = Field(..., min_length=1)
    language: str = Field(default="python")
    instructions: Optional[str] = Field(default=None)


class ReviewResponse(BaseModel):
    result: str
    model: str
    tokens_used: Optional[int] = None


class RewriteRequest(BaseModel):
    code: str = Field(..., min_length=1)
    language: str = Field(default="python")


class RewriteResponse(BaseModel):
    rewritten_code: str
    improvements: list[str]
    model: str
    tokens_used: Optional[int] = None


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    code: str = Field(default="")
    language: str = Field(default="python")
    history: list[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str
    model: str
    tokens_used: Optional[int] = None


class ConvertRequest(BaseModel):
    code: str = Field(..., min_length=1)
    from_language: str = Field(default="python")
    to_language: str = Field(default="javascript")


class ConvertResponse(BaseModel):
    converted_code: str
    notes: list[str]
    model: str
    tokens_used: Optional[int] = None


# ───────────────────────────────────────────────────────────────
# Prompt Engineering
# ───────────────────────────────────────────────────────────────
REVIEW_SYSTEM_PROMPT = """
You are a senior software engineer performing a clear, production-grade code review.

GOAL:
Provide a structured, concise, and easy-to-understand review.
Avoid long paragraphs. No unnecessary explanations. Focus only on real issues.

REVIEW CHECKLIST:
- Bugs & Logic Errors
- Security Risks
- Performance Issues
- Error Handling Problems
- Readability & Best Practices

SEVERITY LEVELS:
🔴 Critical  → Crashes, security risks, data loss
🟠 High      → Logic errors, unhandled exceptions, major flaws
🟡 Medium    → Code smells, missing validation, improvement areas
🔵 Low       → Style or minor readability improvements

RULES:
1. Keep each issue explanation under 3 lines.
2. Be specific — mention exact variable names and line numbers.
3. Do NOT repeat obvious things.
4. Do NOT over-explain.
5. Only include real, meaningful issues.

OUTPUT FORMAT:

══════════════════════════════
📊 REVIEW SUMMARY
══════════════════════════════
🔴 Critical : X
🟠 High     : X
🟡 Medium   : X
🔵 Low      : X
Total Issues: X

Progress:
Critical  : 🔴🔴 (X)
High      : 🟠🟠🟠 (X)
Medium    : 🟡 (X)
Low       : 🔵 (X)

══════════════════════════════
🔍 DETAILED FINDINGS
══════════════════════════════

### 🔴 [Short Issue Title]
Line(s): X
Problem: Short explanation.
Fix:
```python
corrected code snippet"""


REWRITE_SYSTEM_PROMPT = """
You are a senior software engineer rewriting code clearly and minimally.

GOAL:
Fix and improve the code while keeping it simple and proportional to the original size.

CORE RULE:
If the input code is small, keep the output small.
Do NOT over-engineer.

YOU MUST:
- Fix bugs and logical errors.
- Improve clarity and readability.
- Preserve the original intent.
- Keep structure similar unless necessary to change.
- Add minimal comments only if truly helpful.

YOU MUST NOT:
- Add unnecessary functions, classes, or abstractions.
- Add example usage unless explicitly requested.
- Add excessive comments or documentation.
- Introduce new features not present in the original code.
- Make the code significantly longer without strong reason.

OUTPUT FORMAT:
You MUST return a valid JSON object with exactly two keys:
- "rewritten_code": a string containing the clean corrected code
- "improvements": an array of strings, each a short bullet explaining an important fix (max 5)

Example:
{"rewritten_code": "print('Hello, World!')", "improvements": ["No changes needed."]}

Do NOT include markdown, code fences, or any text outside the JSON object.
"""

CHAT_SYSTEM_PROMPT = """
You are a helpful AI coding assistant embedded in a code review tool.
The user may provide a code snippet for context. Answer questions about the code clearly and concisely.

RULES:
1. If code context is provided, refer to it when answering.
2. Provide short, accurate, and actionable answers.
3. Use code snippets in your answers when helpful (wrap them in markdown fenced code blocks).
4. If the question is unrelated to programming, politely redirect the user.
5. Be friendly but professional.
6. Format your response with Markdown for readability.
"""

CONVERT_SYSTEM_PROMPT = """
You are a world-class polyglot compiler engineer. Your SOLE job: convert source code from one language to another so it COMPILES AND RUNS PERFECTLY with ZERO errors on the FIRST attempt.

██ ABSOLUTE RULES — VIOLATING ANY OF THESE IS A FAILURE ██

1. OUTPUT MUST COMPILE/RUN AS-IS — no missing imports, no missing headers, no missing entry points, no syntax errors, no type errors. ZERO tolerance for errors.
2. OUTPUT MUST PRODUCE IDENTICAL BEHAVIOR — same console output, same return values, same side effects.
3. WRITE IDIOMATIC CODE — use the target language's natural patterns, not a line-by-line transliteration.
4. INCLUDE EVERYTHING — all imports, includes, using statements, package declarations, class wrappers, main functions, and entry points that the target language requires.
5. DO NOT ADD EXTRA FUNCTIONALITY — no extra prints, no extra comments explaining basic syntax, no demo code, no test code.

██ MANDATORY SELF-CHECK — DO THIS BEFORE OUTPUTTING ██

Mentally compile/interpret your output line by line:
✓ Are ALL required imports/includes/using statements present?
✓ Does every function have correct return type and parameter types?
✓ Are all variables declared with correct types before use?
✓ Are string operations using the TARGET language's syntax (not the source)?
✓ Is there a proper entry point (main function/method) if the target language requires one?
✓ Would this code produce the EXACT same output when run?
✓ Are there any semicolons missing (if required by target language)?
✓ Are braces/brackets/parentheses all properly matched?

██ LANGUAGE-SPECIFIC REQUIREMENTS (MUST FOLLOW) ██

C:
  - Headers: #include <stdio.h>, #include <stdlib.h>, #include <string.h> as needed
  - I/O: printf() with correct format specifiers (%d, %ld, %f, %s, %c)
  - Strings: char arrays or char*; use snprintf() for formatting, strcat/strcpy for manipulation
  - Entry: int main(void) { ... return 0; } or int main(int argc, char *argv[]) { ... }
  - Functions: explicit return types and parameter types ALWAYS
  - No std::string, no cout, no C++ features whatsoever

C++:
  - Headers: #include <iostream>, #include <string>, #include <vector>, #include <sstream> etc.
  - I/O: std::cout << ... << '\\n';  and  std::cin >>
  - Strings: std::string ONLY (never C-style char* for string manipulation)
  - String concat: + operator or std::ostringstream
  - Number to string: std::to_string()
  - Pass strings by const ref: const std::string&
  - Entry: int main() { ... return 0; }
  - NEVER mix C I/O (printf) with C++ I/O (cout) unless necessary

Java:
  - Structure: public class Main { public static void main(String[] args) { ... } }
  - I/O: System.out.println(), System.out.print(), System.out.printf()
  - Strings: String class, + for concat, String.format() for formatting
  - Every method needs explicit return type and access modifier
  - Static methods if called from main without object instantiation
  - Arrays: int[] arr = new int[]{...}; or ArrayList<Integer>

C#:
  - Using: using System; (and using System.Collections.Generic; etc. as needed)
  - Structure: class Program { static void Main(string[] args) { ... } }
  - I/O: Console.WriteLine(), Console.Write()
  - String interpolation: $"Hello, {name}!"
  - Types: int, string, double, bool (lowercase keywords)
  - NEVER use #include — that is C/C++, NOT C#

Python:
  - Use f-strings: f"Hello, {name}!"
  - Functions: def func_name(param): with snake_case
  - I/O: print()
  - No semicolons, proper 4-space indentation
  - if __name__ == "__main__": guard if appropriate

JavaScript:
  - Variables: const/let (NEVER var)
  - Template literals: `Hello, ${name}!`
  - I/O: console.log()
  - Functions: arrow functions or function declarations
  - Modern ES6+ syntax

TypeScript:
  - Same as JavaScript + explicit type annotations on all functions and variables
  - Use interfaces/types for complex structures

Go:
  - Package: package main
  - Imports: import "fmt" (or import block for multiple)
  - Entry: func main() { ... }
  - Variables: := for short declaration, var for explicit
  - I/O: fmt.Println(), fmt.Printf(), fmt.Sprintf()
  - Exported = Uppercase first letter; unexported = lowercase
  - No semicolons, no unused imports, no unused variables (compiler errors!)

Rust:
  - Entry: fn main() { ... }
  - I/O: println!("format {}", var); with {} placeholders
  - Variables: let (immutable), let mut (mutable)
  - Strings: String::from(), &str, .to_string()
  - Proper ownership and borrowing

PHP:
  - Start with <?php
  - Variables: $varName ($ prefix mandatory)
  - I/O: echo "text\\n"; or print
  - String concat: . operator
  - String interpolation: "Hello, $name!" (double quotes only)
  - Functions: function funcName($param) { ... }

Ruby:
  - I/O: puts (with newline), print (without)
  - Functions: def method_name ... end
  - String interpolation: "Hello, #{name}!" (double quotes only)
  - snake_case for methods and variables

Swift:
  - I/O: print("text")
  - Variables: let (constant), var (mutable)
  - String interpolation: "Hello, \\(name)!"
  - Functions: func name(param: Type) -> ReturnType { ... }
  - No main() needed for scripts; use @main for app entry

Kotlin:
  - Entry: fun main() { ... }  or  fun main(args: Array<String>) { ... }
  - I/O: println("text"), print("text")
  - String templates: "Hello, $name!" or "Result: ${expr}"
  - Variables: val (immutable), var (mutable)
  - Functions: fun name(param: Type): ReturnType { ... }

██ OUTPUT FORMAT ██

Return a valid JSON object with exactly two keys:
- "converted_code": the COMPLETE, READY-TO-RUN code (no placeholders, no TODOs, no "...")
- "notes": array of 1-5 short notes about key conversion decisions

Do NOT include markdown, code fences, or any text outside the JSON object.
"""

CONVERT_VERIFY_PROMPT = """
You are a compiler-level code verifier. You will receive:
1. Original source code in one language
2. Converted code in the target language

Your job: check if the converted code will COMPILE AND RUN WITHOUT ERRORS in the target language.

CHECK EVERY SINGLE ONE OF THESE:
- All imports/includes/using statements present and correct for the target language?
- Entry point (main function/method) present if required?
- All types correctly declared?
- All syntax correct for the TARGET language (not accidentally using source language syntax)?
- All string operations use target language syntax?
- All I/O operations use target language functions?
- Braces, brackets, parentheses all matched?
- Semicolons present where required?
- No mixing of language features (e.g., #include in C#, printf in Java, std::cout in C)?
- Will it produce the same output as the original?

If the code is PERFECT: return it unchanged.
If there are ANY issues: fix ALL of them and return the corrected version.

Return a valid JSON object with exactly two keys:
- "verified_code": the final, verified, ready-to-run code
- "fixes_applied": array of strings describing any fixes (empty array [] if code was already perfect)

Do NOT include markdown, code fences, or any text outside the JSON object.
"""

# ───────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────
def _ensure_gemini() -> None:
    """Raise 503 if Gemini client is not initialised."""
    if gemini_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gemini API key is not configured. Set GEMINI_API_KEY in backend/.env",
        )


def _call_llm(system: str, user: str, json_mode: bool = False, temperature: Optional[float] = None) -> tuple[str, Optional[int]]:
    """Call Google Gemini with retry logic and return (content, total_tokens)."""
    _ensure_gemini()

    temp = temperature if temperature is not None else TEMPERATURE

    max_retries = 3
    for attempt in range(max_retries):
        try:
            config = types.GenerateContentConfig(
                system_instruction=system,
                temperature=temp,
                max_output_tokens=MAX_TOKENS,
                top_p=TOP_P,
            )
            if json_mode:
                config = types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=temp,
                    max_output_tokens=MAX_TOKENS,
                    top_p=TOP_P,
                    response_mime_type="application/json",
                )

            response = gemini_client.models.generate_content(
                model=MODEL_ID,
                contents=user,
                config=config,
            )
            content = response.text
            tokens = None
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                tokens = response.usage_metadata.total_token_count
            return content, tokens
        except Exception as exc:
            error_msg = str(exc).lower()
            if "rate" in error_msg or "quota" in error_msg or "429" in error_msg:
                if attempt < max_retries - 1:
                    wait = (attempt + 1) * 3  # 3s, 6s — fast fail instead of long waits
                    print(f"[Gemini] Rate limited (attempt {attempt+1}/{max_retries}). Retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                raise HTTPException(
                    status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit reached. The free Gemini tier allows ~10 requests/minute. Please wait 30-60 seconds and try again.",
                )
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail=f"Gemini API error: {str(exc)}")

    raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail="Gemini API failed after retries.")


def extract_json(raw: str) -> dict:
    """Robust JSON extraction from LLM output with multi-strategy parsing."""
    if not raw or not isinstance(raw, str):
        raise ValueError("Empty or invalid response from model")

    # Strategy 1 — direct parse
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass

    # Strategy 2 — strip markdown code fences
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned.strip(), flags=re.MULTILINE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 3 — extract first JSON object (use greedy to get the outermost {})
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            # Try fixing common escape issues in the matched JSON
            fixed = match.group()
            # Fix unescaped newlines inside string values
            fixed = re.sub(r'(?<=": ")([\s\S]*?)(?="[,\s*}])', lambda m: m.group(0).replace('\n', '\\n'), fixed)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

    # Strategy 4 — extract converted_code and notes (for /api/convert responses)
    code_match = re.search(r'"converted_code"\s*:\s*"([\s\S]*?)"(?:\s*,|\s*})', raw)
    notes_match = re.search(r'"notes"\s*:\s*\[([\s\S]*?)\]', raw)
    if code_match:
        code_val = code_match.group(1)
        notes = []
        if notes_match:
            notes = re.findall(r'"([^"]*)"', notes_match.group(1))
        return {"converted_code": code_val, "notes": notes}

    # Strategy 5 — extract rewritten_code and improvements (for /api/rewrite responses)
    rw_match = re.search(r'"rewritten_code"\s*:\s*"([\s\S]*?)"(?:\s*,\s*"improvements")', raw)
    imp_match = re.search(r'"improvements"\s*:\s*\[([\s\S]*?)\]', raw)
    if rw_match:
        code_val = rw_match.group(1)
        improvements = []
        if imp_match:
            improvements = re.findall(r'"([^"]*)"', imp_match.group(1))
        return {"rewritten_code": code_val, "improvements": improvements}

    # Strategy 6 — last resort: extract any key-value pairs that look like our expected fields
    # Try to find converted_code between triple-backticks or large string blocks
    code_block = re.search(r'```\w*\n([\s\S]*?)```', raw)
    if code_block:
        return {"converted_code": code_block.group(1).strip(), "notes": ["Extracted from non-JSON model response"]}

    raise ValueError("Failed to parse JSON from model response")


def _serve_html(filename: str) -> FileResponse:
    """Return an HTML file from the frontend directory or raise 404."""
    path = FRONTEND_DIR / filename
    if not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"{filename} not found at {path}")
    return FileResponse(path, media_type="text/html")


# ───────────────────────────────────────────────────────────────
# Routes — Health
# ───────────────────────────────────────────────────────────────
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Return empty favicon to suppress 404."""
    from fastapi.responses import Response
    # 1x1 transparent PNG favicon
    import base64
    ico = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )
    return Response(content=ico, media_type="image/png")


@app.get("/", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health-check endpoint."""
    return HealthResponse(
        status="API Connected",
        model=MODEL_ID,
        gemini_connected=gemini_client is not None,
    )


# ───────────────────────────────────────────────────────────────
# Routes — Code Review
# ───────────────────────────────────────────────────────────────
@app.post("/review", response_model=ReviewResponse, tags=["Code Review"])
async def review_code(req: ReviewRequest):
    """Review code with structured severity categories and line-by-line feedback."""
    focus = f"\n\nAdditional focus areas: {req.instructions}" if req.instructions else ""
    user_prompt = f"Language: {req.language}\n\n```{req.language}\n{req.code}\n```{focus}"

    content, tokens = _call_llm(REVIEW_SYSTEM_PROMPT, user_prompt)

    return ReviewResponse(result=content, model=MODEL_ID, tokens_used=tokens)


# ───────────────────────────────────────────────────────────────
# Routes — Code Rewrite (structured JSON)
# ───────────────────────────────────────────────────────────────
@app.post("/api/rewrite", response_model=RewriteResponse, tags=["Code Rewrite"])
async def rewrite_code(req: RewriteRequest):
    """Rewrite code to production quality — returns structured JSON."""
    user_prompt = f"Language: {req.language}\n\n```{req.language}\n{req.code}\n```"

    content, tokens = _call_llm(REWRITE_SYSTEM_PROMPT, user_prompt, json_mode=True)

    try:
        parsed = extract_json(content)
    except Exception as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail=f"Model returned invalid response. Please retry. ({exc})")

    rewritten = parsed.get("rewritten_code", "")
    improvements = parsed.get("improvements", [])

    if not rewritten:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail="Model did not return rewritten code.")

    return RewriteResponse(
        rewritten_code=rewritten,
        improvements=improvements if isinstance(improvements, list) else [str(improvements)],
        model=MODEL_ID,
        tokens_used=tokens,
    )


# ───────────────────────────────────────────────────────────────
# Routes — Chat
# ───────────────────────────────────────────────────────────────
@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(req: ChatRequest):
    """Chat with the AI about code — supports conversation history."""
    # Build user prompt with optional code context
    parts = []
    if req.code.strip():
        parts.append(f"Code context ({req.language}):\n```{req.language}\n{req.code}\n```\n")

    # Append conversation history
    for msg in req.history[-10:]:  # Keep last 10 messages for context
        parts.append(f"{msg.role}: {msg.content}")

    parts.append(f"user: {req.message}")
    user_prompt = "\n\n".join(parts)

    content, tokens = _call_llm(CHAT_SYSTEM_PROMPT, user_prompt)

    return ChatResponse(reply=content or "Sorry, I couldn't generate a response.", model=MODEL_ID, tokens_used=tokens)


# ───────────────────────────────────────────────────────────────
# Routes — Code Conversion
# ───────────────────────────────────────────────────────────────
@app.post("/api/convert", response_model=ConvertResponse, tags=["Code Conversion"])
async def convert_code(req: ConvertRequest):
    """Convert code from one programming language to another with verification."""
    from_lang = req.from_language.strip().lower()
    to_lang = req.to_language.strip().lower()

    user_prompt = (
        f"Convert the following {from_lang} code to {to_lang}.\n\n"
        f"SOURCE LANGUAGE: {from_lang}\n"
        f"TARGET LANGUAGE: {to_lang}\n\n"
        f"CRITICAL — READ CAREFULLY:\n"
        f"- Output MUST be valid, compilable/runnable {to_lang} code with ZERO errors.\n"
        f"- Use idiomatic {to_lang} patterns — NOT a line-by-line transliteration.\n"
        f"- Include ALL required imports, headers, using statements, entry points, type declarations.\n"
        f"- The code MUST produce the EXACT SAME output as the original when run.\n"
        f"- Do NOT use syntax from {from_lang} in the {to_lang} output.\n"
        f"- Do NOT add excessive comments explaining basic {to_lang} syntax.\n"
        f"- Mentally compile and run your output before returning it.\n\n"
        f"Source code:\n```{from_lang}\n{req.code}\n```"
    )

    # ── Step 1: Generate initial conversion ──
    last_error = None
    converted = ""
    notes = []
    total_tokens = 0

    for attempt in range(2):
        content, tokens = _call_llm(CONVERT_SYSTEM_PROMPT, user_prompt, json_mode=True, temperature=0.1)
        total_tokens += tokens or 0

        try:
            parsed = extract_json(content)
        except Exception as exc:
            print(f"[Convert] JSON parse failed (attempt {attempt+1}). Raw:\n{content[:500]}")
            last_error = exc
            continue

        converted = parsed.get("converted_code", "")
        notes = parsed.get("notes", [])

        if not converted:
            print(f"[Convert] Empty converted_code (attempt {attempt+1}).")
            last_error = ValueError("Model did not return converted code.")
            continue
        break
    else:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            detail=f"Model returned invalid response after 2 attempts. ({last_error})",
        )

    # ── Step 2: Verification pass — ask LLM to check & fix the converted code ──
    try:
        verify_prompt = (
            f"ORIGINAL CODE ({from_lang}):\n```{from_lang}\n{req.code}\n```\n\n"
            f"CONVERTED CODE ({to_lang}):\n```{to_lang}\n{converted}\n```\n\n"
            f"TARGET LANGUAGE: {to_lang}\n\n"
            f"Verify this converted code is 100% correct {to_lang} that compiles/runs with zero errors "
            f"and produces the same output as the original. "
            f"Check for: missing imports, wrong syntax, type errors, missing entry points, "
            f"accidental use of {from_lang} syntax in {to_lang} code. "
            f"If ANY issues exist, fix them ALL. Return the final verified code."
        )

        verify_content, verify_tokens = _call_llm(
            CONVERT_VERIFY_PROMPT, verify_prompt, json_mode=True, temperature=0.05
        )
        total_tokens += verify_tokens or 0

        verify_parsed = extract_json(verify_content)
        verified_code = verify_parsed.get("verified_code", "")
        fixes = verify_parsed.get("fixes_applied", [])

        if verified_code:
            converted = verified_code
            if fixes:
                notes = notes if isinstance(notes, list) else [str(notes)]
                notes.extend([f"[Auto-fix] {f}" for f in fixes])
                print(f"[Convert] Verification fixed {len(fixes)} issue(s): {fixes}")
            else:
                print("[Convert] Verification passed — no issues found.")
        else:
            print("[Convert] Verification returned empty code — using original conversion.")

    except Exception as exc:
        print(f"[Convert] Verification step failed ({exc}) — using original conversion.")

    return ConvertResponse(
        converted_code=converted,
        notes=notes if isinstance(notes, list) else [str(notes)],
        model=MODEL_ID,
        tokens_used=total_tokens if total_tokens > 0 else None,
    )


# ───────────────────────────────────────────────────────────────
# Routes — Frontend Pages
# ───────────────────────────────────────────────────────────────
@app.get("/login", response_class=HTMLResponse, tags=["Frontend"])
async def serve_login():
    """Serve login page."""
    return _serve_html("login.html")


@app.get("/app", response_class=HTMLResponse, tags=["Frontend"])
async def serve_app():
    """Serve main application page."""
    return _serve_html("index.html")


# Mount static assets from frontend folder
if FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ───────────────────────────────────────────────────────────────
# Entry Point — python main.py
# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    print("=" * 56)
    print("  AI Code Review & Rewrite Agent")
    print(f"  Model   : {MODEL_ID} (Google Gemini)")
    print(f"  Server  : http://localhost:8000")
    print(f"  Login   : http://localhost:8000/login")
    print(f"  App     : http://localhost:8000/app")
    print(f"  API Docs: http://localhost:8000/docs")
    print("=" * 56)

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
