# DocSum AI - Document Summarizer & Q&A

A Flask web app where users register, upload PDF or DOCX documents, get a Gemini-generated summary (with domain, keywords and extracted entities) and ask natural-language questions about each document.

## Features

- Register / login / logout (hashed passwords, Flask-Login sessions)
- Upload PDF and DOCX (validated by extension, file signature and size; stored under random names)
- AI summary, domain, keywords and entities per document
- Question answering over a document
- Per-user analytics page (documents, words, reading time, domains, top keywords)
- Per-user access control: other users' documents return 404
- Long documents: summarised in chunks (map-reduce); Q&A selects the most relevant passages
- Clear errors for scanned/image-only PDFs (no OCR), empty, corrupt or password-protected files, and when the AI service is unavailable
- Model fallback chain with retry and exponential backoff on 429/5xx

Not supported: legacy `.doc` files and OCR of scanned PDFs.

## Setup

Requires Python 3.10+ (tested on 3.12) and a free [Gemini API key](https://aistudio.google.com/apikey).

```bash
git clone https://github.com/parvashah29/Document-Summarizer.git
cd Document-Summarizer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then edit: GEMINI_API_KEY and SECRET_KEY
python app.py
```

Open http://localhost:5001 (port 5001 avoids macOS AirPlay on 5000). The SQLite database is created automatically in `instance/`. All settings are documented in `.env.example`.

## Models

`GEMINI_MODEL` is tried first, then `gemini-3-flash-preview`, `gemini-3.1-flash-lite`, `gemini-2.5-flash`. 404 (model unavailable) moves to the next model; 429/503 are retried with backoff first. Google's free tier often returns transient 503s, so a request can occasionally fail with a "temporarily unavailable" message - just retry.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests mock the Gemini call and run offline (auth, upload of generated PDF/DOCX, summary, Q&A, access control, validation, retry/fallback, chunking).

## Project layout

```
app.py                      Flask app factory, models, routes
utils/document_processor.py text extraction, chunking, summary / metadata / Q&A prompts
utils/llm.py                Gemini client with fallback + retry
templates/                  Jinja templates (Tailwind via CDN)
tests/                      pytest suite
```

## Security notes

Passwords are hashed; uploads get random server-side names; AI output is HTML-escaped in the UI. There is no CSRF token layer (session cookies are `SameSite=Lax`); add Flask-WTF before exposing this publicly. Use `SECRET_KEY` and a production WSGI server (e.g. gunicorn) for deployment.
