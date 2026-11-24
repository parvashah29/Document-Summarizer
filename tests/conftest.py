import io
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app, db  # noqa: E402
from utils import llm  # noqa: E402


def make_pdf(text):
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    y = 750
    for line in text.split("\n"):
        c.drawString(72, y, line)
        y -= 16
        if y < 60:
            c.showPage()
            y = 750
    c.save()
    return buf.getvalue()


def make_docx(paragraphs):
    import docx

    d = docx.Document()
    for p in paragraphs:
        d.add_paragraph(p)
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


@pytest.fixture(autouse=True)
def fake_gemini(monkeypatch):
    """Replace the network call; records prompts so tests can assert on them."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BACKOFF_SECONDS", "0")
    calls = []

    def fake(model, prompt):
        calls.append((model, prompt))
        if "Return ONLY a valid JSON" in prompt:
            return ('```json\n{"domain": "Financial", "keywords": ["invoice", "acme"], '
                    '"entities": {"people": [], "organizations": ["Acme Corp"], "locations": [], '
                    '"dates": [], "amounts": ["$1,250.00"]}}\n```')
        if "Question:" in prompt:
            return "The total is $1,250.00."
        return "This is an invoice from Acme Corp for $1,250.00."

    monkeypatch.setattr(llm, "_call_model", fake)
    return calls


@pytest.fixture
def app(tmp_path):
    app = create_app({
        "TESTING": True,
        "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 'test.db'}",
        "UPLOAD_FOLDER": str(tmp_path / "uploads"),
        "SECRET_KEY": "test",
    })
    yield app
    with app.app_context():
        db.session.remove()
        db.engine.dispose()


@pytest.fixture
def client(app):
    return app.test_client()


def register(client, email="a@example.com", name="Alice", password="password123"):
    return client.post("/register", data={
        "name": name, "email": email, "password": password, "confirm_password": password})


def upload(client, data, filename):
    return client.post("/upload", data={"file": (io.BytesIO(data), filename)},
                       content_type="multipart/form-data")
