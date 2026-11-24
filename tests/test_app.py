import io

import pytest

from conftest import make_docx, make_pdf, register, upload
from utils import llm
from utils.document_processor import (DocumentError, parse_metadata, process_document,
                                      select_relevant_text, split_text_into_chunks)

INVOICE = "Invoice #42\nAcme Corp\nTotal due: $1,250.00\nDue date: 2026-01-31"


# ---------- auth ----------
def test_register_login_logout_flow(client):
    assert register(client).status_code == 302
    assert client.get("/dashboard").status_code == 200
    client.get("/logout")
    assert client.get("/dashboard").status_code == 302
    r = client.post("/login", data={"email": "A@Example.com", "password": "password123"})
    assert r.status_code == 302 and r.headers["Location"].endswith("/dashboard")


def test_bad_login_and_validation(client):
    register(client)
    client.get("/logout")
    assert client.post("/login", data={"email": "a@example.com", "password": "nope"}).status_code == 401
    assert register(client, email="not-an-email", name="X").status_code == 400
    assert register(client, email="b@example.com", password="short").status_code == 400
    assert register(client).status_code == 400  # duplicate email


def test_api_requires_login_with_json_401(client):
    r = client.post("/upload")
    assert r.status_code == 401 and r.get_json()["success"] is False
    assert client.get("/document/1").status_code == 401


# ---------- upload / summary / QA ----------
def test_upload_pdf_and_docx_then_ask(client, fake_gemini):
    register(client)
    r = upload(client, make_pdf(INVOICE), "invoice.pdf")
    assert r.get_json()["success"], r.get_json()
    assert "Acme Corp" in fake_gemini[0][1]  # extracted text reached the prompt
    doc_id = r.get_json()["doc_id"]

    r = upload(client, make_docx(["Meeting notes", "We agreed to ship on Friday."]), "notes.docx")
    assert r.get_json()["success"]

    d = client.get(f"/document/{doc_id}").get_json()
    assert d["summary"].startswith("This is an invoice")
    assert d["domain"] == "Financial" and "invoice" in d["keywords"]
    assert d["entities"]["amounts"] == ["$1,250.00"]

    a = client.post("/ask", json={"question": "What is the total?", "doc_id": doc_id}).get_json()
    assert a["success"] and "1,250" in a["answer"]
    assert "Total due" in fake_gemini[-1][1]

    assert b"invoice.pdf" in client.get("/dashboard").data
    assert client.get("/analytics").status_code == 200


def test_delete_document_removes_file(client, app):
    register(client)
    doc_id = upload(client, make_pdf(INVOICE), "invoice.pdf").get_json()["doc_id"]
    import os
    assert len(os.listdir(app.config["UPLOAD_FOLDER"])) == 1
    assert client.delete(f"/document/{doc_id}").get_json()["success"]
    assert os.listdir(app.config["UPLOAD_FOLDER"]) == []
    assert client.get(f"/document/{doc_id}").status_code == 404


# ---------- access control ----------
def test_user_cannot_access_other_users_documents(client, app):
    register(client, "a@example.com")
    doc_id = upload(client, make_pdf(INVOICE), "secret.pdf").get_json()["doc_id"]
    client.get("/logout")
    register(client, "b@example.com", name="Bob")
    assert client.get(f"/document/{doc_id}").status_code == 404
    assert client.delete(f"/document/{doc_id}").status_code == 404
    assert client.post("/ask", json={"question": "hi", "doc_id": doc_id}).status_code == 404
    assert b"secret.pdf" not in client.get("/dashboard").data
    assert b"secret.pdf" not in client.get("/analytics").data
    # and the doc survives
    client.get("/logout")
    client.post("/login", data={"email": "a@example.com", "password": "password123"})
    assert client.get(f"/document/{doc_id}").status_code == 200


# ---------- validation / hardening ----------
def test_rejects_bad_types_and_fake_files(client, app):
    register(client)
    assert upload(client, b"hello", "notes.txt").status_code == 400
    assert upload(client, b"x", "old.doc").status_code == 400
    assert upload(client, b"this is not a pdf", "fake.pdf").status_code == 422
    assert upload(client, b"this is not a docx", "fake.docx").status_code == 422
    assert client.post("/upload", data={}).status_code == 400
    import os
    assert os.listdir(app.config["UPLOAD_FOLDER"]) == []  # nothing left behind


def test_path_traversal_filename_is_sanitised(client, app):
    register(client)
    r = upload(client, make_pdf(INVOICE), "../../etc/evil.pdf")
    assert r.get_json()["success"]
    import os
    assert all(os.path.dirname(os.path.join(app.config["UPLOAD_FOLDER"], f)) == app.config["UPLOAD_FOLDER"]
               for f in os.listdir(app.config["UPLOAD_FOLDER"]))


def test_empty_and_scanned_documents(client):
    register(client)
    r = upload(client, make_docx([]), "empty.docx")
    assert r.status_code == 422 and "no text" in r.get_json()["error"].lower()
    from reportlab.pdfgen import canvas
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.rect(50, 50, 100, 100)  # drawing only, no text, like a scanned page
    c.save()
    r = upload(client, buf.getvalue(), "scan.pdf")
    assert r.status_code == 422 and "scanned" in r.get_json()["error"]


def test_too_large_upload(app, client):
    app.config["MAX_CONTENT_LENGTH"] = 1024
    register(client)
    r = upload(client, make_pdf(INVOICE) + b"0" * 5000, "big.pdf")
    assert r.status_code == 413 and r.get_json()["success"] is False


def test_ask_validation(client):
    register(client)
    assert client.post("/ask", json={"doc_id": 1}).status_code == 400
    assert client.post("/ask", json={"question": "q", "doc_id": "abc"}).status_code == 400
    assert client.post("/ask", data="not json").status_code == 400


class FakeAPIError(Exception):
    def __init__(self, code):
        super().__init__(f"{code} error")
        self.code = code


def test_llm_failure_gives_clean_503_and_no_leftovers(client, app, monkeypatch):
    def boom(model, prompt):
        raise FakeAPIError(503)

    monkeypatch.setattr(llm, "_call_model", boom)
    monkeypatch.setenv("LLM_RETRIES", "1")
    register(client)
    r = upload(client, make_pdf(INVOICE), "invoice.pdf")
    assert r.status_code == 503
    assert "temporarily unavailable" in r.get_json()["error"]
    import os
    assert os.listdir(app.config["UPLOAD_FOLDER"]) == []


def test_missing_api_key_message(client, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    register(client)
    r = upload(client, make_pdf(INVOICE), "invoice.pdf")
    assert r.status_code == 503 and "GEMINI_API_KEY" in r.get_json()["error"]


def test_html_in_filename_is_escaped_on_dashboard(client):
    register(client)
    upload(client, make_pdf(INVOICE), "<script>alert(1)</script>.pdf")
    assert b"<script>alert(1)" not in client.get("/dashboard").data


# ---------- LLM wrapper ----------
def test_fallback_chain_order(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "custom-model")
    assert llm.model_chain()[0] == "custom-model"
    assert llm.model_chain().count("gemini-3-flash-preview") == 1


def test_retries_503_then_succeeds(monkeypatch):
    seen = []

    def flaky(model, prompt):
        seen.append(model)
        if len(seen) < 3:
            raise FakeAPIError(503)
        return "ok"

    monkeypatch.setattr(llm, "_call_model", flaky)
    assert llm.generate("hi") == "ok"
    assert len(set(seen)) == 1  # retried same model


def test_404_falls_through_to_next_model(monkeypatch):
    seen = []

    def gone(model, prompt):
        seen.append(model)
        if model == llm.model_chain()[0]:
            raise FakeAPIError(404)
        return "from fallback"

    monkeypatch.setattr(llm, "_call_model", gone)
    assert llm.generate("hi") == "from fallback"
    assert seen == llm.model_chain()[:2]


def test_auth_error_fails_fast(monkeypatch):
    seen = []

    def denied(model, prompt):
        seen.append(model)
        raise FakeAPIError(403)

    monkeypatch.setattr(llm, "_call_model", denied)
    with pytest.raises(llm.LLMError, match="rejected"):
        llm.generate("hi")
    assert len(seen) == 1


# ---------- long documents ----------
def test_chunking_respects_limit():
    text = " ".join(["word"] * 5000 + ["x" * 300])
    chunks = split_text_into_chunks(text, 1000)
    assert all(len(c) <= 1000 for c in chunks) and len(chunks) > 1
    assert split_text_into_chunks("a" * 2500, 1000) == ["a" * 1000, "a" * 1000, "a" * 500]


def test_long_document_uses_map_reduce(client, fake_gemini):
    register(client)
    para = "The quarterly revenue grew steadily across all regions. " * 20
    pdf = make_docx([para] * 400)  # ~440k chars
    r = upload(client, pdf, "long.docx")
    assert r.get_json()["success"]
    summary_calls = [c for c in fake_gemini if "Return ONLY" not in c[1]]
    assert 3 <= len(summary_calls) <= 12  # chunk summaries + final combine, capped
    assert all(len(c[1]) < 70000 for c in fake_gemini)


def test_qa_on_long_document_picks_relevant_chunk():
    filler = "lorem ipsum dolor sit amet " * 20000
    text = filler + " The secret codeword is zebrafish. " + filler
    ctx = select_relevant_text(text, "What is the secret codeword?")
    assert "zebrafish" in ctx and len(ctx) <= 41000


# ---------- helpers ----------
def test_parse_metadata_is_robust():
    assert parse_metadata("garbage")["domain"] == "Other"
    assert parse_metadata('{"domain": 5, "keywords": "x", "entities": []}')["keywords"] == []
    assert parse_metadata('noise {"domain": "Legal"} noise')["domain"] == "Legal"


def test_process_document_rejects_unknown_extension(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hi")
    with pytest.raises(DocumentError):
        process_document(str(p))
