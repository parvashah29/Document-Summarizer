import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, url_for
from flask_login import (LoginManager, UserMixin, current_user, login_required,
                         login_user, logout_user)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

load_dotenv()

from utils.document_processor import (ALLOWED_EXTENSIONS, DocumentError, calculate_reading_time,  # noqa: E402
                                      get_answer, get_document_metadata, get_summary,
                                      get_word_count, process_document)
from utils.llm import LLMError  # noqa: E402

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
API_PREFIXES = ("/upload", "/ask", "/document", "/reprocess")

db = SQLAlchemy()
login_manager = LoginManager()
login_manager.login_view = "login"


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    documents = db.relationship('Document', backref='owner', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200), nullable=False)
    filepath = db.Column(db.String(500), nullable=False)
    summary = db.Column(db.Text)
    upload_date = db.Column(db.DateTime, default=_utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Metadata fields
    domain = db.Column(db.String(100), default='Unknown')
    keywords = db.Column(db.Text, default='[]')
    entities = db.Column(db.Text, default='{}')
    reading_time = db.Column(db.Integer, default=1)
    word_count = db.Column(db.Integer, default=0)
    
    def get_keywords_list(self):
        try:
            return json.loads(self.keywords) if self.keywords else []
        except (ValueError, TypeError):
            return []
    
    def get_entities_dict(self):
        try:
            return json.loads(self.entities) if self.entities else {}
        except (ValueError, TypeError):
            return {}


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


def _is_api_request():
    return request.path.startswith(API_PREFIXES)


def create_app(test_config=None):
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=os.environ.get("SECRET_KEY") or "dev-only-insecure-secret-key",
        SQLALCHEMY_DATABASE_URI=os.environ.get("DATABASE_URL", "sqlite:///docsum.db"),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        UPLOAD_FOLDER=os.environ.get("UPLOAD_FOLDER") or os.path.join(BASE_DIR, "uploads"),
        MAX_CONTENT_LENGTH=int(os.environ.get("MAX_UPLOAD_MB", "16")) * 1024 * 1024,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
    )
    if test_config:
        app.config.update(test_config)
    if app.config["SECRET_KEY"] == "dev-only-insecure-secret-key":
        app.logger.warning("SECRET_KEY is not set; using an insecure development default.")

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    db.init_app(app)
    login_manager.init_app(app)
    with app.app_context():
        db.create_all()

    @login_manager.unauthorized_handler
    def unauthorized():
        if _is_api_request():
            return jsonify({"success": False, "error": "Please log in first."}), 401
        return redirect(url_for("login"))

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_e):
        mb = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
        return jsonify({"success": False, "error": f"File is too large (max {mb} MB)."}), 413

    @app.errorhandler(Exception)
    def handle_error(exc):
        if isinstance(exc, HTTPException):
            code, message = exc.code, exc.description
        else:
            app.logger.exception("Unhandled error")
            code, message = 500, "Something went wrong on our side."
        if _is_api_request():
            return jsonify({"success": False, "error": message}), code
        return render_template("error.html", code=code, message=message), code

    register_routes(app)
    return app


def _json_error(message, status):
    return jsonify({"success": False, "error": message}), status


def _own_document_or_none(doc_id):
    doc = db.session.get(Document, doc_id)
    if doc is None or doc.user_id != current_user.id:
        return None  # same answer for "missing" and "someone else's"
    return doc


def register_routes(app):
    @app.route("/")
    def index():
        if current_user.is_authenticated:
            return redirect(url_for("dashboard"))
        return render_template("index.html")

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if current_user.is_authenticated:
            return redirect(url_for("dashboard"))
        if request.method == "POST":
            name = (request.form.get("name") or "").strip()
            email = (request.form.get("email") or "").strip().lower()
            password = request.form.get("password") or ""
            error = None
            if not name or len(name) > 100:
                error = "Please enter your name."
            elif not EMAIL_RE.match(email) or len(email) > 120:
                error = "Please enter a valid email address."
            elif len(password) < 8:
                error = "Password must be at least 8 characters."
            elif password != request.form.get("confirm_password"):
                error = "Passwords do not match"
            elif User.query.filter_by(email=email).first():
                error = "Email already registered"
            if error:
                return render_template("register.html", error=error), 400
            user = User(name=name, email=email)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            login_user(user)
            return redirect(url_for("dashboard"))
        return render_template("register.html")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if current_user.is_authenticated:
            return redirect(url_for("dashboard"))
        if request.method == "POST":
            email = (request.form.get("email") or "").strip().lower()
            user = User.query.filter_by(email=email).first()
            if user and user.check_password(request.form.get("password") or ""):
                login_user(user, remember=request.form.get("remember") == "on")
                return redirect(url_for("dashboard"))
            return render_template("login.html", error="Invalid email or password"), 401
        return render_template("login.html")

    @app.route("/logout")
    @login_required
    def logout():
        logout_user()
        return redirect(url_for("index"))

    @app.route("/dashboard")
    @login_required
    def dashboard():
        documents = (Document.query.filter_by(user_id=current_user.id)
                     .order_by(Document.upload_date.desc()).all())
        return render_template("dashboard.html", documents=documents)

    @app.route("/analytics")
    @login_required
    def analytics():
        documents = (Document.query.filter_by(user_id=current_user.id)
                     .order_by(Document.upload_date.desc()).all())
        domain_counts, keyword_counts, monthly_data = {}, {}, {}
        for doc in documents:
            domain = doc.domain or "Unknown"
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            for kw in doc.get_keywords_list():
                keyword_counts[kw] = keyword_counts.get(kw, 0) + 1
            month = doc.upload_date.strftime("%Y-%m")
            monthly_data[month] = monthly_data.get(month, 0) + 1
        top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        return render_template(
            "analytics.html",
            total_documents=len(documents),
            total_words=sum(d.word_count or 0 for d in documents),
            total_reading_time=sum(d.reading_time or 0 for d in documents),
            domain_counts=domain_counts,
            top_keywords=top_keywords,
            recent_docs=documents[:5],
            monthly_data=monthly_data,
        )

    @app.route("/upload", methods=["POST"])
    @login_required
    def upload():
        file = request.files.get("file")
        if file is None:
            return _json_error("No file provided.", 400)
        if not file.filename:
            return _json_error("No file selected.", 400)

        display_name = secure_filename(file.filename) or "document"
        ext = os.path.splitext(display_name)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return _json_error("Unsupported file type. Please upload a PDF or DOCX file.", 400)

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], f"{uuid.uuid4().hex}{ext}")
        file.save(filepath)
        try:
            text = process_document(filepath)
            summary = get_summary(text)
            metadata = get_document_metadata(text)
            document = Document(
                filename=display_name[:200],
                filepath=filepath,
                summary=summary,
                user_id=current_user.id,
                domain=metadata.get("domain", "Unknown"),
                keywords=json.dumps(metadata.get("keywords", [])),
                entities=json.dumps(metadata.get("entities", {})),
                reading_time=calculate_reading_time(text),
                word_count=get_word_count(text),
            )
            db.session.add(document)
            db.session.commit()
            return jsonify({"success": True, "message": "File uploaded and processed successfully!",
                            "doc_id": document.id})
        except DocumentError as exc:
            _discard(filepath)
            return _json_error(str(exc), 422)
        except LLMError as exc:
            _discard(filepath)
            return _json_error(str(exc), 503)
        except Exception:
            _discard(filepath)
            app.logger.exception("Upload processing failed")
            return _json_error("Document processing failed unexpectedly.", 500)

    @app.route("/document/<int:doc_id>")
    @login_required
    def get_document(doc_id):
        document = _own_document_or_none(doc_id)
        if document is None:
            return _json_error("Document not found.", 404)
        return jsonify({
            "success": True,
            "filename": document.filename,
            "summary": document.summary,
            "domain": document.domain,
            "keywords": document.get_keywords_list(),
            "entities": document.get_entities_dict(),
            "reading_time": document.reading_time,
            "word_count": document.word_count,
        })

    @app.route("/document/<int:doc_id>", methods=["DELETE"])
    @login_required
    def delete_document(doc_id):
        document = _own_document_or_none(doc_id)
        if document is None:
            return _json_error("Document not found.", 404)
        _discard(document.filepath)
        db.session.delete(document)
        db.session.commit()
        return jsonify({"success": True, "message": "Document deleted successfully."})

    @app.route("/ask", methods=["POST"])
    @login_required
    def ask():
        data = request.get_json(silent=True) or {}
        question = (data.get("question") or "").strip() if isinstance(data.get("question"), str) else ""
        if not question:
            return _json_error("Missing question.", 400)
        if len(question) > 2000:
            return _json_error("Question is too long (max 2000 characters).", 400)
        try:
            doc_id = int(data.get("doc_id"))
        except (TypeError, ValueError):
            return _json_error("Missing or invalid document ID.", 400)

        document = _own_document_or_none(doc_id)
        if document is None:
            return _json_error("Document not found.", 404)
        try:
            text = process_document(document.filepath)
            return jsonify({"success": True, "answer": get_answer(text, question)})
        except FileNotFoundError:
            return _json_error("The document file is missing on the server. Please re-upload.", 410)
        except DocumentError as exc:
            return _json_error(str(exc), 422)
        except LLMError as exc:
            return _json_error(str(exc), 503)

    @app.route("/reprocess", methods=["POST"])
    @login_required
    def reprocess_documents():
        """Re-run summary + metadata for the current user's documents."""
        processed = errors = 0
        for document in Document.query.filter_by(user_id=current_user.id).all():
            try:
                text = process_document(document.filepath)
                metadata = get_document_metadata(text)
                document.summary = get_summary(text)
                document.domain = metadata.get("domain", "Unknown")
                document.keywords = json.dumps(metadata.get("keywords", []))
                document.entities = json.dumps(metadata.get("entities", {}))
                document.reading_time = calculate_reading_time(text)
                document.word_count = get_word_count(text)
                processed += 1
            except Exception:
                app.logger.exception("Error reprocessing document %s", document.id)
                errors += 1
        db.session.commit()
        return jsonify({"success": True, "processed": processed, "errors": errors,
                        "message": f"Reprocessed {processed} documents. {errors} errors."})


def _discard(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_app().run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", 5001)),
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
    )
