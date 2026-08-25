import os
import io
import subprocess
import tempfile
import uuid
from flask import (
    Flask,
    render_template,
    redirect,
    request,
    url_for,
    session,
    send_file,
    send_from_directory,
    flash,
    jsonify,
)
from flask_login import (
    LoginManager,
    login_user,
    logout_user,
    login_required,
    current_user,
)
from dotenv import load_dotenv
from src.resume_bot.auth import users, User
from src.resume_bot.forms import LoginForm, ResumeChatForm
from src.resume_bot.shared import (
    generate_pdf,
    sanitize_filename_part,
    strip_code_fence,
    get_job_context,
    COVER_LETTER_PROMPT_TEMPLATE,
    GENERIC_QA_PROMPT_TEMPLATE,
    RESUME_TAILORING_PROMPT_TEMPLATE,
    CONTACT_INFO_PROMPT_TEMPLATE,
)
from src.resume_bot.agent.graph import invoke_agent_sync, get_conversation_history
import hashlib

# LangChain & OpenAI
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
import datetime

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")

login_manager = LoginManager()
login_manager.init_app(app)


@login_manager.user_loader
def load_user(user_id):
    if user_id in users:
        return User(user_id)
    return None


@app.route("/", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        password = hashlib.md5(password.encode()).hexdigest()
        if username in users and users[username]["password"] == password:
            login_user(User(username))
            return redirect(url_for("chat"))
        else:
            return render_template("login.html", form=form, error="Invalid credentials")
    return render_template("login.html", form=form)


@app.route("/chat", methods=["GET", "POST"])
@login_required
def chat():
    form = ResumeChatForm()
    answer = None
    extracted_info = session.get("extracted_info")
    resume_summary = session.get("resume_summary")

    user_resume_path = f"static/resumes/{current_user.id}_resume.pdf"
    os.makedirs("static/resumes", exist_ok=True)

    if form.validate_on_submit():
        job_desc = form.job_description.data
        # summarize job description
        query = form.question.data or ""

        if form.resume.data:
            file = form.resume.data
            file.save(user_resume_path)

        if not resume_summary or not extracted_info:
            if os.path.exists(user_resume_path):
                loader = PyPDFLoader(user_resume_path)
                pages = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200
                )
                docs = text_splitter.split_documents(pages)
                full_resume_text = "\n".join([doc.page_content for doc in docs])

                contact_prompt = PromptTemplate(
                    input_variables=["resume_text"],
                    template=CONTACT_INFO_PROMPT_TEMPLATE,
                )
                llm = ChatOpenAI(temperature=1, model="o4-mini")
                contact_chain = LLMChain(llm=llm, prompt=contact_prompt)
                session["extracted_info"] = contact_chain.run(
                    resume_text=full_resume_text
                )
                session["resume_summary"] = full_resume_text[:4000]
                extracted_info = session["extracted_info"]
                resume_summary = session["resume_summary"]

        job_summary, company = get_job_context(job_desc)
        session["last_company"] = company

        llm = ChatOpenAI(temperature=0.5)
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        if "cover letter" in query.lower():
            cover_prompt = PromptTemplate(
                input_variables=["resume_summary", "job_description", "contact_info"],
                template=COVER_LETTER_PROMPT_TEMPLATE,
            )
            chain = LLMChain(llm=llm, prompt=cover_prompt)
            answer = chain.run(
                resume_summary=resume_summary,
                job_description=job_summary,
                contact_info=extracted_info,
                today=today,
                relevant_projects="",
                revision_feedback="None.",
            )

        else:
            generic_prompt = PromptTemplate(
                input_variables=["resume_summary", "job_description", "user_question"],
                template=GENERIC_QA_PROMPT_TEMPLATE,
            )
            prompt = generic_prompt.format(
                resume_summary=resume_summary,
                job_description=job_summary,
                user_question=query,
            )
            answer = llm.predict(prompt)

        session["last_answer"] = answer

    return render_template(
        "chat.html", form=form, answer=answer, extracted_info=extracted_info
    )


@app.route("/download-pdf")
@login_required
def download_pdf():
    answer = session.get("last_answer")
    if not answer:
        return redirect(url_for("chat"))
    company = sanitize_filename_part(session.get("last_company"))
    filename = f"SiddhiCoverLetter{company}.pdf" if company else "SiddhiCoverLetter.pdf"
    pdf_bytes = generate_pdf(answer)
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=filename,
    )


@app.route("/download-resume", methods=["POST"])
@login_required
def download_resume():
    form = ResumeChatForm()
    if not form.validate_on_submit():
        flash("Job description is required to generate a tailored resume.")
        return redirect(url_for("chat"))

    job_desc = form.job_description.data
    job_summary, company = get_job_context(job_desc)
    session["last_company"] = company

    user_resume_tex_path = f"static/resumes/{current_user.id}_resume_template.tex"
    if not os.path.exists(user_resume_tex_path):
        flash(
            f"No resume template found. Place your resume's .tex file at {user_resume_tex_path} and try again."
        )
        return redirect(url_for("chat"))

    with open(user_resume_tex_path, "r", encoding="utf-8", errors="replace") as f:
        tex_source = f.read()

    resume_llm = ChatOpenAI(temperature=0, model="gpt-4o")
    resume_prompt = PromptTemplate(
        input_variables=["tex_source", "job_description"],
        template=RESUME_TAILORING_PROMPT_TEMPLATE,
    )
    resume_chain = LLMChain(llm=resume_llm, prompt=resume_prompt)
    tailored_tex = strip_code_fence(
        resume_chain.run(
            tex_source=tex_source,
            job_description=job_summary,
            relevant_projects="None found.",
            revision_feedback="None.",
        )
    )

    resumes_dir = "static/resumes"
    tailored_tex_path = f"{resumes_dir}/{current_user.id}_tailored_resume.tex"
    tailored_pdf_path = f"{resumes_dir}/{current_user.id}_tailored_resume.pdf"
    with open(tailored_tex_path, "w", encoding="utf-8") as f:
        f.write(tailored_tex)

    company_sanitized = sanitize_filename_part(company)
    base_filename = (
        f"SiddhiResume{company_sanitized}" if company_sanitized else "SiddhiResume"
    )

    compile_result = subprocess.run(
        ["tectonic", "--outdir", resumes_dir, tailored_tex_path],
        capture_output=True,
        text=True,
        timeout=300,
    )

    if compile_result.returncode == 0 and os.path.exists(tailored_pdf_path):
        with open(tailored_pdf_path, "rb") as f:
            pdf_bytes = f.read()
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"{base_filename}.pdf",
        )

    return send_file(
        io.BytesIO(tailored_tex.encode("utf-8")),
        mimetype="application/x-tex",
        as_attachment=True,
        download_name=f"{base_filename}.tex",
    )


def _agent_thread_id():
    # Defaults to the user's id (the original single-thread behavior) so existing sessions/
    # conversations keep working; a "New Chat" click stores a fresh id here instead.
    return session.setdefault("agent_thread_id", current_user.id)


@app.route("/agent", methods=["GET"])
@login_required
def agent_chat():
    thread_id = _agent_thread_id()
    history, artifact_path, artifact_label = get_conversation_history(thread_id)
    initial_artifact = None
    if artifact_path:
        initial_artifact = {
            "url": url_for("agent_download", filename=artifact_path),
            "label": artifact_label or artifact_path,
        }
    return render_template(
        "agent_chat.html",
        history=history,
        initial_artifact=initial_artifact,
    )


@app.route("/agent/new", methods=["POST"])
@login_required
def agent_new_chat():
    session["agent_thread_id"] = f"{current_user.id}:{uuid.uuid4().hex[:12]}"
    return redirect(url_for("agent_chat"))


@app.route("/agent/message", methods=["POST"])
@login_required
def agent_message():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "empty message"}), 400

    thread_id = _agent_thread_id()
    result = invoke_agent_sync(message, thread_id, current_user.id)

    reply = ""
    for m in reversed(result.get("messages", [])):
        if getattr(m, "type", None) == "ai" and m.content:
            reply = m.content
            break

    artifact_path = result.get("last_artifact_path")
    artifact = None
    if artifact_path:
        artifact = {
            "url": url_for("agent_download", filename=artifact_path),
            "label": result.get("last_artifact_label") or artifact_path,
        }

    return jsonify({"reply": reply, "artifact": artifact})


@app.route("/agent/download/<path:filename>")
@login_required
def agent_download(filename):
    directory = os.path.join("data", "agent_artifacts", current_user.id)
    return send_from_directory(directory, filename, as_attachment=True)


@app.route("/logout")
@login_required
def logout():
    logout_user()
    session.clear()
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True)
