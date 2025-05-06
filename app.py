import os
import tempfile
from flask import Flask, render_template, redirect, request, url_for, session
from flask_login import (
    LoginManager,
    login_user,
    logout_user,
    login_required,
    current_user,
)
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from src.resume_bot.auth import users, User
from src.resume_bot.forms import LoginForm, ResumeChatForm
import hashlib

# LangChain & OpenAI
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain

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
        query = form.question.data

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
                    template="""
                    Extract the following from the resume:
                    - Your Name
                    - Your Address
                    - City, State, Zip Code
                    - Email Address
                    - Phone Number

                    Resume:
                    {resume_text}

                    Respond in JSON format.
                    """,
                )
                llm = ChatOpenAI(temperature=0.5)
                contact_chain = LLMChain(llm=llm, prompt=contact_prompt)
                session["extracted_info"] = contact_chain.run(
                    resume_text=full_resume_text
                )
                session["resume_summary"] = full_resume_text[:4000]
                extracted_info = session["extracted_info"]
                resume_summary = session["resume_summary"]

        llm = ChatOpenAI(temperature=0.5)
        # Summarize job description prompt

        summary_prompt = PromptTemplate(
            input_variables=["job_description"],
            template="""
                You are an expert in job descriptions. Summarize the following job description into:
                0. Location
                1. Job Title
                2. Key Responsibilities
                3. required Skills
                4. Preferred Qualifications
                5. Any other relevant information
                Job Description:
                {job_description}
                """,
        )
        summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
        job_summary = summary_chain.run(job_description=job_desc)
        if "cover letter" in query.lower():
            cover_prompt = PromptTemplate(
                input_variables=["resume_summary", "job_description"],
                template="""
                Based on the following resume and job description, write a professional cover letter tailored for this position.
                You are to use the relevant information from the resume and job description to create a personalized cover letter.
                Talk about willingness to relocate if the job is far away from the location in resume.
                Resume:
                {resume_summary}

                Job Description:
                {job_description}
                """,
            )
            chain = LLMChain(llm=llm, prompt=cover_prompt)
            answer = chain.run(
                resume_summary=resume_summary, job_description=job_summary
            )

        else:
            generic_prompt = PromptTemplate(
                input_variables=["resume_summary", "job_description", "user_question"],
                template="""
                You are an AI assistant that helps job applicants.
                ROLE: You are the applicant. Answer the question as if you were the human applicant.

                Resume:
                {resume_summary}

                Job Description:
                {job_description}

                Task:
                {user_question}

                Give a detailed, personalized response.
                """,
            )
            prompt = generic_prompt.format(
                resume_summary=resume_summary,
                job_description=job_summary,
                user_question=query,
            )
            answer = llm.predict(prompt)

    return render_template(
        "chat.html", form=form, answer=answer, extracted_info=extracted_info
    )


@app.route("/logout")
@login_required
def logout():
    logout_user()
    session.clear()
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True)
