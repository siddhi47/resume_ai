import html as html_lib
import re

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from fpdf import FPDF

MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")


def _line_to_html(line):
    # Escape first so real markup can't be injected via the LLM's own text, then convert the
    # one markdown construct we care about (links) into a real, clickable <a> tag — otherwise
    # a link like [project](https://...) would render as inert bracket text in the PDF.
    escaped = html_lib.escape(line)
    return MARKDOWN_LINK_PATTERN.sub(r'<a href="\2">\1</a>', escaped)


def generate_pdf(text):
    safe_text = text.encode("latin-1", "replace").decode("latin-1").strip()
    paragraphs = re.split(r"\n\s*\n", safe_text)

    pdf = FPDF(format="Letter")
    pdf.set_margins(25, 25, 25)
    pdf.set_auto_page_break(True, margin=25)
    pdf.add_page()
    pdf.set_font("Times", size=12)

    html_paragraphs = []
    for paragraph in paragraphs:
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if lines:
            html_paragraphs.append("<br>".join(_line_to_html(line) for line in lines))

    pdf.write_html(
        "".join(f"<p>{p}</p>" for p in html_paragraphs),
        font_family="Times",
    )

    return bytes(pdf.output())


def sanitize_filename_part(text):
    return re.sub(r"[^A-Za-z0-9]", "", text or "")


def strip_code_fence(text):
    stripped = text.strip()
    match = re.match(r"^```[a-zA-Z]*\n(.*)\n```$", stripped, re.DOTALL)
    return match.group(1) if match else stripped


BANNED_PHRASES = [
    "thrilled",
    "excited to leverage",
    "passionate about",
    "fast-paced environment",
    "dynamic team",
    "furthermore",
    "moreover",
    "in today's",
    "proven track record",
    "seamlessly",
    "robust",
    "delve",
    "testament to",
    "unlock",
    "unwavering",
]

_BANNED_PHRASES_LINE = ", ".join(f'"{p}"' for p in BANNED_PHRASES)


def get_job_context(job_desc):
    llm = ChatOpenAI(temperature=0.5)

    summary_prompt = PromptTemplate(
        input_variables=["job_description"],
        template="""
            You are an expert in job descriptions. Summarize the following job description into:
            0. Company Name
            1. Location
            2. Job Title
            3. Key Responsibilities
            4. required Skills
            5. Preferred Qualifications
            6. Any other relevant information
            If the company name is not explicitly stated, infer it from context (e.g. "About <Company>" section) rather than leaving it blank.
            Job Description:
            {job_description}
            """,
    )
    summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
    job_summary = summary_chain.run(job_description=job_desc)

    company_prompt = PromptTemplate(
        input_variables=["job_description"],
        template="""
            Extract only the hiring company's name from the following job description.
            Respond with the company name only, nothing else, no punctuation, no explanation.
            If not explicitly stated, infer it from context (e.g. an "About <Company>" section).

            Job Description:
            {job_description}
            """,
    )
    company_chain = LLMChain(llm=llm, prompt=company_prompt)
    company = company_chain.run(job_description=job_desc).strip()

    return job_summary, company


COVER_LETTER_PROMPT_TEMPLATE = """
Based on the following resume and job description, write a professional cover letter tailored for this position.
You are to use the relevant information from the resume and job description to create a personalized cover letter.
Address the letter to the company named in the job description below, and reference that company by name in the body.
Use today's date: {today}.

Placeholders:
- Never write bracketed or guessed placeholder text of any kind, such as "[Company Address]", "[City, State, Zip Code]", "[Hiring Manager Name]", "[Your Name]", or anything similar. This applies to every detail, not just the examples listed here.
- If a specific detail (company street address, hiring manager's name, your own mailing address, etc.) is not explicitly present in the Resume, Contact Info, or Job Description below, omit that line or field entirely instead of guessing or inserting a placeholder.
- Only include a company address block if the exact address is explicitly stated in the Job Description below. Otherwise, skip straight from the date to the greeting.
- Only greet a named hiring manager if their name is explicitly stated in the Job Description below. Otherwise use a generic greeting such as "Dear Hiring Team," instead of guessing a name.

Location:
- The candidate's current city is given in Contact Info below. Compare it to the job's location in the job description.
- If the job's city/region is the same as (or a normal commuting distance from) the candidate's current city, do NOT talk about relocation. Instead, work in a brief, natural mention that the candidate is already local (e.g. based in the same city or region as the job) somewhere in the letter, without making it sound like a formulaic disclaimer.
- If the job's city/region is clearly different from the candidate's current city, mention willingness to relocate instead.

Writing style:
- Pull specific, concrete details from the resume (project names, tools, metrics, past employers) instead of vague claims of enthusiasm or skill.
- Vary sentence length. Do not write every sentence in the same "subject-verb-adjective-noun" shape.
- Do not use these words/phrases or close variants of them: {banned_phrases}.
- Do not use em dashes.
- Do not open the letter with any variant of the pattern "I am [writing/excited/eager/pleased/thrilled] to [express my interest in/apply for/join] the [role] position" — this entire sentence shape is a cliché, not just the exact words "writing to express my interest". Instead, open with a specific fact from the resume that connects directly to the role (e.g. naming a relevant project, tool, or result in the first sentence).
- Keep it to 3-4 short paragraphs. No filler sentences that restate the job posting back at the reader.

Relevant GitHub Projects:
- {relevant_projects}
- If the above is empty or says none were found, do not mention GitHub or any specific project name at all — just write the letter from the Resume alone.
- If specific real projects are listed, you may reference them by name as further evidence of the skills already claimed in the Resume, but do not invent details about them beyond what's given here.

Revision feedback:
- {revision_feedback}
- If the above is not "None.", it describes a problem with a previous draft. Fix that specific problem while keeping everything else about the letter the same.

Contact Info:
{contact_info}

Resume:
{resume_summary}

Job Description:
{job_description}
""".replace("{banned_phrases}", _BANNED_PHRASES_LINE)

GENERIC_QA_PROMPT_TEMPLATE = """
You are an AI assistant that helps job applicants.
ROLE: You are the applicant. Answer the question as if you were the human applicant.

Resume:
{resume_summary}

Job Description:
{job_description}

Task:
{user_question}

Placeholders:
- Never write bracketed or guessed placeholder text of any kind, such as "[Company Address]", "[City, State, Zip Code]", "[Hiring Manager Name]", "[Your Name]", or anything similar.
- If a specific detail is not explicitly present in the Resume or Job Description above, or explicitly given in the Task, omit it entirely instead of guessing or inserting a placeholder.

Writing style:
- Answer with specific, concrete details pulled from the resume (project names, tools, metrics, past employers) rather than vague claims of enthusiasm or skill.
- Vary sentence length. Do not write every sentence in the same "subject-verb-adjective-noun" shape.
- Do not use these words/phrases or close variants of them: {banned_phrases}.
- Do not use em dashes.
- Do not restate the question back at the reader before answering it.
- Give a detailed, personalized response, but no filler sentences.
""".replace("{banned_phrases}", _BANNED_PHRASES_LINE)

RESUME_TAILORING_PROMPT_TEMPLATE = """
You are tailoring a candidate's LaTeX resume for a specific job.

Rules:
- Preserve the LaTeX document structure, commands, environments, packages, and section list exactly. Do not remove, add, or reorder any \\section or preamble commands, and do not touch any section other than Projects (see below for the one exception there).
- Only rewrite the wording of existing bullet points, summary text, and skill list ordering to better match the job description.
- Do not invent or fabricate any experience, employer, title, date, skill, or metric that is not already present in the original resume or explicitly given to you as a verified GitHub project below.
- You may reorder bullet points within a section, and reorder items within a skills list, by relevance to the job, but every original bullet/skill must still appear somewhere.
- Copy every existing \\href{{URL}} exactly character-for-character, even if a URL looks misspelled — it is a real link and "fixing" it will break it. Never alter, "correct", or retype an existing URL.
- In any NEW text you write (a new project's name, description, or a \\href's display text), escape LaTeX special characters or the document will fail to compile: underscore as \\_, ampersand as \\&, percent as \\%, hash as \\#. This matters most for GitHub repo names/URLs, which often contain underscores — e.g. write \\href{{https://github.com/user/my_repo}}{{github.com/user/my\\_repo}}, never the unescaped form.
- Return the complete, compilable .tex file and nothing else: no explanation, no markdown code fences, no commentary before or after.

Relevant GitHub Projects:
- {relevant_projects}
- If this lists specific real GitHub projects (with a name, description, and URL) that are NOT already in the Original Resume's Projects section, you may add ONE new entry per such project to the Projects section, copying the exact same LaTeX pattern already used there (a \\begin{{twocolentry}}...\\textbf{{Project Name}}\\end{{twocolentry}} followed by \\begin{{onecolentry}}\\begin{{highlights}}\\item description with a \\href{{URL}}{{link text}}\\end{{highlights}}\\end{{onecolentry}}). Use the exact name/description/URL given — never invent details about a project beyond what's given here.
- If nothing here is new (already in the resume, or this says "None found."), do not add anything — just use it as a signal for which EXISTING bullets/skills to prioritize when reordering.

Revision feedback:
- {revision_feedback}
- If the above is not "None.", it describes a problem with a previous attempt (e.g. a compile error, a dropped bullet, or a fabricated detail). Fix that specific problem while keeping everything else the same.

Job Description:
{job_description}

Original Resume (.tex):
{tex_source}
"""

CONTACT_INFO_PROMPT_TEMPLATE = """
Extract the following from the resume:
- Your Name
- Your Address
- City, State, Zip Code
- Email Address
- Phone Number

Resume:
{resume_text}

Respond in JSON format.
"""
