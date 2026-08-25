import re

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from fpdf import FPDF
from fpdf.enums import XPos, YPos


def generate_pdf(text):
    safe_text = text.encode("latin-1", "replace").decode("latin-1").strip()
    paragraphs = re.split(r"\n\s*\n", safe_text)

    pdf = FPDF(format="Letter")
    pdf.set_margins(25, 25, 25)
    pdf.set_auto_page_break(True, margin=25)
    pdf.add_page()
    pdf.set_font("Times", size=12)

    line_height = 6
    paragraph_spacing = 4
    for i, paragraph in enumerate(paragraphs):
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        for line in lines:
            pdf.multi_cell(
                0, line_height, line, align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT
            )
        if i != len(paragraphs) - 1:
            pdf.ln(paragraph_spacing)

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
- Do not open with "I am writing to express my interest" or any close variant.
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
- Preserve the LaTeX document structure, commands, environments, and packages exactly. Do not remove, add, or reorder any \\section, \\begin/\\end, or preamble commands.
- Only rewrite the wording of existing bullet points, summary text, and skill list ordering to better match the job description.
- Do not invent or fabricate any experience, employer, title, date, skill, or metric that is not already present in the original resume.
- You may reorder bullet points within a section, and reorder items within a skills list, by relevance to the job, but every original bullet/skill must still appear somewhere.
- Return the complete, compilable .tex file and nothing else: no explanation, no markdown code fences, no commentary before or after.

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
