"""
Personal Portfolio AI — FastAPI Backend
========================================
Parses a PDF resume into structured JSON via Groq LLM, then serves
as a conversational AI assistant that answers questions about the candidate.
"""

import json
import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from groq import Groq
from pydantic import BaseModel, Field
from pypdf import PdfReader

# ──────────────────────────────────────────────
#  Environment & Client Setup
# ──────────────────────────────────────────────

# Load .env from the same directory as this file
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")

# Initialise client lazily — None until first use
_groq_client: Optional[Groq] = None

# Model that reliably supports JSON mode on your account
MODEL_NAME = "openai/gpt-oss-20b"


def get_groq_client() -> Groq:
    """Return the Groq client, creating it on first call."""
    global _groq_client
    if _groq_client is not None:
        return _groq_client
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. "
            "Create a .env file with GROQ_API_KEY=your_key (see .env.example)."
        )
    _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


# ──────────────────────────────────────────────
#  Pydantic Schemas (v2 syntax)
# ──────────────────────────────────────────────


class Experience(BaseModel):
    """A single work or internship experience entry."""
    company: str = Field(description="Company or organization name")
    role: str = Field(description="Job title or role held")
    duration: str = Field(description="Time period of the role")
    description: list[str] = Field(default_factory=list)


class Education(BaseModel):
    """An academic qualification."""
    institution: str = Field(description="University or school name")
    degree: str = Field(description="Degree or certification obtained")
    duration: str = Field(description="Time period of study")
    grade: Optional[str] = Field(default=None, description="GPA, percentage, or grade")


class Project(BaseModel):
    """A notable project from the resume."""
    name: str = Field(description="Project title")
    technologies: list[str] = Field(
        default_factory=list, description="Tech stack used"
    )
    description: list[str] = Field(
        default_factory=list, description="What the project does / key highlights"
    )


class Skills(BaseModel):
    """Categorised technical and soft skills."""
    languages: list[str] = Field(default_factory=list)
    frameworks: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    soft_skills: list[str] = Field(default_factory=list)


class ResumeSchema(BaseModel):
    """Top-level structured representation of the entire resume."""
    name: Optional[str] = Field(default=None, description="Candidate full name")
    email: Optional[str] = Field(default=None)
    phone: Optional[str] = Field(default=None)
    linkedin: Optional[str] = Field(default=None)
    github: Optional[str] = Field(default=None)
    summary: Optional[str] = Field(
        default=None, description="Professional summary or objective"
    )
    education: list[Education] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    skills: Skills = Field(default_factory=Skills)
    certifications: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)


# ──────────────────────────────────────────────
#  PDF Parsing Utility
# ──────────────────────────────────────────────


def read_pdf(file_path: str) -> str:
    """Extract text from every page of a resume PDF."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Resume PDF not found at '{file_path}'. "
            "Please place your resume as 'my_resume.pdf' in the backend/ directory."
        )
    try:
        reader = PdfReader(file_path)
        text_parts: list[str] = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

        full_text = "\n".join(text_parts).strip()
        if not full_text:
            raise RuntimeError("The PDF appears to contain no extractable text.")
        return full_text
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Failed to read PDF: {exc}") from exc


# ──────────────────────────────────────────────
#  Structured LLM Extraction
# ──────────────────────────────────────────────


def parse_resume(resume_text: str) -> ResumeSchema:
    """
    Send raw resume text to Groq LLM and receive structured JSON
    matching our ResumeSchema.
    """
    schema_json = ResumeSchema.model_json_schema()

    system_prompt = (
        "You are an expert resume parser. Extract ALL information from the "
        "provided resume text and return it as a single JSON object that "
        "strictly follows this schema:\n\n"
        f"{json.dumps(schema_json, indent=2)}\n\n"
        "Rules:\n"
        "- Extract every detail faithfully; do NOT invent information.\n"
        "- If a field is not present in the resume, use null or an empty list.\n"
        "- Return ONLY the JSON object, no extra commentary.\n"
        "- Do NOT include any thinking, reasoning, or explanation."
    )

    try:
        response = get_groq_client().chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": resume_text},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        raw_content: str = response.choices[0].message.content or ""
        parsed_data = json.loads(raw_content)
        return ResumeSchema.model_validate(parsed_data)

    except json.JSONDecodeError as exc:
        raise RuntimeError(f"LLM returned invalid JSON: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Resume parsing failed: {exc}") from exc


# ──────────────────────────────────────────────
#  Global State — Parsed Resume Cache
# ──────────────────────────────────────────────

_parsed_resume: Optional[ResumeSchema] = None
RESUME_PATH = os.path.join(os.path.dirname(__file__) or ".", "my_resume.pdf")


def get_parsed_resume() -> ResumeSchema:
    """Return the cached parsed resume, parsing it on first call."""
    global _parsed_resume
    if _parsed_resume is None:
        raw_text = read_pdf(RESUME_PATH)
        _parsed_resume = parse_resume(raw_text)
    return _parsed_resume


# ──────────────────────────────────────────────
#  FastAPI Application
# ──────────────────────────────────────────────

app = FastAPI(
    title="Personal Portfolio AI",
    description="An AI assistant that answers questions about Gunjan Singh's portfolio.",
    version="1.0.0",
)

# CORS — allow all origins (Vercel frontend + local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ────────────────


class ChatRequest(BaseModel):
    """Incoming chat question from the frontend."""
    question: str = Field(
        ..., min_length=1, description="The user's question about the portfolio"
    )


class ChatResponse(BaseModel):
    """AI-generated answer returned to the frontend."""
    answer: str


# ── Endpoints ────────────────────────────────


@app.get("/")
async def root():
    """
    Health-check endpoint that also triggers resume parsing on the first call.
    Prints the parsed JSON to the server console for debugging.
    """
    try:
        resume = get_parsed_resume()
        print("\n✅ Parsed Resume JSON:")
        print(resume.model_dump_json(indent=2))
        return {"message": "Portfolio AI Backend is running"}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat")
async def chat(request: ChatRequest, stream: bool = False):
    """
    Receive a question, combine it with the parsed resume context,
    query the Groq LLM, and return the AI's answer.

    Query params:
        stream: If true, returns Server-Sent Events. Otherwise returns JSON.
    """
    try:
        resume = get_parsed_resume()
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    resume_json: str = resume.model_dump_json(indent=2)

    system_prompt = (
        "You are a friendly, professional AI assistant who personally represents "
        "Gunjan Singh. Think of yourself as Gunjan's knowledgeable spokesperson "
        "who has deeply understood her background and can discuss it naturally.\n\n"
        "STRICT BOUNDARIES — FOLLOW THESE FIRST:\n"
        "- You ONLY answer questions related to Gunjan's professional profile: "
        "skills, education, projects, work experience, certifications, achievements, "
        "and career-related topics.\n"
        "- If someone asks about personal life, favourite food, relationships, hobbies, "
        "opinions, politics, or ANYTHING not covered in the resume data below, "
        "politely but firmly decline. Say something like: "
        "'I'm here to help with questions about Gunjan's professional background — "
        "her skills, projects, education, and experience. Feel free to ask about those!'\n"
        "- Do NOT try to guess, speculate, or be helpful about off-topic questions. "
        "Simply redirect to professional topics.\n"
        "- Never invent or exaggerate anything.\n\n"
        "HOW TO RESPOND (for valid questions):\n"
        "- Write in natural, flowing English — like a real person talking, not a database.\n"
        "- NEVER just list or copy-paste resume fields. Instead, synthesize the "
        "information into thoughtful, conversational paragraphs.\n"
        "- Analyze and connect the dots — e.g., relate skills to projects, "
        "explain how experiences build on each other, highlight what makes Gunjan stand out.\n"
        "- Use a warm, confident tone (e.g., 'Gunjan has...' or 'She has worked on...').\n"
        "- Keep answers concise but insightful — 2 to 4 sentences for simple questions, "
        "a short paragraph for detailed ones.\n"
        "- Do NOT use bullet points, bold markers, or structured formatting unless "
        "the user specifically asks for a list.\n\n"
        f"GUNJAN'S RESUME DATA (use as your knowledge source, not as a template to copy):\n"
        f"{resume_json}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request.question},
    ]

    # ── Streaming mode (SSE) ──
    if stream:
        async def stream_response():
            try:
                chunks = get_groq_client().chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1024,
                    stream=True,
                )
                for chunk in chunks:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        data = json.dumps({"content": delta.content})
                        yield f"data: {data}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as exc:
                error_data = json.dumps({"error": str(exc)})
                yield f"data: {error_data}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    # ── Normal mode (JSON) ──
    try:
        response = get_groq_client().chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
        )
        answer: str = response.choices[0].message.content or ""
        return {"answer": answer}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM query failed: {exc}") from exc


# ──────────────────────────────────────────────
#  Run with: uvicorn main:app --reload --port 8000
# ──────────────────────────────────────────────
