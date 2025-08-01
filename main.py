from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import os
import json
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load API Key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic request model matching your input
class ResumeRequest(BaseModel):
    resume: dict
    summary: str = ""  # Optional, can be empty

def build_prompt(resume_json: dict, summary: str, job_title: str = "") -> str:
    formatted_resume = json.dumps(resume_json, indent=2)
    summary_section = f'Existing Summary: "{summary}"' if summary else "No summary provided."
    return f"""
You are a highly specialized Resume Summary Evaluator and Generator for an AI-powered ATS (Applicant Tracking System). Your job is to analyze the summary section of a resume and provide constructive feedback, an ATS score, and suggest improved summaries tailored to the resume content.

Below is a JSON resume:

{formatted_resume}

{summary_section}

Follow the steps below in order:

---

**Step 1: Extract Existing Summary**
- Extract and display the current "Summary" field from the resume, or use the provided summary if present.

---

**Step 2: Give ATS Score**
- Evaluate the extracted Summary and give an **ATS score out of 10** based on:
  - Keyword relevance to the job title
  - Alignment with skills, experience, education, certifications
  - Structure and clarity

---

**Step 3: Highlight Weak Sentences**
- Identify **sentences or phrases that reduced the ATS score**.
- Explain briefly why each one is ineffective or damaging.

---

**Step 4: Highlight Strong Sentences**
- Identify **sentences or phrases that improved the ATS score**.
- Explain briefly why each one is effective and ATS-friendly.

---

**Step 5: Score Justification**
- Give **2 to 4 bullet points** explaining why you gave this score.
- Use terms like keyword optimization, job relevance, measurable impact, etc.

---

**Step 6: Generate 4 New Summaries**
- Generate **4 optimized resume summaries** that would score **10/10 in an ATS**.
- Each summary should:
  - Highlight relevant skills, experience, education, and certifications
  - Use strong action words, avoid buzzwords or vague phrases
  - Be under 4 sentences

---

Respond in this JSON format:
{{
  "extracted_summary": "...",
  "ats_score": 0,
  "weak_sentences": ["..."],
  "strong_sentences": ["..."],
  "score_feedback": ["...", "..."],
  "new_summaries": ["...", "...", "...", "..."]
}}
"""

# Main route
@app.post("/evaluate-resume")
async def evaluate_resume(data: ResumeRequest):
    try:
        # Use provided summary if present, else fallback to resume["Summary"]
        summary = data.summary or data.resume.get("Summary", "")
        prompt = build_prompt(data.resume, summary)

        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                { "role": "system", "content": "You are a helpful assistant." },
                { "role": "user", "content": prompt }
            ],
            temperature=0.7
        )
        content = response.choices[0].message.content
        
        try:
            json_output = json.loads(content)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Response from OpenAI is not valid JSON")

        return {
            "status": "success",
            "output": json_output
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
