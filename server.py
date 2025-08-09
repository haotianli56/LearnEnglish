import io
import difflib
import re
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from openai import OpenAI
import uvicorn
import os

# ---- Setup ----
# Set OPENAI_API_KEY in your environment before running.
client = OpenAI()
app = FastAPI(title="English Learning API", version="1.0")

# ---------- Schemas ----------

class ContentParams(BaseModel):
    learner_level: str = Field(..., description="CEFR level like A1–C2, or 'beginner/intermediate/advanced'")
    topic: str = Field(..., description="Topic, e.g., 'ordering coffee', 'job interview'")
    length: str = Field("short", description="short | medium | long")
    grammar_focus: Optional[List[str]] = Field(default=None, description="e.g., ['present perfect', 'comparatives']")
    vocab: Optional[List[str]] = Field(default=None, description="Words/phrases to include")
    format: str = Field("lesson", description="lesson | dialogue | reading | quiz | mixed")
    include_answers: bool = Field(True, description="If quiz present, include answers/keys")
    use_ipa: bool = Field(True, description="Show IPA for key vocabulary")

class ContentResponse(BaseModel):
    content: str

class TTSRequest(BaseModel):
    text: str
    voice: str = "alloy"  # voices vary by model; 'alloy' is a safe default
    format: str = "mp3"   # mp3 | wav | flac

# ---------- Helpers ----------

def tokenize_words(s: str) -> List[str]:
    # simple word tokenizer
    return re.findall(r"[A-Za-z']+", s.lower())

def word_diffs(target: str, spoken: str):
    """Return alignment-style diffs (kept, replaced, added, removed)"""
    t = tokenize_words(target)
    s = tokenize_words(spoken)
    sm = difflib.SequenceMatcher(a=t, b=s)
    ops = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        ops.append({"tag": tag, "target": t[i1:i2], "spoken": s[j1:j2]})
    return ops

def summarize_diff_for_llm(ops):
    problems = []
    correct = []
    for op in ops:
        if op["tag"] == "equal" and op["target"]:
            correct.extend(op["target"])
        elif op["tag"] in ("replace", "delete", "insert"):
            if op["target"] or op["spoken"]:
                problems.append(op)
    return correct[:50], problems[:100]

# ---------- Routes ----------

@app.post("/content/english", response_model=ContentResponse)
def generate_content(params: ContentParams):
    """Generate leveled English-learning content."""
    system_text = (
        "You are an expert ESL curriculum designer. "
        "Produce engaging, level-appropriate content with clear structure. "
        "If IPA is requested, add / slashes for phonemes. Keep sections tidy."
    )

    resp = client.responses.create(
        model="gpt-4.1-mini",
        temperature=0.7,
        max_output_tokens=1200,
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_text}],
            },
            {
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": (
                        f"Create an English {params.format} for a {params.learner_level} learner.\n"
                        f"Topic: {params.topic}\n"
                        f"Target length: {params.length}\n"
                        f"Grammar focus: {', '.join(params.grammar_focus or []) or 'none'}\n"
                        f"Vocabulary to include: {', '.join(params.vocab or []) or 'none'}\n"
                        f"Include answers: {params.include_answers}\n"
                        f"Use IPA for key words: {params.use_ipa}\n"
                        "Structure with headings, examples, and short practice.\n"
                        "End with 3 practical usage tips."
                    )
                }],
            },
        ],
    )

    content = resp.output_text
    return ContentResponse(content=content)


@app.post("/pronunciation/evaluate")
async def evaluate_pronunciation(
    target_text: str = Form(..., description="The ground-truth text the learner attempted to read"),
    audio: UploadFile = File(..., description="Learner's recording: wav/mp3/m4a"),
    use_ipa: bool = Form(True),
):
    # 1) Transcribe learner audio
    try:
        data = await audio.read()
        ext = audio.filename.split(".")[-1].lower() if "." in audio.filename else "wav"
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=(f"audio.{ext}", data),
        )
        spoken_text = (transcript.text or "").strip()
        if not spoken_text:
            raise RuntimeError("Empty transcription result")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Transcription failed: {e!s}")

    # 2) Rough alignment to locate issues
    ops = word_diffs(target_text, spoken_text)
    kept, problems = summarize_diff_for_llm(ops)

    # 3) LLM evaluation
    judge_prompt = (
        "Evaluate English pronunciation quality based on mismatches between the target text and the ASR transcript. "
        "Assume ASR is mostly correct but not perfect. "
        "Identify (a) likely mispronounced words or reductions, (b) stress/intonation issues, (c) specific phoneme tips. "
        "If IPA requested, include it only for problematic words. "
        "Return a JSON-ish report with: overall_score(0-100), strengths[], issues[{word, why, ipa?, tip}]"
    )

    analysis = client.responses.create(
        model="gpt-4.1-mini",
        temperature=0.3,
        max_output_tokens=900,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": "You are a strict but helpful English pronunciation coach."}]},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": judge_prompt},
                    {"type": "input_text", "text": f"TARGET:\n{target_text}"},
                    {"type": "input_text", "text": f"TRANSCRIPT:\n{spoken_text}"},
                    {"type": "input_text", "text": f"DIFF_OPS (summary):\n{problems}"},
                    {"type": "input_text", "text": f"KEPT_WORDS (sample):\n{kept}"},
                    {"type": "input_text", "text": f"Include IPA: {use_ipa}"},
                ],
            },
        ],
    )

    return JSONResponse({
        "target_text": target_text,
        "transcript": spoken_text,
        "word_alignment_ops": ops,
        "evaluation": analysis.output_text,
    })


@app.post("/tts")
def text_to_speech(req: TTSRequest):
    """
    Generate a clean reference audio reading to model ideal pronunciation.
    """
    if req.format not in {"mp3", "wav", "flac"}:
        raise HTTPException(status_code=400, detail="format must be mp3|wav|flac")

    try:
        speech = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice=req.voice,
            input=req.text,
            response_format=req.format
        )

        if isinstance(speech, (bytes, bytearray)):
            audio_bytes = bytes(speech)
        elif hasattr(speech, "read"):
            audio_bytes = speech.read()
        elif hasattr(speech, "content"):
            audio_bytes = speech.content
        else:
            audio_bytes = bytes(speech)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"TTS failed: {e!s}")

    media_type = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "flac": "audio/flac",
    }[req.format]

    return StreamingResponse(io.BytesIO(audio_bytes), media_type=media_type)


# --------- Local runner ---------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
