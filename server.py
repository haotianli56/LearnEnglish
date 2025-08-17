import io
import difflib
import re
import base64
import importlib
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from PIL import Image

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


class ImageRequest(BaseModel):
    text: str = Field(..., description="Text prompt: word, sentence, or conversation")
    size: str = Field("1024x1024", description="Image size: 256x256 | 512x512 | 1024x1024")
    format: str = Field("png", description="png | webp | jpeg")
    transparent_background: bool = Field(False, description="Use transparent background (png/webp only)")
    style_hint: Optional[str] = Field(
        default=None,
        description="Optional style hint: 'flat icon', 'watercolor', 'storybook illustration', 'realistic photo'"
    )


# ---------- Helpers ----------


VALID_SIZES = {"256x256", "512x512", "1024x1024"}
VALID_FORMATS = {"png", "jpeg", "webp"}


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


@app.post("/image/generate")
def generate_image(req: ImageRequest):
    if req.format not in {"png", "webp", "jpeg"}:
        raise HTTPException(status_code=400, detail="format must be png|webp|jpeg")
    if req.size not in VALID_SIZES:
        raise HTTPException(status_code=400, detail=f"size must be one of {sorted(VALID_SIZES)}")

    style = f"\nStyle hint: {req.style_hint}" if req.style_hint else ""
    prompt = (
        "Create a single, clear illustrative image that conveys the meaning of the text below. "
        "Avoid long text in the image; small labels are acceptable if helpful."
        f"{style}\n\nText:\n{req.text.strip()}"
    )

    gen_kwargs = dict(
        model="gpt-image-1",
        prompt=prompt,
        size=req.size,
        n=1,
    )
    wants_transparent = req.transparent_background and req.format in {"png", "webp"}
    if wants_transparent:
        gen_kwargs["background"] = "transparent"  # 不要传 "auto"

    try:
        img = client.images.generate(**gen_kwargs)
        b64 = img.data[0].b64_json
        png_bytes = base64.b64decode(b64)  # OpenAI 通常回的是 PNG
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image generation failed: {e}")

    try:
        with Image.open(io.BytesIO(png_bytes)) as im:
            if req.format == "jpeg":
                bg = Image.new("RGB", im.size, (255, 255, 255))
                if im.mode in ("RGBA", "LA"):
                    bg.paste(im, mask=im.split()[-1])
                else:
                    bg.paste(im)
                out_io = io.BytesIO()
                bg.save(out_io, format="JPEG", quality=95)
                out_bytes = out_io.getvalue()
            elif req.format == "webp":
                out_io = io.BytesIO()
                if wants_transparent:
                    im.save(out_io, format="WEBP", lossless=True)  # 保留透明
                else:
                    im.convert("RGB").save(out_io, format="WEBP", quality=95)
                out_bytes = out_io.getvalue()
            else:  # png
                out_io = io.BytesIO()
                im.save(out_io, format="PNG")
                out_bytes = out_io.getvalue()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PIL conversion failed: {e}")

    media_type = {
        "png": "image/png",
        "webp": "image/webp",
        "jpeg": "image/jpeg",
    }[req.format]

    headers = {"Content-Disposition": f'inline; filename="image.{req.format}"'}
    return StreamingResponse(io.BytesIO(out_bytes), media_type=media_type, headers=headers)


# --------- Local runner ---------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
