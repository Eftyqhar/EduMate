from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import tempfile, os, io, urllib.parse

from rag import ingest_pdf, summarize, explain_topic, ask
from quiz import generate_quiz
from voice import transcribe_audio, text_to_speech

app = FastAPI(title="AI Study Assistant")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")

def err(e: Exception, status: int = 500):
    return JSONResponse(status_code=status, content={"detail": str(e)})


@app.post("/upload")
async def upload_pdf(session_id: str = Form(...), file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        return err(ValueError("Only PDF files are supported."), 400)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        ingest_pdf(session_id, tmp_path)
    except Exception as e:
        return err(e)
    finally:
        os.unlink(tmp_path)
    return {"message": "PDF uploaded and indexed.", "session_id": session_id}


@app.get("/summarize/{session_id}")
async def summarize_pdf(session_id: str, lang: str = "en"):
    try:
        return {"summary": summarize(session_id, lang)}
    except Exception as e:
        return err(e)


class TopicRequest(BaseModel):
    topic: str
    lang: str = "en"

@app.post("/explain/{session_id}")
async def explain(session_id: str, body: TopicRequest):
    try:
        return {"explanation": explain_topic(session_id, body.topic, body.lang)}
    except Exception as e:
        return err(e)


@app.get("/quiz/{session_id}")
async def quiz(session_id: str, num_questions: int = 5, lang: str = "en"):
    try:
        return {"quiz": generate_quiz(session_id, num_questions, lang)}
    except Exception as e:
        return err(e)


class QuestionRequest(BaseModel):
    question: str
    lang: str = "en"

@app.post("/ask/{session_id}")
async def ask_question(session_id: str, body: QuestionRequest):
    try:
        return {"answer": ask(session_id, body.question, body.lang)}
    except Exception as e:
        return err(e)


@app.post("/voice/{session_id}")
async def voice_qa(session_id: str, audio: UploadFile = File(...), lang: str = Form(default="en")):
    try:
        audio_bytes = await audio.read()
        question = transcribe_audio(audio_bytes, lang)
        answer = ask(session_id, question, lang)
        audio_response = text_to_speech(answer, lang)
        return StreamingResponse(
            io.BytesIO(audio_response),
            media_type="audio/mpeg",
            headers={
                "X-Question": urllib.parse.quote(question),
                "X-Answer": urllib.parse.quote(answer),
            },
        )
    except Exception as e:
        return err(e)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
