from fastapi import FastAPI
from pydantic import BaseModel

from search import search, get_answer
from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import JSONResponse


class QueryRequest(BaseModel):
    query: str


class AnswerRequest(BaseModel):
    query: str
    paragraphs: list[str]


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_origins=["*"],
)


@app.get("/health")
async def health_check():
    return "All good ;)"


@app.post("/query")
async def get_paragraphs(req: QueryRequest):
    result = search(req.query)
    return result


@app.post("/answer")
async def answer(req: AnswerRequest):
    result = get_answer(req.query, req.paragraphs)

    return {"answer": result}
