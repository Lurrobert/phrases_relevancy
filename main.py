from typing import Union
import json
import logging
import logging.config

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from relevancy import top_match

app = FastAPI()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class RelevantInputs(BaseModel):
    vitae: str
    reqs: str


@app.post("/relevant_score")
def read_item(data: RelevantInputs):
    vitae, reqs = data.vitae.split('\n'), data.reqs.split('\n')
    return {"top_matches": top_match(reqs, vitae)}
