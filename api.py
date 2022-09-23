from fastapi import FastAPI
from pydantic import BaseModel
from src.utils import get_logger,load_configs
from inference import t5inference


configs = load_configs('./src/conf/config.yaml')
logger = get_logger('logs.log')
t5_infer = t5inference(configs,logger)

class Summarizer(BaseModel):
    text:str

app = FastAPI()

@app.get('/')
def show_status():
    return {"health status: running!"}


@app.post('/predictions')
def predict(sz:Summarizer):
    output = t5_infer.infer_single(sz.text)
    return {"Summarized ":output}