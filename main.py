from config import Settings
from preprocess import Preprocessor
import ocr
from PIL import Image
from transformers import LiltForTokenClassification
import token_classification
import torch
from fastapi import FastAPI, UploadFile
from contextlib import asynccontextmanager
import json
import io
from models import LiLTRobertaLikeForRelationExtraction
config = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    config['settings'] = settings
    config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['vision_client'] = ocr.VisionClient(settings.GCV_AUTH)
    config['processor'] = Preprocessor(settings.TOKENIZER)
    config['ser_model'] = LiltForTokenClassification.from_pretrained(settings.SER_MODEL)
    config['re_model'] = LiLTRobertaLikeForRelationExtraction.from_pretrained(settings.RE_MODEL)
    yield
    # Clean up and release the resources
    config.clear()

app = FastAPI(lifespan=lifespan)

@app.post("/submit-doc")
async def ProcessDocument(file: UploadFile):
  tokenClassificationOutput = await LabelTokens(file)
  reOutput = ExtractRelations(tokenClassificationOutput)
  return reOutput

async def LabelTokens(file):
  content = await file.read()
  image = Image.open(io.BytesIO(content))
  ocr_df = config['vision_client'].ocr(content, image)
  input_ids, attention_mask, token_type_ids, bbox, token_actual_boxes, offset_mapping = config['processor'].process(ocr_df, image = image)
  token_labels = token_classification.classifyTokens(config['ser_model'], input_ids, attention_mask, bbox, offset_mapping)
  return {"token_labels": token_labels, "input_ids": input_ids, "bbox":bbox, "offset_mapping":offset_mapping, "attention_mask":attention_mask}

def ExtractRelations(tokenClassificationOutput):
  token_labels = tokenClassificationOutput['token_labels']
  input_ids = tokenClassificationOutput['input_ids']
  offset_mapping =  tokenClassificationOutput["offset_mapping"]
  attention_mask = tokenClassificationOutput["attention_mask"]
  bbox = tokenClassificationOutput["bbox"]

  entities = token_classification.createEntities(config['ser_model'], token_labels, input_ids, offset_mapping)
  
  config['re_model'].to(config['device'])
  entity_dict = {'start': [entity[0] for entity in entities], 'end': [entity[1] for entity in entities], 'label': [entity[3] for entity in entities]}
  relations = [{'start_index': [], 'end_index': [], 'head': [], 'tail': []}]
  with torch.no_grad():
    outputs = config['re_model'](input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, entities=[entity_dict], relations=relations)

  print(type(outputs.pred_relations[0]))
  print(type(entities))
  print(type(input_ids))
  print(type(bbox))
  print(type(token_labels))
  # "pred_relations":json.dumps(outputs.pred_relations[0]), "entities":json.dumps(entities), "input_ids": json.dumps(input_ids.tolist()), 

  return {"pred_relations":json.dumps(outputs.pred_relations[0]), "entities":json.dumps(entities), "input_ids": json.dumps(input_ids.tolist()), "bboxes": json.dumps(bbox.tolist()),"token_labels":json.dumps(token_labels)}