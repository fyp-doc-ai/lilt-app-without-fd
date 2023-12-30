from config import Settings
from preprocess import Preprocessor
import ocr
from PIL import Image
from transformers import LiltForTokenClassification, AutoTokenizer
import token_classification
import torch
from fastapi import FastAPI, UploadFile, Form
from contextlib import asynccontextmanager
import json
import io
from models import LiLTRobertaLikeForRelationExtraction
from base64 import b64decode 
config = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    config['settings'] = settings
    config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['vision_client'] = ocr.VisionClient(settings.GCV_AUTH)
    config['processor'] = Preprocessor(settings.TOKENIZER)
    config['tokenizer'] = AutoTokenizer.from_pretrained(settings.TOKENIZER)
    config['ser_model'] = LiltForTokenClassification.from_pretrained(settings.SER_MODEL)
    config['re_model'] = LiLTRobertaLikeForRelationExtraction.from_pretrained(settings.RE_MODEL)
    yield
    # Clean up and release the resources
    config.clear()

app = FastAPI(lifespan=lifespan)

@app.post("/submit-doc")
async def ProcessDocument(file: UploadFile):
  content = await file.read()
  tokenClassificationOutput, ocr_df, img_size = LabelTokens(content)
  reOutput = ExtractRelations(tokenClassificationOutput, ocr_df, img_size)
  return reOutput

@app.post("/submit-doc-mobile")
async def ProcessDocument(base64str: str = Form(...)):
  str_as_bytes = str.encode(base64str)
  content = b64decode(str_as_bytes)
  tokenClassificationOutput, ocr_df, img_size = LabelTokens(content)
  reOutput = ExtractRelations(tokenClassificationOutput, ocr_df, img_size)
  return reOutput

def LabelTokens(content):
  image = Image.open(io.BytesIO(content))
  ocr_df = config['vision_client'].ocr(content, image)
  input_ids, attention_mask, token_type_ids, bbox, token_actual_boxes, offset_mapping = config['processor'].process(ocr_df, image = image)
  token_labels = token_classification.classifyTokens(config['ser_model'], input_ids, attention_mask, bbox, offset_mapping)
  return {"token_labels": token_labels, "input_ids": input_ids, "bbox":bbox, "attention_mask":attention_mask}, ocr_df, image.size

def ExtractRelations(tokenClassificationOutput, ocr_df, img_size):
  token_labels = tokenClassificationOutput['token_labels']
  input_ids = tokenClassificationOutput['input_ids']
  attention_mask = tokenClassificationOutput["attention_mask"]
  bbox_org = tokenClassificationOutput["bbox"]

  merged_output, merged_words = token_classification.createEntities(config['ser_model'], token_labels, input_ids, ocr_df, config['tokenizer'], img_size, bbox_org)
  
  entities = merged_output['entities']
  input_ids = torch.tensor([merged_output['input_ids']]).to(config['device'])
  bbox = torch.tensor([merged_output['bbox']]).to(config['device'])
  attention_mask = torch.tensor([merged_output['attention_mask']]).to(config['device'])

  id2label = {"HEADER":0, "QUESTION":1, "ANSWER":2}
  decoded_entities = []
  for entity in entities:
    decoded_entities.append((entity['label'], config['tokenizer'].decode(input_ids[0][entity['start']:entity['end']])))
    entity['label'] = id2label[entity['label']]

  config['re_model'].to(config['device'])
  entity_dict = {'start': [entity['start'] for entity in entities], 'end': [entity['end'] for entity in entities], 'label': [entity['label'] for entity in entities]}
  relations = [{'start_index': [], 'end_index': [], 'head': [], 'tail': []}]
  with torch.no_grad():
    outputs = config['re_model'](input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, entities=[entity_dict], relations=relations)

  decoded_pred_relations = []
  for relation in outputs.pred_relations[0]:
    head_start, head_end = relation['head']
    tail_start, tail_end = relation['tail']
    question =  config['tokenizer'].decode(input_ids[0][head_start:head_end])
    answer = config['tokenizer'].decode(input_ids[0][tail_start:tail_end])
    decoded_pred_relations.append((question, answer))

  return {"pred_relations":json.dumps(outputs.pred_relations[0]), "entities":json.dumps(entities), "input_ids": json.dumps(input_ids.tolist()), "bboxes": json.dumps(bbox_org.tolist()),"token_labels":json.dumps(token_labels), "decoded_entities": json.dumps(decoded_entities), "decoded_pred_relations":json.dumps(decoded_pred_relations)}