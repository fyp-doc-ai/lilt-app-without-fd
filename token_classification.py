import numpy as np
from preprocess import normalize_box
import copy

def classifyTokens(model, input_ids, attention_mask, bbox, offset_mapping):
    outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask)
    # take argmax on last dimension to get predicted class ID per token
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    return predictions

def compare_boxes(b1,b2):
  b1 = np.array([c for c in b1])
  b2 = np.array([c for c in b2])
  equal = np.array_equal(b1,b2)
  return equal

def mergable(w1,w2):
  if w1['label'] == w2['label']:
    threshold = 7
    if abs(w1['box'][1] - w2['box'][1]) < threshold or abs(w1['box'][-1] - w2['box'][-1]) < threshold:
      return True
    return False
  return False

def convert_data(data, tokenizer, img_size):
  def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]


  def simplify_bbox(bbox):
      return [
          min(bbox[0::2]),
          min(bbox[1::2]),
          max(bbox[2::2]),
          max(bbox[3::2]),
      ]

  def merge_bbox(bbox_list):
    x0, y0, x1, y1 = list(zip(*bbox_list))
    return [min(x0), min(y0), max(x1), max(y1)]

  tokenized_doc = {"input_ids": [], "bbox": [], "labels": [], "attention_mask":[]}
  entities = []
  id2label = {}
  entity_id_to_index_map = {}
  empty_entity = set()
  for line in data:
      if len(line["text"]) == 0:
          empty_entity.add(line["id"])
          continue
      id2label[line["id"]] = line["label"]
      tokenized_inputs = tokenizer(
          line["text"],
          add_special_tokens=False,
          return_offsets_mapping=True,
          return_attention_mask=True,
      )
      text_length = 0
      ocr_length = 0
      bbox = []
      for token_id, offset in zip(tokenized_inputs["input_ids"], tokenized_inputs["offset_mapping"]):
          if token_id == 6:
              bbox.append(None)
              continue
          text_length += offset[1] - offset[0]
          tmp_box = []
          while ocr_length < text_length:
              ocr_word = line["words"].pop(0)
              ocr_length += len(
                  tokenizer._tokenizer.normalizer.normalize_str(ocr_word["text"].strip())
              )
              tmp_box.append(simplify_bbox(ocr_word["box"]))
          if len(tmp_box) == 0:
              tmp_box = last_box
          bbox.append(normalize_bbox(merge_bbox(tmp_box), img_size))
          last_box = tmp_box  # noqa
      bbox = [
          [bbox[i + 1][0], bbox[i + 1][1], bbox[i + 1][0], bbox[i + 1][1]] if b is None else b
          for i, b in enumerate(bbox)
      ]
      if line["label"] == "other":
          label = ["O"] * len(bbox)
      else:
          label = [f"I-{line['label'].upper()}"] * len(bbox)
          label[0] = f"B-{line['label'].upper()}"
      tokenized_inputs.update({"bbox": bbox, "labels": label})
      if label[0] != "O":
          entity_id_to_index_map[line["id"]] = len(entities)
          entities.append(
              {
                  "start": len(tokenized_doc["input_ids"]),
                  "end": len(tokenized_doc["input_ids"]) + len(tokenized_inputs["input_ids"]),
                  "label": line["label"].upper(),
              }
          )
      for i in tokenized_doc:
          tokenized_doc[i] = tokenized_doc[i] + tokenized_inputs[i]

  chunk_size = 512
  output = {}
  for chunk_id, index in enumerate(range(0, len(tokenized_doc["input_ids"]), chunk_size)):
    item = {}
    entities_in_this_span = []
    for k in tokenized_doc:
        item[k] = tokenized_doc[k][index : index + chunk_size]
    global_to_local_map = {}
    for entity_id, entity in enumerate(entities):
        if (
            index <= entity["start"] < index + chunk_size
            and index <= entity["end"] < index + chunk_size
        ):
            entity["start"] = entity["start"] - index
            entity["end"] = entity["end"] - index
            global_to_local_map[entity_id] = len(entities_in_this_span)
            entities_in_this_span.append(entity)
    item.update(
        {
            "entities": entities_in_this_span
        }
    )
    for key in item.keys():
      output[key] = output.get(key, []) + item[key]
  return output

def dfs(i, merged, width, height, visited, df_words):
    v_threshold = int(.01 * height)
    h_threshold = int(.08 * width)
    visited.add(i)
    merged.append(df_words[i])

    for j in range(len(df_words)):
        if j not in visited:
            w1 = df_words[i]['words'][0]
            w2 = df_words[j]['words'][0]

            # and
            if (abs(w1['box'][1] - w2['box'][1]) < v_threshold or abs(w1['box'][-1] - w2['box'][-1]) < v_threshold) \
                and (df_words[i]['label'] == df_words[j]['label']) \
                and (abs(w1['box'][0] - w2['box'][0]) < h_threshold or abs(w1['box'][-2] - w2['box'][-2]) < h_threshold):
                dfs(j,merged, width, height, visited, df_words)
    return merged

def createEntities(model, predictions, input_ids, ocr_df, tokenizer, img_size, bbox):
    width, height = img_size
    words = []
    for index,row in ocr_df.iterrows():
        word = {}
        origin_box = [row['left'],row['top'],row['left']+row['width'],row['top']+row['height']]
        word['word_text'] = row['text']
        word['word_box'] = origin_box
        word['normalized_box'] = normalize_box(word['word_box'], width, height)
        words.append(word)

    raw_input_ids = input_ids[0].tolist()
    token_boxes = bbox.squeeze().tolist()
    special_tokens = [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]

    input_ids = [id for id in raw_input_ids if id not in special_tokens]
    predictions = [model.config.id2label[prediction] for i,prediction in enumerate(predictions) if not (raw_input_ids[i] in special_tokens)]
    actual_boxes = [box for i,box in enumerate(token_boxes) if not (raw_input_ids[i] in special_tokens )]

    assert(len(actual_boxes) == len(predictions))

    for word in words:
        word_labels = []
        token_labels = []
        word_tagging = None
        for i,box in enumerate(actual_boxes,start=0):
            if compare_boxes(word['normalized_box'],box):
                if predictions[i] != 'O':
                    word_labels.append(predictions[i][2:])
                else:
                    word_labels.append('O')
                token_labels.append(predictions[i])
        if word_labels != []:
            word_tagging =  word_labels[0] if word_labels[0] != 'O' else word_labels[-1]
        else:
            word_tagging = 'O'
        word['word_labels'] = token_labels
        word['word_tagging'] = word_tagging

    filtered_words = [{'id':i,'text':word['word_text'],
                'label':word['word_tagging'],
                'box':word['word_box'],
                'words':[{'box':word['word_box'],'text':word['word_text']}]} for i,word in enumerate(words) if word['word_tagging'] != 'O']

    merged_taggings = []
    df_words = filtered_words.copy()
    visited = set()
    for i in range(len(df_words)):
        if i not in visited:
            merged_taggings.append(dfs(i,[], width, height, visited, df_words))

    merged_words = []
    for i,merged_tagging in enumerate(merged_taggings):
        if len(merged_tagging) > 1:
            new_word = {}
            merging_word = " ".join([word['text'] for word in merged_tagging])
            merging_box = [merged_tagging[0]['box'][0]-5,merged_tagging[0]['box'][1]-10,merged_tagging[-1]['box'][2]+5,merged_tagging[-1]['box'][3]+10]
            new_word['text'] = merging_word
            new_word['box'] = merging_box
            new_word['label'] = merged_tagging[0]['label']
            new_word['id'] = filtered_words[-1]['id']+i+1
            new_word['words'] = [{'box':word['box'],'text':word['text']} for word in merged_tagging]
            # new_word['start'] =
            merged_words.append(new_word)

    filtered_words.extend(merged_words)
    predictions = [word['label'] for word in filtered_words]
    actual_boxes = [word['box'] for word in filtered_words]
    unique_taggings = set(predictions)

    output = convert_data(copy.deepcopy(merged_words), tokenizer, img_size)
    return output, merged_words

