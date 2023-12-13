import numpy as np

def classifyTokens(model, input_ids, attention_mask, bbox, offset_mapping):
    outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask)
    # take argmax on last dimension to get predicted class ID per token
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    return predictions

def createEntities(model, predictions, input_ids, offset_mapping):
    # we're only interested in tokens which aren't subwords
    # we'll use the offset mapping for that
    offset_mapping = np.array(offset_mapping)
    is_subword = np.array(offset_mapping.squeeze().tolist())[:,0] != 0

    id2label = {"HEADER":0, "QUESTION":1, "ANSWER":2}

    # finally, store recognized "question" and "answer" entities in a list
    entities = []
    current_entity = None
    start = None
    end = None

    for idx, (id, pred) in enumerate(zip(input_ids[0].tolist(), predictions)):
        if not is_subword[idx]:
            predicted_label = model.config.id2label[pred]
            if predicted_label.startswith("B") and current_entity is None:
                # means we're at the start of a new entity
                current_entity = predicted_label.replace("B-", "")
                start = idx
            if current_entity is not None and current_entity not in predicted_label:
                # means we're at the end of a new entity
                end = idx
                entities.append((start, end, current_entity, id2label[current_entity]))
                current_entity = None

    return entities