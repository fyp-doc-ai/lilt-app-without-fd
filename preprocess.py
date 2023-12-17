import torch
from transformers import AutoTokenizer

def normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]

# class to turn the keys of a dict into attributes (thanks Stackoverflow)
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class Preprocessor():
    def __init__(self, tokenizer):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.argsdict = {'max_seq_length': 512}
        self.args = AttrDict(self.argsdict)

    def get_boxes(self, ocr_df, image):
        words = list(ocr_df.text)
        coordinates = ocr_df[['left', 'top', 'width', 'height']]
        actual_boxes = []
        width, height = image.size
        for idx, row in coordinates.iterrows():
            x, y, w, h = tuple(row) # the row comes in (left, top, width, height) format
            actual_box = [x, y, x+w, y+h] # we turn it into (left, top, left+widght, top+height) to get the actual box 
            actual_boxes.append(actual_box)

        boxes = []
        for box in actual_boxes:
            boxes.append(normalize_box(box, width, height))

        return words, boxes, actual_boxes

    def convert_example_to_features(self, image, words, boxes, actual_boxes, cls_token_box=[0, 0, 0, 0],
                                    sep_token_box=[1000, 1000, 1000, 1000],
                                    pad_token_box=[0, 0, 0, 0]):
        width, height = image.size

        tokens = []
        token_boxes = []
        actual_bboxes = [] # we use an extra b because actual_boxes is already used
        token_actual_boxes = []
        offset_mapping = []
        for word, box, actual_bbox in zip(words, boxes, actual_boxes):
            word_tokens = self.tokenizer.tokenize(word)
            mapping = self.tokenizer(word, return_offsets_mapping=True).offset_mapping
            offset_mapping.extend(mapping)
            tokens.extend(word_tokens)
            token_boxes.extend([box] * len(word_tokens))
            actual_bboxes.extend([actual_bbox] * len(word_tokens))
            token_actual_boxes.extend([actual_bbox] * len(word_tokens))

        # Truncation: account for [CLS] and [SEP] with "- 2".
        special_tokens_count = 2
        if len(tokens) > self.args.max_seq_length - special_tokens_count:
            tokens = tokens[: (self.args.max_seq_length - special_tokens_count)]
            token_boxes = token_boxes[: (self.args.max_seq_length - special_tokens_count)]
            actual_bboxes = actual_bboxes[: (self.args.max_seq_length - special_tokens_count)]
            token_actual_boxes = token_actual_boxes[: (self.args.max_seq_length - special_tokens_count)]

        # add [SEP] token, with corresponding token boxes and actual boxes
        tokens += [self.tokenizer.sep_token]
        token_boxes += [sep_token_box]
        actual_bboxes += [[0, 0, width, height]]
        token_actual_boxes += [[0, 0, width, height]]

        segment_ids = [0] * len(tokens)

        # next: [CLS] token
        tokens = [self.tokenizer.cls_token] + tokens
        token_boxes = [cls_token_box] + token_boxes
        actual_bboxes = [[0, 0, width, height]] + actual_bboxes
        token_actual_boxes = [[0, 0, width, height]] + token_actual_boxes
        segment_ids = [1] + segment_ids

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = self.args.max_seq_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding_length
        input_mask += [0] * padding_length
        segment_ids += [self.tokenizer.pad_token_id] * padding_length
        token_boxes += [pad_token_box] * padding_length
        token_actual_boxes += [pad_token_box] * padding_length

        assert len(input_ids) == self.args.max_seq_length
        assert len(input_mask) == self.args.max_seq_length
        assert len(segment_ids) == self.args.max_seq_length
        assert len(token_boxes) == self.args.max_seq_length
        assert len(token_actual_boxes) == self.args.max_seq_length

        return input_ids, input_mask, segment_ids, token_boxes, token_actual_boxes, offset_mapping

    def process(self, ocr_df, image):
        words, boxes, actual_boxes = self.get_boxes(ocr_df, image)
        input_ids, input_mask, segment_ids, token_boxes, token_actual_boxes, offset_mapping = self.convert_example_to_features(image=image, words=words, boxes=boxes, actual_boxes=actual_boxes)
        input_ids = torch.tensor(input_ids).unsqueeze(0)
        attention_mask = torch.tensor(input_mask).unsqueeze(0)
        token_type_ids = torch.tensor(segment_ids).unsqueeze(0)
        bbox = torch.tensor(token_boxes).unsqueeze(0)
        return input_ids, attention_mask, token_type_ids, bbox, token_actual_boxes, offset_mapping