from google.cloud import vision
from google.oauth2 import service_account
from google.protobuf.json_format import MessageToJson
import pandas as pd
import json
import numpy as np
from PIL import Image
import io

image_ext = ("*.jpg", "*.jpeg", "*.png")

class VisionClient:
    def __init__(self, auth):
        credentials = service_account.Credentials.from_service_account_info(
            auth
        )
        self.client = vision.ImageAnnotatorClient(credentials=credentials)

    def send_request(self, image):
        try:
            image = vision.Image(content=image)
        except ValueError as e:
            print("Image could not be read")
            return
        response = self.client.document_text_detection(image, timeout=10)
        return response

    def get_response(self, content):
        try:
            resp_js = self.send_request(content)
        except Exception as e:
            print("OCR request failed. Reason : {}".format(e))
        
        return resp_js

    def post_process(self, resp_js):
        boxObjects = []
        for i in range(1, len(resp_js.text_annotations)):
            # We need to do that because vision sometimes reverse the left and right coords so then we have negative
            # width which causes problems when drawing link buttons
            obj = resp_js
            if obj.text_annotations[i].bounding_poly.vertices[1].x > obj.text_annotations[i].bounding_poly.vertices[3].x:
                leftX = obj.text_annotations[i].bounding_poly.vertices[3].x
            else:
                leftX = obj.text_annotations[i].bounding_poly.vertices[1].x

            if obj.text_annotations[i].bounding_poly.vertices[1].x > obj.text_annotations[i].bounding_poly.vertices[3].x:
                rightX = obj.text_annotations[i].bounding_poly.vertices[1].x
            else:
                rightX = obj.text_annotations[i].bounding_poly.vertices[3].x

            boxObjects.append({
                "id": i-1,
                "text": obj.text_annotations[i].description,
                "left": leftX,
                "width": rightX - leftX,
                "top": obj.text_annotations[i].bounding_poly.vertices[1].y,
                "height":obj.text_annotations[i].bounding_poly.vertices[3].y - obj.text_annotations[i].bounding_poly.vertices[1].y
            })

        return boxObjects

    def convert_to_df(self, boxObjects, image):
        ocr_df = pd.DataFrame(boxObjects)

        # ocr_df = ocr_df.sort_values(by=['top', 'left'], ascending=True).reset_index(drop=True)
        width, height = image.size
        w_scale = 1000/width
        h_scale = 1000/height

        ocr_df = ocr_df.dropna() \
                    .assign(left_scaled = ocr_df.left*w_scale,
                            width_scaled = ocr_df.width*w_scale,
                            top_scaled = ocr_df.top*h_scale,
                            height_scaled = ocr_df.height*h_scale,
                            right_scaled = lambda x: x.left_scaled + x.width_scaled,
                            bottom_scaled = lambda x: x.top_scaled + x.height_scaled)

        float_cols = ocr_df.select_dtypes('float').columns
        ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
        ocr_df = ocr_df.replace(r'^\s*$', np.nan, regex=True)
        ocr_df = ocr_df.dropna().reset_index(drop=True)
        return ocr_df

    def ocr(self, content, image):
        resp_js = self.get_response(content)
        boxObjects = self.post_process(resp_js)
        ocr_df = self.convert_to_df(boxObjects, image)
        return ocr_df