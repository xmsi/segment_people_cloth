from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import requests
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np

import gradio as gr

processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

classes = {0: "Background", 1: "Hat", 2: "Hair", 3: "Sunglasses", 4: "Upper-clothes", 5: "Skirt", 6: "Pants", 7: "Dress", 8: "Belt", 9: "Left-shoe", 10: "Right-shoe", 11: "Face", 12: "Left-leg", 13: "Right-leg", 14: "Left-arm", 15: "Right-arm", 16: "Bag", 17: "Scarf"}


def segment(image): 
    image_array = Image.fromarray(image.astype('uint8'))
    inputs = processor(images=image_array, return_tensors="pt")

    outputs = model(**inputs)
    logits = outputs.logits.cpu()

    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image_array.size[::-1],
        mode="bilinear",
        align_corners=False,
    )

    pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
    mask = np.vectorize(classes.get)(pred_seg)
    segmented = [(mask == cls, cls) for cls in np.unique(mask)]
    return (image, segmented)

webcam = gr.inputs.Image(shape=(640, 480), source="webcam")
demo = gr.Interface(fn=segment, 
                    inputs=webcam, 
                    outputs=gr.AnnotatedImage(),
                    title='Segment people + clothes',
                    description='model: Segformer B2. Hint: hover on classes uder output image. Thanks for the entire open source community!')
    
demo.launch(server_name="0.0.0.0", server_port=7000)   

