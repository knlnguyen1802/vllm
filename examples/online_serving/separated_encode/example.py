#!/usr/bin/env python3
# pip install vllm pillow

import base64
import io
from PIL import Image
from vllm import LLM, SamplingParams

# --------------------------------------------------------------------------- #
#                       Helper: JPEG → base-64 encoder                        #
# --------------------------------------------------------------------------- #
def encode_image(img: Image.Image) -> str:
    """Return a base-64 string of the input Pillow image."""
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

#temperature=0.0
sampling_params = SamplingParams(
    
)
# --------------------------------------------------------------------------- #
#                             Load the single image                           #
# --------------------------------------------------------------------------- #
IMAGE_PATH = "/workspace/n0090/epd/cat38.jpg"
img = Image.open(IMAGE_PATH)

# --------------------------------------------------------------------------- #
#                           Initialise the vLLM engine                        #
# --------------------------------------------------------------------------- #
llm = LLM(
    "/workspace/models/Qwen2.5-VL-3B-Instruct",
    limit_mm_per_prompt={"image": 1},   # only ONE image is allowed
    enforce_eager=True,
)

# --------------------------------------------------------------------------- #
#                         Build the multimodal request                        #
# --------------------------------------------------------------------------- #
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe what you see in this image."},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image(img)}"
            },
        },
    ],
}

# --------------------------------------------------------------------------- #
#                                 Inference                                   #
# --------------------------------------------------------------------------- #
outputs = llm.chat([message],sampling_params=sampling_params)
print(outputs[0].outputs[0].text)