import ollama

VLM_MODEL_NAME = "llava"


def read_image_segments(image_path):
  
    response = ollama.chat(
        model=VLM_MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": (
                    "Describe everything visible in this image in detail. "
                    "Include any text, labels, numbers, or headings you can "
                    "read, and explain how different parts of the image "
                    "relate to each other (e.g. boxes, arrows, sections, "
                    "flow). Be thorough."
                ),
                "images": [image_path]
            }
        ],
        options={"num_predict": 500}
    )

    description = response["message"]["content"]
    return [{"text": description, "type": "IMAGE"}]