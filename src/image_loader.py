import base64
from pathlib import Path
import logging

from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from .logging_config import get_logger

logger = get_logger(__name__)

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_image_description(image_path: str, llm_model: str) -> str:
    logger.debug("Getting image description for %s using model %s", image_path, llm_model)
    llm = ChatOpenAI(model=llm_model, temperature=0)
    base64_image = encode_image_to_base64(image_path)

    # Create multimodal message
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "Describe this image in detail, focusing on key visual elements, text, and context that would be useful for retrieval."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        ]
    )

    response = llm.invoke([message])
    return response.content

def load_image(image_path: Path, llm_model: str) -> Document:
    logger.info("Loading image: %s", image_path)
    document = None

    # Get image description
    suffix = image_path.suffix.lower()
    description = get_image_description(image_path, llm_model)

    # Create document with metadata
    document = Document(
        page_content=description,
        metadata = {"source": str(image_path.name), "path" : str(image_path), "file_ext": suffix, "file_type": "image"}
    )

    return document
