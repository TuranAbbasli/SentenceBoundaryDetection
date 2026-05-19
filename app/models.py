from pydantic import BaseModel

class SegmentationRequest(BaseModel):
    text: str

class SegmentationResponse(BaseModel):
    sentences: list[str]
    num_sentences: int
