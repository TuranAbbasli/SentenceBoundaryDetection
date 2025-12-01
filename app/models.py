from pydantic import BaseModel


class SegmentationResponse(BaseModel):
    sentences: list[str]
    num_sentences: int
