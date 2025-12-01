from pydantic import BaseModel


class SegmentationResponse(BaseModel):
    original_text: str
    segmented_text: str
    num_boundaries: int
