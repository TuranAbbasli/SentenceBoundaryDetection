from pydantic import BaseModel, RootModel
from typing import List

class Sbd_Output(BaseModel):
    results: list[str]

class AbbrvOutput(BaseModel):
    abbr: List[str]
    types: List[str]

class BatchAbbrvOutput(RootModel[List[AbbrvOutput]]):
    pass