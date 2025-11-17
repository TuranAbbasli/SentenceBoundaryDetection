from pydantic import BaseModel

class Sbd_Output(BaseModel):
    results: list[str]