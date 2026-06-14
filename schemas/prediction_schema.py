from pydantic import BaseModel
from typing import List

class ManualInputSchema(BaseModel):
    # Minimal MVP payload accepting a list of dummy values
    dummy_features: List[float] = [0.0] * 25
