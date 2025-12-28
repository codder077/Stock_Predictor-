from pydantic import BaseModel, Field
from typing import List, Optional

class PredictionRequest(BaseModel):
    data_lag1: float = Field(..., description="The 'Data' value from yesterday")
    data_change_prev_day: float = Field(..., description="Change in 'Data' (Yesterday - DayBefore)")
    data_rolling_mean: float = Field(..., description="5-day moving average of 'Data'")

    class Config:
        schema_extra = {
            "example": {
                "data_lag1": 2.35,
                "data_change_prev_day": -0.015,
                "data_rolling_mean": 2.38
            }
        }

    # Optional class: can be added if we want sample values to appear instead generic text/numbers
    # class Config:
    #     schema_extra = {
    #         "example": {
    #             "data_lag1": 2.35,
    #             "data_change_prev_day": -0.015,
    #             "data_rolling_mean": 2.38
    #         }
    #     }

class PredictionResponse(BaseModel):
    predicted_price_change: float
    message: str