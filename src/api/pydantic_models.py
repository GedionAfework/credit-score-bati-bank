from pydantic import BaseModel
from typing import Optional


class CustomerData(BaseModel):
    Amount_sum: float
    Amount_mean: float
    Amount_count: int
    Amount_std: float
    Value_sum: float
    Value_mean: float
    Value_count: int
    Value_std: float
    woe_Amount: Optional[float] = 0.0
    woe_Value: Optional[float] = 0.0
    woe_TransactionHour: Optional[float] = 0.0
    woe_TransactionDay: Optional[float] = 0.0
    woe_TransactionMonth: Optional[float] = 0.0
    woe_ProductCategory: Optional[float] = 0.0
    woe_ChannelId: Optional[float] = 0.0
    woe_PricingStrategy: Optional[float] = 0.0


class PredictionResponse(BaseModel):
    risk_probability: float
    credit_score: float
    loan_amount: float
    loan_duration: int
