from pydantic import BaseModel, Field


class CustomerData(BaseModel):
    """Schema for customer data input."""
    age: int = Field(..., description="Customer age")
    gender: str = Field(..., pattern="^(Male|Female)$", description="Customer gender")
    tenure: int = Field(..., description="Months as a customer")
    usage_frequency: int = Field(..., description="Service usage frequency")
    support_calls: int = Field(..., ge=0, le=10, description="Number of support calls")
    payment_delay: int = Field(..., ge=0, le=30, description="Payment delay in days")
    subscription_type: str = Field(
        ..., 
        pattern="^(Basic|Standard|Premium)$",
        description="Type of subscription"
    )
    contract_length: str = Field(
        ..., 
        pattern="^(Monthly|Quarterly|Annual)$",
        description="Length of contract"
    )
    total_spend: float = Field(
        ..., 
        description="Total amount spent"
    )
    last_interaction: int = Field(
        ..., 
        description="Days since last interaction"
    )


class PredictionResponse(BaseModel):
    """Schema for prediction response."""
    churn_probability: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Probability of customer churn"
    )
    is_likely_to_churn: bool = Field(
        ...,
        description="True if customer is likely to churn (probability > 0.5)"
    )
