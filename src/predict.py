from fastapi import FastAPI, HTTPException, status
import uvicorn

from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import List, Literal, Dict
import numpy as np
import pickle


class NetworkMetrics(BaseModel):
    """Pydantic model for selected network metrics data with validation."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "dl_mcs": [4.12121, 0.181818, 0.0],
                "ul_sinr": [34.0305, 32.3918, 0.0],
                "tx_brate_downlink_mbps": [0.009184, 0.00276305, 0.0],
                "dl_buffer_bytes": [0, 461, 4],
                "ul_turbo_iters": [1.0, 0.0, 2.5],
                "dl_cqi": [8.0, 5.66667, 7.0],
                "ul_n_samples": [42, 14, 0],
                "network_load": ["night", "night", "night"],
                "mcs_sinr_ratio": [0.121, 0.0056, 0.0],
                "grant_ratio": [1.76, 0.5, 0.0],
                "prb_efficiency": [0.0003, 0.00009, 0.0],
                "latency_proxy": [0.0, 461.0, 4.0]
            }
        }
    )
    
    dl_mcs: List[float] = Field(
        ..., 
        description="Downlink Modulation and Coding Scheme"
    )
    ul_sinr: List[float] = Field(
        ..., 
        description="Uplink Signal-to-Interference-plus-Noise Ratio"
    )
    tx_brate_downlink_mbps: List[float] = Field(
        ..., 
        description="Downlink transmission bitrate in Mbps"
    )
    dl_buffer_bytes: List[int] = Field(
        ..., 
        description="Downlink buffer size in bytes"
    )
    ul_turbo_iters: List[float] = Field(
        ..., 
        description="Uplink turbo iterations"
    )
    dl_cqi: List[float] = Field(
        ..., 
        description="Downlink Channel Quality Indicator"
    )
    ul_n_samples: List[int] = Field(
        ..., 
        description="Number of uplink samples"
    )
    network_load: List[str] = Field(
        ..., 
        description="Network load category: 'peak', 'off-peak', or 'night'"
    )
    mcs_sinr_ratio: List[float] = Field(
        ...,
        description="Ratio of MCS to SINR"
    )
    grant_ratio: List[float] = Field(
        ...,
        description="Ratio of granted PRBs to requested PRBs"
    )
    prb_efficiency: List[float] = Field(
        ...,
        description="PRB efficiency metric"
    )
    latency_proxy: List[float] = Field(
        ...,
        description="Latency proxy metric"
    )
    
    @field_validator('dl_mcs')
    @classmethod
    def validate_dl_mcs(cls, v):
        """Validate DL MCS values are within valid range (0-28)."""
        for val in v:
            if val < 0 or val > 28:
                raise ValueError(f'dl_mcs must be between 0 and 28, got {val}')
        return v
    
    @field_validator('ul_sinr')
    @classmethod
    def validate_ul_sinr(cls, v):
        """Validate UL SINR values are within reasonable range (-10 to 40 dB)."""
        for val in v:
            if val < -10 or val > 40:
                raise ValueError(f'ul_sinr must be between -10 and 40 dB, got {val}')
        return v
    
    @field_validator('tx_brate_downlink_mbps')
    @classmethod
    def validate_tx_brate(cls, v):
        """Validate downlink bitrate is non-negative."""
        for val in v:
            if val < 0:
                raise ValueError(f'tx_brate_downlink_mbps must be non-negative, got {val}')
        return v
    
    @field_validator('dl_buffer_bytes')
    @classmethod
    def validate_dl_buffer(cls, v):
        """Validate buffer bytes is non-negative."""
        for val in v:
            if val < 0:
                raise ValueError(f'dl_buffer_bytes must be non-negative, got {val}')
        return v
    
    @field_validator('ul_turbo_iters')
    @classmethod
    def validate_turbo_iters(cls, v):
        """Validate turbo iterations are within valid range (0-10)."""
        for val in v:
            if val < 0 or val > 10:
                raise ValueError(f'ul_turbo_iters must be between 0 and 10, got {val}')
        return v
    
    @field_validator('dl_cqi')
    @classmethod
    def validate_dl_cqi(cls, v):
        """Validate CQI values are within valid range (0-15)."""
        for val in v:
            if val < 0 or val > 15:
                raise ValueError(f'dl_cqi must be between 0 and 15, got {val}')
        return v
    
    @field_validator('ul_n_samples')
    @classmethod
    def validate_ul_n_samples(cls, v):
        """Validate sample count is non-negative."""
        for val in v:
            if val < 0:
                raise ValueError(f'ul_n_samples must be non-negative, got {val}')
        return v
    
    @field_validator('network_load')
    @classmethod
    def validate_network_load(cls, v):
        """Validate network_load values are one of the allowed categories."""
        valid_loads = {'peak', 'off-peak', 'night'}
        for val in v:
            if val not in valid_loads:
                raise ValueError(f'network_load must be one of {valid_loads}, got {val}')
        return v
    
    @field_validator('mcs_sinr_ratio')
    @classmethod
    def validate_mcs_sinr_ratio(cls, v):
        """Validate MCS/SINR ratio is non-negative."""
        for val in v:
            if val < 0:
                raise ValueError(f'mcs_sinr_ratio must be non-negative, got {val}')
        return v
    
    @field_validator('grant_ratio')
    @classmethod
    def validate_grant_ratio(cls, v):
        """Validate grant ratio is non-negative."""
        for val in v:
            if val < 0:
                raise ValueError(f'grant_ratio must be non-negative, got {val}')
        return v
    
    @field_validator('prb_efficiency')
    @classmethod
    def validate_prb_efficiency(cls, v):
        """Validate PRB efficiency is non-negative."""
        for val in v:
            if val < 0:
                raise ValueError(f'prb_efficiency must be non-negative, got {val}')
        return v
    
    @field_validator('latency_proxy')
    @classmethod
    def validate_latency_proxy(cls, v):
        """Validate latency proxy is non-negative."""
        for val in v:
            if val < 0:
                raise ValueError(f'latency_proxy must be non-negative, got {val}')
        return v


class NetworkMetricsSingle(BaseModel):
    """Pydantic model for a single network metrics record with validation."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "dl_mcs": 4.12121,
                "ul_sinr": 34.0305,
                "tx_brate_downlink_mbps": 0.009184,
                "dl_buffer_bytes": 0,
                "ul_turbo_iters": 1.0,
                "dl_cqi": 8.0,
                "ul_n_samples": 42,
                "network_load": "night",
                "mcs_sinr_ratio": 0.121,
                "grant_ratio": 1.76,
                "prb_efficiency": 0.0003,
                "latency_proxy": 0.0
            }
        }
    )
    
    dl_mcs: float = Field(
        ..., 
        ge=0, 
        le=28,
        description="Downlink Modulation and Coding Scheme (0-28)"
    )
    ul_sinr: float = Field(
        ..., 
        ge=-10, 
        le=40,
        description="Uplink Signal-to-Interference-plus-Noise Ratio (-10 to 40 dB)"
    )
    tx_brate_downlink_mbps: float = Field(
        ..., 
        ge=0,
        description="Downlink transmission bitrate in Mbps"
    )
    dl_buffer_bytes: int = Field(
        ..., 
        ge=0,
        description="Downlink buffer size in bytes"
    )
    ul_turbo_iters: float = Field(
        ..., 
        ge=0, 
        le=10,
        description="Uplink turbo iterations (0-10)"
    )
    dl_cqi: float = Field(
        ..., 
        ge=0, 
        le=15,
        description="Downlink Channel Quality Indicator (0-15)"
    )
    ul_n_samples: int = Field(
        ..., 
        ge=0,
        description="Number of uplink samples"
    )
    network_load: Literal['peak', 'off-peak', 'night'] = Field(
        ..., 
        description="Network load category"
    )
    mcs_sinr_ratio: float = Field(
        ...,
        ge=0,
        description="Ratio of MCS to SINR"
    )
    grant_ratio: float = Field(
        ...,
        ge=0,
        description="Ratio of granted PRBs to requested PRBs"
    )
    prb_efficiency: float = Field(
        ...,
        ge=0,
        description="PRB efficiency metric"
    )
    latency_proxy: float = Field(
        ...,
        ge=0,
        description="Latency proxy metric"
    )


app = FastAPI()

class PredictResponse(BaseModel):
    slice_category: str # Predicted class (Slice 0: eMBB UEs, Slice 1: MTC UEs, Slice 2: URLLC UEs)
    probabilities: Dict[str, float]  # Dictionary of class probabilities

# Prediction function
def predict_slice(customer ,pipeline) -> (np.ndarray, np.ndarray):
    y_pred = pipeline.predict(customer)  # Wrap it in a list to match the input shape
    y_pred_proba = pipeline.predict_proba(customer)
    return y_pred, y_pred_proba

@app.post("/predict", response_model=PredictResponse)
def predict_networkslice_category(customer: NetworkMetricsSingle) -> PredictResponse:
    try:
        ypred, y_pred_proba = predict_slice(customer.model_dump(), pipeline)

        classes = ['eMBB enhanced MobileBroadBand', 'MTC MassTransport Communication' ,'URLLC Ultra Reliable LowLatency Communication']

        if isinstance(ypred[0], str):  # If it's already a class label
            predicted_class = ypred[0]  # Directly use the class label
        else:  # If it's an index, map it to the class label
            predicted_class = classes[ypred[0]]

        class_probabilities = {
            classes[i]: round(float(y_pred_proba[0][i] * 100), 4) for i in range(len(classes))
        }

        response = PredictResponse(
            slice_category=predicted_class,
            probabilities=class_probabilities
        )

        return response

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input data: {str(e)}"
        )

# Load the pre-trained model
def load_model():
    with open('model/xgboost.bin', 'rb') as f_in:
        pipeline = pickle.load(f_in)
    return pipeline

# Load the model when the app starts
pipeline = load_model()

if __name__ == "__main__":
    uvicorn.run(app,host="0.0.0.0",port=8080)
