"""Ophys Indicator Benchmark metadata model"""
from typing import Optional, List
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field


class OptoModel(BaseModel):
    """Ophys Indicator Benchmark model for extracting metadata."""

    opto_metadata: dict = Field(..., title="Optogenetics metadata", 
                                description="Metadata for Optogenetics")

    stimulus_epochs: dict = Field(..., title="Optogenetics Stimulus Epochs",
                                  description="Optogenetics stimulus epoch information")

class OphysIndicatorBenchmarkModel(BaseModel):
    """Intermediate data structure"""
    opto_data: OptoModel