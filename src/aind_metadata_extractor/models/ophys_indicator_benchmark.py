"""Ophys Indicator Benchmark metadata model"""

from pydantic import BaseModel, Field
from aind_metadata_extractor.models.fip import FiberData


class OptoModel(BaseModel):
    """Ophys Indicator Benchmark model for extracting metadata."""

    opto_metadata: dict = Field(
        ...,
        title="Optogenetics metadata",
        description="Metadata for Optogenetics",
    )

    stimulus_epochs: dict = Field(
        ...,
        title="Optogenetics Stimulus Epochs",
        description="Optogenetics stimulus epoch information",
    )


class OphysIndicatorBenchmarkModel(BaseModel):
    """Intermediate data structure"""

    opto_data: OptoModel
    fiber_data: FiberData
