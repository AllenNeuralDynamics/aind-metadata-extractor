"""Ophys Indicator Benchmark job settings configuration."""

from pathlib import Path
from typing import List, Optional

from pydantic import Field

from aind_metadata_extractor.core import BaseJobSettings


class JobSettings(BaseJobSettings):
    """Parameters for extracting from raw data."""

    data_directory: Path = Field(
        ...,
        description="Path to data directory",
    )

    # Optogenetics parameters
    stimulus_name: str
    pulse_shape: str = Field(..., title="Pulse shape")
    pulse_frequency: List[float] = Field(..., title="Pulse frequency (Hz)")
    number_pulse_trains: List[int] = Field(..., title="Number of pulse trains")
    pulse_width: List[int] = Field(..., title="Pulse width (ms)")

    pulse_train_duration: List[float] = Field(
        ..., title="Pulse train duration (s)"
    )
    fixed_pulse_train_interval: bool = Field(
        ..., title="Fixed pulse train interval"
    )
    pulse_train_interval: Optional[float] = Field(
        default=None,
        title="Pulse train interval (s)",
        description="Time between pulse trains",
    )
    baseline_duration: float = Field(
        ...,
        title="Baseline duration (s)",
        description="Duration of baseline recording prior to first pulse train",
    )

    # Stimulus epoch laser configs
    wavelength: int = Field(..., title="Wavelength (nm)")
    power: float = Field(..., title="Excitation power")
