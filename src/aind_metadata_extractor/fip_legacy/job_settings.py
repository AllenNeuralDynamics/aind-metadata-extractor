"""Fiber Photometry job settings configuration."""

from pathlib import Path
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field


class JobSettings(BaseModel):
    """Data to be entered by the user."""

    # Field can be used to switch between different acquisition etl jobs
    job_settings_name: Literal["FiberPhotometry"] = "FiberPhotometry"

    # Required fields
    subject_id: str
    rig_id: str
    iacuc_protocol: str
    notes: str

    # Session metadata
    experimenter_full_name: List[str] = Field(default_factory=list, description="List of experimenter names")
    session_type: str = Field(default="FIB", description="Type of session")
    mouse_platform_name: Optional[str] = Field(default=None, description="Name of the mouse platform used")
    active_mouse_platform: bool = Field(
        default=False, description="Whether the mouse platform was active during the session"
    )

    # Optional session details
    anaesthesia: Optional[str] = Field(default=None, description="Anaesthesia used")
    animal_weight_post: Optional[float] = Field(default=None, description="Animal weight after session")
    animal_weight_prior: Optional[float] = Field(default=None, description="Animal weight before session")

    # Data configuration
    data_streams: List[dict] = Field(default_factory=list, description="List of data stream configurations")
    protocol_id: List[str] = Field(default_factory=list, description="List of protocol identifiers")

    # File paths
    data_directory: Optional[Union[str, Path]] = Field(
        default=None, description="Path to data directory containing fiber photometry files"
    )
    output_directory: Optional[Union[str, Path]] = Field(
        default=None, description="Output directory for generated files"
    )
    output_filename: str = Field(default="session_fip.json", description="Name of output file")

    # Timing configuration
    local_timezone: str = Field(default="America/Los_Angeles", description="Timezone for the session")
