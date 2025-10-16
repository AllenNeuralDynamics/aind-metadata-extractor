"""Module defining JobSettings for Fiber Photometry Contract-based Extractor"""

from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional, Union

from pydantic import Field, field_validator
from aind_metadata_extractor.core import BaseJobSettings


class JobSettings(BaseJobSettings):
    """Data to be entered by the user for Fiber Photometry."""

    job_settings_name: Literal["FIP"] = Field(default="FIP", title="Name of the job settings")
    data_directory: Union[str, Path] = Field(..., title="Path to data directory containing fiber photometry files")
    experimenter_full_name: List[str] = Field(default_factory=list, title="List of experimenter names")
    subject_id: Optional[str] = Field(default=None, title="Subject identifier")
    rig_id: Optional[str] = Field(default=None, title="Identifier for the experimental rig")
    mouse_platform_name: Optional[str] = Field(default=None, title="Name of the mouse platform used")
    active_mouse_platform: bool = Field(
        default=False, title="Whether the mouse platform was actively controlled (motor-driven) during the session"
    )
    data_streams: Optional[List[dict]] = Field(default_factory=list, title="List of data stream configurations")
    iacuc_protocol: Optional[str] = Field(default=None, title="IACUC protocol identifier")
    notes: Optional[str] = Field(default=None, title="Session notes")
    anaesthesia: Optional[str] = Field(default=None, title="Anaesthesia used")
    animal_weight_post: Optional[float] = Field(default=None, title="Animal weight after session")
    animal_weight_prior: Optional[float] = Field(default=None, title="Animal weight before session")
    protocol_id: Optional[List[str]] = Field(default_factory=list, title="List of protocol identifiers")
    session_type: Optional[str] = Field(default="FIP", title="Type of session")
    data_files: Optional[List[str]] = Field(default_factory=list, title="List of data file paths")
    rig_config: Optional[dict] = Field(default=None, title="Rig configuration dictionary")
    session_config: Optional[dict] = Field(default=None, title="Session configuration dictionary")
    local_timezone: str = Field(default="America/Los_Angeles", title="Timezone for the session")
    output_directory: Optional[Union[str, Path]] = Field(
        default=None, title="Output directory for generated files (defaults to data_directory)"
    )
    output_filename: str = Field(default="session_fip.json", title="Name of output file")

    @field_validator("data_directory", "output_directory", mode="before")
    @classmethod
    def validate_path_is_dir(cls, v):
        """Validate that the path is a directory if not None."""
        if v is not None and not Path(v).is_dir():
            raise ValueError(f"{v} is not a directory")
        return v
