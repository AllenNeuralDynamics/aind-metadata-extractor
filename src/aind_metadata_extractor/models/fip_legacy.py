from typing import Literal, Union, Optional, List
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field

class FiberData(BaseModel):
    """
    Intermediate data model for fiber photometry data.

    This model holds the extracted and processed data before final
    transformation into a Session object. It serves as a structured
    intermediate representation of the fiber photometry session data.
    """

    job_settings_name: Literal["FiberPhotometry"] = Field(default="FiberPhotometry", description="Name of the job settings type")
    experimenter_full_name: List[str] = Field(..., description="List of experimenter names")
    session_start_time: Optional[datetime] = Field(None, description="Start time of the session")
    session_end_time: Optional[datetime] = Field(None, description="End time of the session")
    subject_id: str = Field(..., description="Subject identifier")
    rig_id: str = Field(..., description="Identifier for the experimental rig")
    mouse_platform_name: str = Field(..., description="Name of the mouse platform used")
    active_mouse_platform: bool = Field(..., description="Whether the mouse platform was active during the session")
    data_streams: List[dict] = Field(..., description="List of data stream configurations")
    session_type: str = Field(default="FIB", description="Type of session")
    iacuc_protocol: str = Field(..., description="IACUC protocol identifier")
    notes: str = Field(..., description="Session notes")
    anaesthesia: Optional[str] = Field(None, description="Anaesthesia used")
    animal_weight_post: Optional[float] = Field(None, description="Animal weight after session")
    animal_weight_prior: Optional[float] = Field(None, description="Animal weight before session")
    protocol_id: List[str] = Field(default_factory=list, description="List of protocol identifiers")
    data_directory: Optional[Union[str, Path]] = Field(None, description="Path to data directory containing fiber photometry files")
    local_timezone: str = Field(default="America/Los_Angeles", description="Timezone for the session")
    output_directory: Optional[Union[str, Path]] = Field(None, description="Output directory for generated files")
    output_filename: str = Field(default="session_fip.json", description="Name of output file")