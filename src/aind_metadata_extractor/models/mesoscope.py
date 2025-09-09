"""Mesoscope metadata model"""

from pydantic import BaseModel, Field


class MesoscopeExtractModel(BaseModel):
    """Mesoscope model for extracting metadata."""

    tiff_header: list = Field(title="Header information from TIFF files")
    session_metadata: dict = Field(title="Metadata extracted from the session platform JSON and other data")
    camstim_epchs: list = Field(title="List of epochs from the Camstim platform")
    camstim_session_type: str = Field(title="Type of session from the Camstim platform")
    job_settings: dict = Field(title="Job settings used for extraction")
