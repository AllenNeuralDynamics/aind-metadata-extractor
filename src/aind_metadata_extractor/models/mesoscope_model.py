"""Mesoscope metadata model"""

from pydanitic import BaseSettings, Field

class MesoscopeExtractModel(BaseSettings):
    """Mesoscope model for extracting metadata."""

    tiff_header: dict = Field(title="Header information from TIFF files")
    session_metadata: dict = Field(
        title="Metadata extracted from the session platform JSON and other data")
    camstim_epchs: list = Field(
        title="List of epochs from the Camstim platform")
    camstim_session_type: str = Field(
        title="Type of session from the Camstim platform")
    