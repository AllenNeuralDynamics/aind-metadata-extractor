"""SmartSPIM extractor model"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel


class ImmersionModel(BaseModel):
    """Model for immersion medium configuration"""

    medium: Optional[str] = None
    refractive_index: Optional[float] = None


class AxesModel(BaseModel):
    """Model for axis configuration"""

    x: Optional[str] = None
    y: Optional[str] = None
    z: Optional[str] = None


class ProcessingStepsModel(BaseModel):
    """Model for processing steps"""

    channel_name: str
    process_name: List[str]


class ChannelModel(BaseModel):
    """Model for channel information in tiles"""

    channel_name: str
    light_source_name: str
    filter_names: List[str] = []
    detector_name: str = ""
    additional_device_names: List[str] = []
    excitation_wavelength: int
    excitation_wavelength_unit: str  # Will be SizeUnit.NM
    excitation_power: Union[int, float]
    excitation_power_unit: str  # Will be PowerUnit.PERCENT
    filter_wheel_index: int


class TileModel(BaseModel):
    """Model for tile information from make_acq_tiles function"""

    channel: ChannelModel
    notes: str
    coordinate_transformations: List[Dict[str, Union[int, float]]]
    file_name: str


class SmartspimModel(BaseModel):
    """SmartSPIM extractor model for intermediate data structure"""

    # Core identification
    specimen_id: str
    subject_id: str

    # Timing
    session_start_time: datetime
    session_end_time: Optional[datetime] = None

    # Hardware configuration
    instrument_id: Optional[str] = None
    active_objectives: Optional[List[str]] = None
    external_storage_directory: str = ""

    # Personnel
    experimenter_full_name: List[str] = []
    protocol_id: List[str] = []

    # Immersion settings
    chamber_immersion: Optional[ImmersionModel] = None
    sample_immersion: Optional[ImmersionModel] = None

    # Spatial configuration
    axes: Optional[AxesModel] = None

    # Processing information
    processing_steps: List[ProcessingStepsModel] = []

    # Tile information
    tiles: Optional[List[TileModel]] = None

    # Raw metadata containers
    file_metadata: Dict[str, Any] = {}
    slims_metadata: Dict[str, Any] = {}

    # Additional microscope metadata
    session_config: Optional[Dict[str, Any]] = None
    wavelength_config: Optional[Dict[str, Any]] = None
    tile_config: Optional[Dict[str, Any]] = None
    filter_mapping: Optional[Dict[str, Any]] = None
