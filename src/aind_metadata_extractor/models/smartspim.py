"""SmartSPIM extractor model"""

from datetime import datetime
from typing import Dict, List, Optional, Any

from pydantic import BaseModel


class FileMetadataModel(BaseModel):
    """Model for metadata extracted from microscope files"""

    session_config: Dict[str, Any]
    wavelength_config: Dict[str, Any]
    tile_config: Dict[str, Any]
    session_start_time: datetime
    session_end_time: datetime
    filter_mapping: Dict[str, Any]


class SlimsMetadataModel(BaseModel):
    """Model for metadata extracted from SLIMS API response"""

    experiment_run_created_on: Optional[str] = None
    order_created_by: Optional[str] = None
    order_project_id: Optional[str] = None
    specimen_id: Optional[str] = None
    subject_id: Optional[str] = None
    protocol_name: Optional[str] = None
    protocol_id: Optional[str] = None
    date_performed: Optional[str] = None
    chamber_immersion_medium: Optional[str] = None
    sample_immersion_medium: Optional[str] = None
    chamber_refractive_index: Optional[str] = None
    sample_refractive_index: Optional[str] = None
    instrument_id: Optional[str] = None
    experimenter_name: Optional[str] = None
    z_direction: Optional[str] = None
    y_direction: Optional[str] = None
    x_direction: Optional[str] = None
    imaging_channels: Optional[List[str]] = None
    stitching_channels: Optional[str] = None
    ccf_registration_channels: Optional[str] = None
    cell_segmentation_channels: Optional[str] = None


class SmartspimModel(BaseModel):
    """SmartSPIM extractor model for intermediate data structure"""

    acquisition_type: str
    file_metadata: FileMetadataModel
    slims_metadata: SlimsMetadataModel
