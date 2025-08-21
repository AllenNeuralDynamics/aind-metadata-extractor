

from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import Field
from aind_metadata_extractor.core import BaseJobSettings


class JobSettings(BaseJobSettings):
    """Data to be entered by the user."""

    # Field can be used to switch between different acquisition etl jobs
    job_settings_name: Literal["SmartSPIM"] = "SmartSPIM"
    raw_dataset_path: Optional[Union[Path, str]] = Field(
        default=None, description=("Deprecated, use input_source instead.")
    )
    subject_id: str

    # Metadata names
    asi_filename: str = Field(
        default="derivatives/ASI_logging.txt",
        description="Path to ASI logging file.",
    )
    mdata_filename_json: str = Field(
        default="derivatives/metadata.json",
        description=(
            "Path to metadata file, expected to be a .json or .txt file."
        ),
    )
    # Fetch info provided by microscope operators in SLIMS
    processing_manifest_path: Optional[Union[Path, str]] = Field(
        default="derivatives/processing_manifest.json",
        description=("Deprecated, use metadata_service_path instead."),
    )
    metadata_service_path: str

    # Optional field for SLIMS datetime
    slims_datetime: Optional[str] = Field(
        default=None,
        description=(
            "Datetime of the SLIMS entry, if not provided, the datetime "
            "window will be extracted from the metadata file."
        ),
    )
