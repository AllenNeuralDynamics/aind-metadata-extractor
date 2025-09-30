"""SmartSPIM extractor module"""

from datetime import datetime
import os
import re
from typing import Optional
from pathlib import Path

import requests

from aind_metadata_extractor.core import BaseExtractor
from aind_metadata_extractor.smartspim.job_settings import JobSettings
from aind_metadata_extractor.models.smartspim import SmartspimModel, FileMetadataModel, SlimsMetadataModel
from aind_metadata_extractor.smartspim.utils import get_excitation_emission_waves, get_session_end, read_json_as_dict

REGEX_DATE = r"(20[0-9]{2})-([0-9]{2})-([0-9]{2})_([0-9]{2})-" r"([0-9]{2})-([0-9]{2})"
REGEX_MOUSE_ID = r"([0-9]{6})"


class SmartspimExtractor(BaseExtractor):
    """Extractor for SmartSPIM metadata from microscope files and SLIMS."""

    def __init__(self, job_settings: JobSettings):
        """Initialize the SmartSPIM extractor with job settings."""
        self.metadata = None
        self.job_settings = job_settings

    def run_job(self) -> dict:
        """Run the extraction job."""
        self.metadata = self._extract()
        return self.metadata.model_dump()

    def _extract(self) -> SmartspimModel:
        """Run extraction process"""

        file_metadata = self._extract_metadata_from_microscope_files()
        slims_metadata = self._extract_metadata_from_slims()

        # Create model objects
        file_metadata_model = FileMetadataModel(**file_metadata)
        slims_metadata_model = SlimsMetadataModel(**slims_metadata)

        smartspim_metadata = SmartspimModel(
            acquisition_type=self.job_settings.acquisition_type,
            file_metadata=file_metadata_model,
            slims_metadata=slims_metadata_model,
        )

        return smartspim_metadata

    def _extract_metadata_from_microscope_files(self) -> dict:
        """
        Extracts metadata from the microscope metadata files.

        Returns
        -------
        Dict
            Dictionary containing metadata from
            the microscope for the current acquisition. This
            is needed to build the acquisition.json.
        """
        # Convert input_source to Path - handle various input types
        if isinstance(self.job_settings.input_source, (str, Path)):
            input_path = Path(self.job_settings.input_source)
        elif isinstance(self.job_settings.input_source, list) and len(self.job_settings.input_source) > 0:
            # Take the first path if it's a list
            input_path = Path(self.job_settings.input_source[0])
        else:
            raise ValueError("input_source must be a valid path or list of paths")

        # Path where the channels are stored
        smartspim_channel_root = input_path.joinpath("SmartSPIM")

        # Getting only valid folders
        channels = [
            folder
            for folder in os.listdir(smartspim_channel_root)
            if os.path.isdir(f"{smartspim_channel_root}/{folder}")
        ]

        # Path to metadata files
        asi_file_path_txt = input_path.joinpath(self.job_settings.asi_filename)

        mdata_path = input_path.joinpath(self.job_settings.mdata_filename_json)

        # ASI file does not exist, needed for acquisition
        if not asi_file_path_txt.exists():
            raise FileNotFoundError(f"File {asi_file_path_txt} does not exist")

        if not mdata_path.exists():
            raise FileNotFoundError(f"File {mdata_path} does not exist")

        # Getting acquisition metadata from the microscope
        metadata_info = read_json_as_dict(str(mdata_path))

        filter_mapping = get_excitation_emission_waves(channels)
        session_config = metadata_info["session_config"]
        wavelength_config = metadata_info["wavelength_config"]
        tile_config = metadata_info["tile_config"]

        if None in [session_config, wavelength_config, tile_config]:
            raise ValueError("Metadata json is empty")

        session_end_time = get_session_end(asi_file_path_txt)
        mdate_match = re.search(REGEX_DATE, input_path.stem)
        if not (mdate_match):
            raise ValueError("Error while extracting session date.")
        session_start = datetime.strptime(mdate_match.group(), "%Y-%m-%d_%H-%M-%S")

        metadata_dict = {
            "session_config": session_config,
            "wavelength_config": wavelength_config,
            "tile_config": tile_config,
            "session_start_time": session_start,
            "session_end_time": session_end_time,
            "filter_mapping": filter_mapping,
        }

        return metadata_dict

    def _extract_metadata_from_slims(
        self, start_date_gte: Optional[str] = None, end_date_lte: Optional[str] = None
    ) -> dict:
        """
        Method to retrieve smartspim imaging info from SLIMS
        using the metadata service endpoint.
        Parameters
        ----------
        start_date_gte: str
            Start date for the search.
        end_date_lte: str
            End date for the search.
        Returns
        -------
        Dict
            Dictionary containing metadata from SLIMS for an acquisition.
        """
        query_params = {"subject_id": self.job_settings.subject_id}
        query_params["start_date_gte"] = start_date_gte if start_date_gte else "2020-01-01"
        query_params["end_date_lte"] = end_date_lte if end_date_lte else "2100-01-01"

        response = requests.get(
            f"{self.job_settings.metadata_service_path}",
            params=query_params,
        )
        response.raise_for_status()
        response_data = response.json().get("data", [])
        if response.status_code == 200 and len(response_data) > 1:
            raise ValueError(
                "More than one imaging session found for the same subject_id. " "Please refine your search."
            )
        elif response.status_code == 200 and len(response_data) == 1:
            imaging_info = response_data[0]
        else:
            raise ValueError("No imaging session found for the given subject_id and date range.")
        return imaging_info
