"""Fiber Photometry extractor module using data contract"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING, cast
from zoneinfo import ZoneInfo

from aind_behavior_services.session import AindBehaviorSessionModel
from aind_metadata_extractor.fip.job_settings import JobSettings
from aind_metadata_extractor.models.fip import FIPDataModel as FiberData
from aind_physiology_fip.data_contract import dataset
from aind_physiology_fip.rig import AindPhysioFipRig
from contraqctor.contract import Dataset, FilePathBaseParam

if TYPE_CHECKING:
    from pandas import DataFrame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FiberPhotometryExtractor:
    """Extractor for Fiber Photometry metadata using data contract."""

    def __init__(self, job_settings: JobSettings):
        """
        Initialize the Fiber Photometry extractor with job settings.

        Parameters
        ----------
        job_settings : JobSettings
            Configuration settings for the extraction process
        """
        self.job_settings = job_settings
        self._dataset: Optional[Dataset] = None

    @property
    def dataset(self) -> Dataset:
        # This is a read-only property to access the dataset and ensures it is not null for type hinting
        if self._dataset is None:
            raise ValueError("Dataset has not been initialized.")
        return self._dataset

    def extract(self) -> dict:
        """Run extraction process using the GitHub data contract.

        Uses the official data contract from:
        https://github.com/AllenNeuralDynamics/FIP_DAQ_Control/blob/bc-major-refactor/src/aind_physiology_fip/data_contract.py

        Returns
        -------
        dict
            Extracted metadata as a dictionary
        """

        self._dataset = dataset(self.job_settings.data_directory)

        # Extract metadata using the data contract
        file_metadata = self._extract_metadata_from_contract()

        # Map extracted start_time/end_time to session_start_time/session_end_time
        if "start_time" in file_metadata:
            file_metadata["session_start_time"] = file_metadata.pop("start_time")
        if "end_time" in file_metadata:
            file_metadata["session_end_time"] = file_metadata.pop("end_time")

        # Extract rig_id from rig_config if available
        if "rig_config" in file_metadata and file_metadata["rig_config"]:
            if "rig_name" in file_metadata["rig_config"]:
                file_metadata["rig_id"] = file_metadata["rig_config"]["rig_name"]

        # Extract subject_id from session_config if available
        if "session_config" in file_metadata and file_metadata["session_config"]:
            if "subject" in file_metadata["session_config"]:
                file_metadata["subject_id"] = file_metadata["session_config"]["subject"]
            if "experimenter" in file_metadata["session_config"]:
                file_metadata["experimenter_full_name"] = file_metadata["session_config"]["experimenter"]

        # Update with job settings, but don't overwrite extracted values
        job_settings_dict = self.job_settings.model_dump()
        for key in ["rig_config", "session_config", "rig_id", "subject_id", "experimenter_full_name"]:
            if key in file_metadata and file_metadata[key] is not None:
                job_settings_dict.pop(key, None)
        file_metadata.update(job_settings_dict)

        logger.info("Extracted metadata from data contract:")
        logger.info(json.dumps(file_metadata, indent=3, default=str))

        # Create the fiber data model
        fiber_data = FiberData.model_validate(file_metadata)

        return fiber_data.model_dump()

    def _extract_metadata_from_contract(self) -> dict:
        """
        Extract metadata using the data contract approach.

        Returns
        -------
        dict
            Extracted metadata as a dictionary
        """
        metadata: dict[str, Any] = {}

        metadata["start_time"], metadata["end_time"] = self._extract_timing_from_csv()
        metadata["data_files"] = self._extract_data_files()
        metadata["session_config"], metadata["rig_config"] = self._extract_hardware_config()

        return metadata

    def _extract_timing_from_csv(self) -> tuple[datetime, datetime]:
        """
        Extract session timing from camera metadata CSV files.

        Uses CpuTime column which contains timezone-aware ISO 8601 timestamps,
        and converts them to the local timezone specified in job_settings.

        Returns
        -------
        tuple[datetime, datetime]
            Start and end time of the session as timezone-aware datetime objects
        """

        local_tz = ZoneInfo(self.job_settings.local_timezone)
        _known_streams = ["camera_green_iso_metadata", "camera_red_metadata"]
        # Try to get timing from camera_green_iso_metadata stream
        for stream in _known_streams:
            logger.debug(f"Checking for timing in stream: {stream}")
            metadata_stream = cast(DataFrame, self.dataset[stream].read())
            if metadata_stream is not None and not metadata_stream.empty:
                start_utc = datetime.fromisoformat(metadata_stream["CpuTime"].iloc[0])
                end_utc = datetime.fromisoformat(metadata_stream["CpuTime"].iloc[-1])
                return start_utc.astimezone(local_tz), end_utc.astimezone(local_tz)

        raise ValueError(
            "Could not extract session timing from camera metadata. "
            "Expected to find CpuTime column in camera_green_iso_metadata.csv or camera_red_metadata.csv. "
            "Please verify that camera metadata files exist in the data directory."
        )

    def _extract_data_files(self) -> list[str]:
        """
        Extract data files information from the dataset.

        Returns
        -------
        list[str]
            Extracted data files information. Each entry is a file path string.
        """
        data_files = []

        # Get all data streams that represent files
        for stream_name in [
            "raw_green",
            "raw_red",
            "raw_iso",
            "green",
            "red",
            "iso",
        ]:
            stream = self.dataset[stream_name]
            assert isinstance(stream.reader_params, FilePathBaseParam)
            if Path(stream.reader_params.path).exists():
                data_files.append(str(stream.reader_params.path))
            else:
                logger.warning(f"Data file for stream '{stream_name}' does not exist: {stream.reader_params.path}")

        return data_files

    def _extract_hardware_config(self) -> tuple[AindPhysioFipRig, AindBehaviorSessionModel]:
        """
        Extract hardware configuration from rig and session inputs.

        Returns
        -------
        tuple[AindPhysioFipRig, AindBehaviorSessionModel]
            Extracted rig configuration and session configuration
        """

        # Try to extract rig configuration
        rig_config = self.dataset["rig_input"].read()
        assert isinstance(rig_config, AindPhysioFipRig)

        session_config = self.dataset["session_input"].read()
        assert isinstance(session_config, AindBehaviorSessionModel)

        return rig_config, session_config

    def save_to_file(self, fiber_data: FiberData, output_path: Optional[Path] = None) -> Path:
        """Save FiberData to a JSON file.

        Parameters
        ----------
        fiber_data : FiberData
            The fiber data to save
        output_path : Optional[Path]
            Path where to save the file. If None, saves to the data directory
            with the output filename from job settings.

        Returns
        -------
        Path
            Path where the file was saved
        """
        if output_path is None:
            output_dir = Path(self.job_settings.data_directory)
            output_path = output_dir / self.job_settings.output_filename

        with open(output_path, "w") as f:
            f.write(fiber_data.model_dump_json(indent=3))

        return output_path
