"""Module for extracting Fiber Photometry session metadata.

This module provides functionality for extracting and structuring fiber
photometry experiment data into a standardized format. It handles:

- Extraction of session times from data files
- Collection of configuration data from various sources
- Structuring data into a standardized FiberData model

The extractor provides a simple interface for processing fiber photometry
data and returning structured metadata.
"""

import sys
import json
from typing import Union, Optional, List
from pathlib import Path
from datetime import datetime
import pandas as pd

from aind_metadata_extractor.fip.job_settings import JobSettings
from aind_metadata_mapper.fip.utils import (
    extract_session_start_time_from_files,
    extract_session_end_time_from_files,
)
from aind_metadata_extractor.models.fip import FiberData


class FiberPhotometryExtractor:
    """Extracts fiber photometry session metadata.

    This class handles the extraction of metadata and timing information
    from fiber photometry experiment files, structuring the data into
    a standardized FiberData model.

    The extractor processes raw data files to extract session timing,
    configuration data, and other metadata required for analysis.
    """

    def __init__(self, job_settings: Union[str, JobSettings]):
        """Initialize extractor with job settings.

        Parameters
        ----------
        job_settings : Union[str, JobSettings]
            Either a JobSettings object or a JSON string that can
            be parsed into one. The settings define all required parameters
            for the session metadata, including experimenter info, subject
            ID, data paths, etc.

        Raises
        ------
        ValidationError
            If the provided settings fail schema validation
        JSONDecodeError
            If job_settings is a string but not valid JSON
        """
        if isinstance(job_settings, str):
            job_settings = JobSettings(**json.loads(job_settings))
        self.job_settings = job_settings

    def extract(self) -> FiberData:
        """Extract metadata and raw data from fiber photometry files.

        This method parses the raw data files to create a
        FiberData model containing all necessary information
        from the fiber photometry session.

        Returns
        -------
        FiberData
            Structured data model containing parsed file data and metadata
        """
        settings = self.job_settings
        data_dir = Path(settings.data_directory)

        data_files = list(data_dir.glob("FIP_Data*.csv"))
        local_timezone = settings.local_timezone
        start_time = extract_session_start_time_from_files(
            data_dir, local_timezone
        )
        end_time = (
            extract_session_end_time_from_files(
                data_dir, start_time, local_timezone
            )
            if start_time
            else None
        )

        timestamps = []
        for file in data_files:
            df = pd.read_csv(file, header=None)
            timestamps.extend(df[0].tolist())

        stream_data = settings.data_streams[0]

        return FiberData(
            start_time=start_time,
            end_time=end_time,
            data_files=data_files,
            timestamps=timestamps,
            light_source_configs=stream_data["light_sources"],
            detector_configs=stream_data["detectors"],
            fiber_configs=stream_data["fiber_connections"],
            subject_id=settings.subject_id,
            experimenter_full_name=settings.experimenter_full_name,
            rig_id=settings.rig_id,
            iacuc_protocol=settings.iacuc_protocol,
            notes=settings.notes,
            mouse_platform_name=settings.mouse_platform_name,
            active_mouse_platform=settings.active_mouse_platform,
            session_type=settings.session_type,
            anaesthesia=settings.anaesthesia,
            animal_weight_post=settings.animal_weight_post,
            animal_weight_prior=settings.animal_weight_prior,
        )

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


if __name__ == "__main__":
    sys_args = sys.argv[1:]
    main_job_settings = JobSettings.from_args(sys_args)
    extractor = FiberPhotometryExtractor(job_settings=main_job_settings)
    fiber_data = extractor.extract()

    # Print the extracted data as JSON
    print(fiber_data.model_dump_json(indent=2))