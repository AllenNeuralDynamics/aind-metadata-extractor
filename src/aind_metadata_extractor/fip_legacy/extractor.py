"""Fiber Photometry extractor module"""

import argparse
import json
from datetime import datetime
import os
import re
import sys
from typing import Optional, List
from pathlib import Path

import pandas as pd

from aind_metadata_extractor.fip_legacy.job_settings import JobSettings
from aind_metadata_extractor.models.fip_legacy import FiberData

REGEX_DATE = (
    r"(20[0-9]{2})-([0-9]{2})-([0-9]{2})_([0-9]{2})-" r"([0-9]{2})-([0-9]{2})"
)
REGEX_MOUSE_ID = r"([0-9]{6})"


class FiberPhotometryExtractor:
    """Extractor for Fiber Photometry metadata from data files."""

    def __init__(self, job_settings: JobSettings):
        """Initialize the Fiber Photometry extractor with job settings."""
        self.job_settings = job_settings

    @classmethod
    def from_args(cls, args: List[str]) -> "FiberPhotometryExtractor":
        """Create FiberPhotometryExtractor from command line arguments.

        Parameters
        ----------
        args : List[str]
            Command line arguments

        Returns
        -------
        FiberPhotometryExtractor
            Configured extractor instance
        """
        parser = argparse.ArgumentParser(
            description="Fiber Photometry ETL Job Settings"
        )

        # Required arguments
        parser.add_argument(
            "--subject_id", required=True, help="Subject identifier"
        )
        parser.add_argument("--rig_id", required=True, help="Rig identifier")
        parser.add_argument(
            "--iacuc_protocol", required=True, help="IACUC protocol"
        )
        parser.add_argument("--notes", required=True, help="Session notes")
        parser.add_argument(
            "--data_directory", required=True, help="Data directory path"
        )

        # Optional arguments
        parser.add_argument(
            "--experimenter_full_name",
            nargs="+",
            default=[],
            help="Experimenter names",
        )
        parser.add_argument(
            "--session_type", default="FIB", help="Session type"
        )
        parser.add_argument(
            "--mouse_platform_name", help="Mouse platform name"
        )
        parser.add_argument(
            "--active_mouse_platform",
            action="store_true",
            help="Mouse platform active",
        )
        parser.add_argument("--anaesthesia", help="Anaesthesia used")
        parser.add_argument(
            "--animal_weight_post",
            type=float,
            help="Animal weight post session",
        )
        parser.add_argument(
            "--animal_weight_prior",
            type=float,
            help="Animal weight prior to session",
        )
        parser.add_argument(
            "--local_timezone",
            default="America/Los_Angeles",
            help="Local timezone",
        )
        parser.add_argument("--output_directory", help="Output directory")
        parser.add_argument(
            "--output_filename",
            default="session_fip.json",
            help="Output filename",
        )
        parser.add_argument(
            "--data_streams", help="JSON string of data streams configuration"
        )

        parsed_args = parser.parse_args(args)

        # Parse data_streams if provided as JSON string
        data_streams = []
        if parsed_args.data_streams:
            data_streams = json.loads(parsed_args.data_streams)

        job_settings = JobSettings(
            subject_id=parsed_args.subject_id,
            rig_id=parsed_args.rig_id,
            iacuc_protocol=parsed_args.iacuc_protocol,
            notes=parsed_args.notes,
            data_directory=parsed_args.data_directory,
            experimenter_full_name=parsed_args.experimenter_full_name,
            session_type=parsed_args.session_type,
            mouse_platform_name=parsed_args.mouse_platform_name,
            active_mouse_platform=parsed_args.active_mouse_platform,
            anaesthesia=parsed_args.anaesthesia,
            animal_weight_post=parsed_args.animal_weight_post,
            animal_weight_prior=parsed_args.animal_weight_prior,
            local_timezone=parsed_args.local_timezone,
            output_directory=parsed_args.output_directory,
            output_filename=parsed_args.output_filename,
            data_streams=data_streams,
        )

        return cls(job_settings)

    def extract(self) -> dict:
        """Run extraction process"""

        file_metadata = self._extract_metadata_from_data_files()

        # Create the fiber data model
        fiber_data = FiberData(**file_metadata)

        return fiber_data.model_dump()

    def _extract_metadata_from_data_files(self) -> dict:
        """
        Extracts metadata from the fiber photometry data files.

        Returns
        -------
        Dict
            Dictionary containing metadata from
            the data files for the current acquisition.
        """
        # Convert input_source to Path - handle various input types
        if isinstance(self.job_settings.data_directory, (str, Path)):
            data_dir = Path(self.job_settings.data_directory)
        else:
            raise ValueError("data_directory must be a valid path")

        if not data_dir.exists():
            raise FileNotFoundError(
                f"Data directory {data_dir} does not exist"
            )

        # Find FIP data files
        data_files = list(data_dir.glob("FIP_Data*.csv"))
        if not data_files:
            # Try alternative patterns
            data_files = list(data_dir.glob("*.csv"))

        if not data_files:
            raise FileNotFoundError(f"No data files found in {data_dir}")

        # Extract session timing
        start_time, end_time = self._extract_session_timing(data_files)

        # Extract timestamps from data files
        timestamps = self._extract_timestamps(data_files)

        # Get hardware configurations from job settings
        light_source_configs = []
        detector_configs = []
        fiber_configs = []

        if self.job_settings.data_streams:
            stream_data = (
                self.job_settings.data_streams[0]
                if self.job_settings.data_streams
                else {}
            )
            light_source_configs = stream_data.get("light_sources", [])
            detector_configs = stream_data.get("detectors", [])
            fiber_configs = stream_data.get("fiber_connections", [])

        metadata_dict = {
            "start_time": start_time,
            "end_time": end_time,
            "data_files": data_files,
            "timestamps": timestamps,
            "light_source_configs": light_source_configs,
            "detector_configs": detector_configs,
            "fiber_configs": fiber_configs,
            "subject_id": self.job_settings.subject_id,
            "experimenter_full_name": self.job_settings.experimenter_full_name,
            "rig_id": self.job_settings.rig_id,
            "iacuc_protocol": self.job_settings.iacuc_protocol,
            "notes": self.job_settings.notes,
            "mouse_platform_name": self.job_settings.mouse_platform_name,
            "active_mouse_platform": self.job_settings.active_mouse_platform,
            "session_type": self.job_settings.session_type,
            "anaesthesia": self.job_settings.anaesthesia,
            "animal_weight_post": self.job_settings.animal_weight_post,
            "animal_weight_prior": self.job_settings.animal_weight_prior,
        }

        return metadata_dict

    def _extract_session_timing(
        self, data_files: List[Path]
    ) -> tuple[Optional[datetime], Optional[datetime]]:
        """Extract session start and end times from data files."""
        if not data_files:
            return None, None

        try:
            # Read the first data file to get timing information
            first_file = data_files[0]
            df = pd.read_csv(first_file)

            # Try to find timestamp column
            timestamp_cols = [
                col
                for col in df.columns
                if "time" in col.lower() or "timestamp" in col.lower()
            ]

            if timestamp_cols:
                timestamps = pd.to_datetime(df[timestamp_cols[0]])
                start_time = timestamps.min().to_pydatetime()
                end_time = timestamps.max().to_pydatetime()
                return start_time, end_time
            else:
                # Try to extract from filename if timestamp column not found
                return self._extract_timing_from_filename(first_file)

        except Exception:
            # Fallback to filename extraction
            return self._extract_timing_from_filename(data_files[0])

    def _extract_timing_from_filename(
        self, file_path: Path
    ) -> tuple[Optional[datetime], Optional[datetime]]:
        """Extract timing information from filename using regex."""
        try:
            # Try to match date pattern in filename or parent directory
            for path_part in [file_path.name, file_path.parent.name]:
                date_match = re.search(REGEX_DATE, path_part)
                if date_match:
                    start_time = datetime.strptime(
                        date_match.group(), "%Y-%m-%d_%H-%M-%S"
                    )
                    return start_time, None
            return None, None
        except Exception:
            return None, None

    def _extract_timestamps(self, data_files: List[Path]) -> List[float]:
        """Extract timestamps from all data files."""
        all_timestamps = []

        for file_path in data_files:
            try:
                df = pd.read_csv(file_path)

                # Look for timestamp columns
                timestamp_cols = [
                    col for col in df.columns if "time" in col.lower()
                ]

                if timestamp_cols:
                    timestamps = pd.to_datetime(df[timestamp_cols[0]])
                    # Convert to relative timestamps (seconds from start)
                    if not timestamps.empty:
                        relative_timestamps = (
                            timestamps - timestamps.min()
                        ).dt.total_seconds()
                        all_timestamps.extend(relative_timestamps.tolist())
                else:
                    # If no timestamp column, use row index as proxy
                    all_timestamps.extend(list(range(len(df))))

            except Exception:
                # Skip files that can't be read
                continue

if __name__ == "__main__":
    # Example usage
    if len(sys.argv) < 2:
        print("Usage: python extractor.py <data_directory>")
        sys.exit(1)

    data_directory = sys.argv[1]

    # Create job settings with minimal required fields
    job_settings = JobSettings(
        data_directory=data_directory,
        experimenter_full_name=["Auto Extractor"],
        subject_id="UNKNOWN",
        rig_id="UNKNOWN",
        iacuc_protocol="UNKNOWN",
        notes="Extracted using data contract"
    )

    extractor = FiberPhotometryExtractor(job_settings)
    extracted_data = extractor.extract()

    fiber_data = FiberData(**extracted_data)
    print(fiber_data.model_dump_json(indent=3))