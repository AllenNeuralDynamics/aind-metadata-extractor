"""Fiber Photometry extractor module using data contract"""
import dataclasses
import json
import os
import sys
import typing as t
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Union


from aind_physiology_fip.data_contract import dataset

from aind_metadata_extractor.fip.job_settings import JobSettings
from aind_metadata_extractor.models.fip import FiberData


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
        self.dataset = None

    def extract(self) -> dict:
        """Run extraction process using the GitHub data contract.

        Uses the official data contract from:
        https://github.com/AllenNeuralDynamics/FIP_DAQ_Control/blob/bc-major-refactor/src/aind_physiology_fip/data_contract.py

        Returns
        -------
        dict
            Extracted metadata as a dictionary
        """

        # Create dataset from data directory using the GitHub data contract
        if not self.job_settings.data_directory:
            raise ValueError(
                "data_directory must be specified in job settings"
            )

        self.dataset = dataset(self.job_settings.data_directory)

        # Extract metadata using the data contract
        file_metadata = self._extract_metadata_from_contract()

        print("Extracted metadata from data contract:")
        print(json.dumps(file_metadata, indent=3, default=str))

        # Create the fiber data model
        fiber_data = FiberData(**file_metadata)

        return fiber_data.model_dump()

    def _extract_metadata_from_contract(self) -> dict:
        """
        Extract metadata using the data contract approach.

        Returns
        -------
        dict
            Extracted metadata as a dictionary
        """
        metadata = {}

        print(self.dataset)
        timing_data = self._extract_timing_from_csv()
        metadata.update(timing_data)

        # Extract data files information
        files_data = self._extract_data_files()
        metadata.update(files_data)

        hardware_data = self._extract_hardware_config()
        metadata.update(hardware_data)

        return metadata

    def _extract_index(self) -> dict:
        """
        Extract index key information from the dataset contract configuration.

        Returns
        -------
        dict
            Extracted index key information as a dictionary
        """
        index_data = {}

        try:
            # Try to get index key from green channel CSV configuration
            green_stream = self._get_data_stream("green")
            if green_stream and hasattr(green_stream, "reader_params"):
                index_key = getattr(green_stream.reader_params, "index", None)
                if index_key:
                    index_data["index_key"] = index_key
                    return index_data

        except Exception:
            pass

        try:
            # Fall back to red channel CSV configuration
            red_stream = self._get_data_stream("red")
            if red_stream and hasattr(red_stream, "reader_params"):
                index_key = getattr(red_stream.reader_params, "index", None)
                if index_key:
                    index_data["index_key"] = index_key
                    return index_data

        except Exception:
            pass

        # Default index key if none found
        index_data["index_key"] = "ReferenceTime"

        return index_data

    def _extract_timing_from_csv(self) -> dict:
        """
        Extract session timing from CSV data streams using
            the contract's index key.

        Returns
        -------
        dict
            Extracted timing information with 'start_time' and 'end_time' keys
        """
        timing_data = {}
        print(self._get_data_stream("green"))
        try:
            # Try to get timing from green channel CSV
            green_stream = self._get_data_stream("green")
            if green_stream:
                green_data = green_stream.read()
                if not green_data.empty:
                    # Get the index key from the contract configuration
                    index_key = self._extract_index().get(
                        "index_key", "ReferenceTime"
                    )

                    # Use the index key to access the timing column
                    if index_key in green_data.columns:
                        timing_data["start_time"] = green_data[index_key].min()
                        timing_data["end_time"] = green_data[index_key].max()
                        return timing_data
                    # Fallback to DataFrame index if column not found
                    elif not green_data.index.empty:
                        timing_data["start_time"] = green_data.index.min()
                        timing_data["end_time"] = green_data.index.max()
                        return timing_data

        except Exception:
            pass

        try:
            # Fall back to red channel CSV (also uses index key from contract)
            red_stream = self._get_data_stream("red")
            if red_stream:
                red_data = red_stream.read()
                if not red_data.empty:
                    # Get the index key from the contract configuration
                    index_key = self._extract_index().get(
                        "index_key", "ReferenceTime"
                    )

                    # Use the index key to access the timing column
                    if index_key in red_data.columns:
                        timing_data["start_time"] = red_data[index_key].min()
                        timing_data["end_time"] = red_data[index_key].max()
                        return timing_data
                    # Fallback to DataFrame index if column not found
                    elif not red_data.index.empty:
                        timing_data["start_time"] = red_data.index.min()
                        timing_data["end_time"] = red_data.index.max()
                        return timing_data

        except Exception:
            pass

        # Default timing if no CSV available
        timing_data["start_time"] = datetime.now()
        timing_data["end_time"] = datetime.now()

        return timing_data

    def _get_data_stream(self, stream_name: str):
        """
        Get a data stream by name from the dataset.

        Parameters
        ----------
        stream_name : str
            The name of the data stream to retrieve.

        Returns
        -------
        DataStream
            The requested data stream, or None if not found.
        """
        for stream in self.dataset._data:
            if hasattr(stream, "name") and stream.name == stream_name:
                return stream
        return None

    def _extract_data_files(self) -> dict:
        """
        Extract data files information from the dataset.

        Returns
        -------
        dict
            Extracted data files information with 'data_files' key
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
            try:
                stream = self._get_data_stream(stream_name)
                if stream:
                    file_path = getattr(stream.reader_params, "path", None)
                    if file_path and Path(file_path).exists():
                        data_files.append(str(file_path))
            except Exception:
                continue

        return {"data_files": data_files}

    def _extract_hardware_config(self) -> dict:
        """
        Extract hardware configuration from rig and session inputs.

        Returns
        -------
        dict
            Extracted hardware configuration with
                'rig_config' and 'session_config' keys
        """
        hardware_data = {}

        try:
            # Try to extract rig configuration
            rig_stream = self._get_data_stream("rig_input")
            if rig_stream:
                rig_data = rig_stream.read()
                hardware_data["rig_config"] = (
                    rig_data.model_dump()
                    if hasattr(rig_data, "model_dump")
                    else {}
                )
        except Exception:
            pass

        try:
            # Try to extract session configuration
            session_stream = self._get_data_stream("session_input")
            if session_stream:
                session_data = session_stream.read()
                hardware_data["session_config"] = (
                    session_data.model_dump()
                    if hasattr(session_data, "model_dump")
                    else {}
                )
        except Exception:
            pass

        return hardware_data

    def _extract_basic_metadata(self) -> dict:
        """
        Extract basic metadata when contract approach needs fallback data.

        Returns
        -------
        dict
            Extracted basic metadata with 'start_time',
                'end_time', and 'data_files' keys
        """
        data_dir = Path(self.job_settings.data_directory)

        # Find any available data files
        data_files = []
        for pattern in ["*.bin", "*.csv", "*.json"]:
            data_files.extend([str(f) for f in data_dir.glob(pattern)])

        # Use current time as fallback
        current_time = datetime.now()

        return {
            "start_time": current_time,
            "end_time": current_time,
            "data_files": data_files,
            "rig_config": {},
            "session_config": {},
        }

    def save_to_file(
        self, fiber_data: FiberData, output_path: Optional[Path] = None
    ) -> Path:
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
