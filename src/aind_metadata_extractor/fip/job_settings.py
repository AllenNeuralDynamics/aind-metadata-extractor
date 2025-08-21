"""Module defining JobSettings for Fiber Photometry ETL"""

from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional, Union
import argparse
import json
from pydantic import BaseModel, Field


class JobSettings(BaseModel):
    """Settings for generating Fiber Photometry session metadata."""

    job_settings_name: Literal["FiberPhotometry"] = Field(default="FiberPhotometry", description="Name of the job settings type")
    experimenter_full_name: List[str] = Field(..., description="List of experimenter names")
    session_start_time: Optional[datetime] = Field(None, description="Start time of the session")
    session_end_time: Optional[datetime] = Field(None, description="End time of the session")
    subject_id: str = Field(..., description="Subject identifier")
    rig_id: str = Field(..., description="Identifier for the experimental rig")
    mouse_platform_name: str = Field(..., description="Name of the mouse platform used")
    active_mouse_platform: bool = Field(..., description="Whether the mouse platform was active during the session")
    data_streams: List[dict] = Field(..., description="List of data stream configurations")
    session_type: str = Field(default="FIB", description="Type of session")
    iacuc_protocol: str = Field(..., description="IACUC protocol identifier")
    notes: str = Field(..., description="Session notes")
    anaesthesia: Optional[str] = Field(None, description="Anaesthesia used")
    animal_weight_post: Optional[float] = Field(None, description="Animal weight after session")
    animal_weight_prior: Optional[float] = Field(None, description="Animal weight before session")
    protocol_id: List[str] = Field(default_factory=list, description="List of protocol identifiers")
    data_directory: Optional[Union[str, Path]] = Field(None, description="Path to data directory containing fiber photometry files")
    local_timezone: str = Field(default="America/Los_Angeles", description="Timezone for the session")
    output_directory: Optional[Union[str, Path]] = Field(None, description="Output directory for generated files")
    output_filename: str = Field(default="session_fip.json", description="Name of output file")

    @classmethod
    def from_args(cls, args: List[str]) -> "JobSettings":
        """Create JobSettings from command line arguments.

        Parameters
        ----------
        args : List[str]
            Command line arguments

        Returns
        -------
        JobSettings
            Parsed job settings
        """
        parser = argparse.ArgumentParser(description="Fiber Photometry ETL Job Settings")

        # Required arguments
        parser.add_argument("--subject_id", required=True, help="Subject identifier")
        parser.add_argument("--rig_id", required=True, help="Rig identifier")
        parser.add_argument("--mouse_platform_name", required=True, help="Mouse platform name")
        parser.add_argument("--iacuc_protocol", required=True, help="IACUC protocol")
        parser.add_argument("--notes", required=True, help="Session notes")
        parser.add_argument("--data_directory", required=True, help="Data directory path")

        # Optional arguments
        parser.add_argument("--experimenter_full_name", nargs="+", default=[], help="Experimenter names")
        parser.add_argument("--active_mouse_platform", action="store_true", help="Mouse platform active")
        parser.add_argument("--session_type", default="FIB", help="Session type")
        parser.add_argument("--anaesthesia", help="Anaesthesia used")
        parser.add_argument("--animal_weight_post", type=float, help="Animal weight post session")
        parser.add_argument("--animal_weight_prior", type=float, help="Animal weight prior to session")
        parser.add_argument("--local_timezone", default="America/Los_Angeles", help="Local timezone")
        parser.add_argument("--output_directory", help="Output directory")
        parser.add_argument("--output_filename", default="session_fip.json", help="Output filename")
        parser.add_argument("--data_streams", help="JSON string of data streams configuration")

        parsed_args = parser.parse_args(args)

        # Parse data_streams if provided as JSON string
        data_streams = []
        if parsed_args.data_streams:
            data_streams = json.loads(parsed_args.data_streams)
        else:
            # Default data stream configuration for fiber photometry
            data_streams = [{
                "light_sources": [],
                "detectors": [],
                "fiber_connections": []
            }]

        return cls(
            subject_id=parsed_args.subject_id,
            rig_id=parsed_args.rig_id,
            mouse_platform_name=parsed_args.mouse_platform_name,
            active_mouse_platform=parsed_args.active_mouse_platform,
            iacuc_protocol=parsed_args.iacuc_protocol,
            notes=parsed_args.notes,
            data_directory=parsed_args.data_directory,
            experimenter_full_name=parsed_args.experimenter_full_name,
            session_type=parsed_args.session_type,
            anaesthesia=parsed_args.anaesthesia,
            animal_weight_post=parsed_args.animal_weight_post,
            animal_weight_prior=parsed_args.animal_weight_prior,
            local_timezone=parsed_args.local_timezone,
            output_directory=parsed_args.output_directory,
            output_filename=parsed_args.output_filename,
            data_streams=data_streams,
        )