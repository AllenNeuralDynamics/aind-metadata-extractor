"""Module defining JobSettings for Fiber Photometry Contract-based Extractor"""

from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional, Union
import argparse
import json
from pydantic import BaseModel, Field


class JobSettings(BaseModel):
    """Settings for the contract-based Fiber Photometry metadata extractor.

    This simplified job settings model focuses on the essential parameters
    needed for the contract-based extractor. The extractor will extract
    most metadata directly from the data contract rather than requiring
    it to be specified in the job settings.
    """

    job_settings_name: Literal["FiberPhotometry"] = Field(
        default="FiberPhotometry",
        description="Name of the job settings type"
    )

    # Core required fields
    data_directory: Union[str, Path] = Field(
        ...,
        description="Path to data directory containing fiber photometry files"
    )

    # Optional metadata fields (extracted from data contract when available)
    experimenter_full_name: List[str] = Field(
        default_factory=list,
        description="List of experimenter names"
    )
    subject_id: Optional[str] = Field(
        None,
        description="Subject identifier (extracted from session data if not provided)"
    )
    session_start_time: Optional[datetime] = Field(
        None,
        description="Start time of the session (extracted from data if not provided)"
    )
    session_end_time: Optional[datetime] = Field(
        None,
        description="End time of the session (extracted from data if not provided)"
    )
    notes: Optional[str] = Field(
        None,
        description="Additional session notes"
    )

    # Technical settings
    local_timezone: str = Field(
        default="America/Los_Angeles",
        description="Timezone for the session"
    )
    output_directory: Optional[Union[str, Path]] = Field(
        None,
        description="Output directory for generated files (defaults to data_directory)"
    )
    output_filename: str = Field(
        default="session_fip.json",
        description="Name of output file"
    )

    @classmethod
    def from_args(cls, args: List[str]) -> "JobSettings":
        """Create JobSettings from command line arguments.

        The contract-based extractor requires minimal configuration since
        most metadata is extracted directly from the data contract.

        Parameters
        ----------
        args : List[str]
            Command line arguments

        Returns
        -------
        JobSettings
            Parsed job settings
        """
        parser = argparse.ArgumentParser(
            description="Fiber Photometry Contract-based Extractor Job Settings"
        )

        # Required arguments
        parser.add_argument(
            "--data_directory",
            required=True,
            help="Data directory path containing FIP data files"
        )

        # Optional arguments
        parser.add_argument(
            "--experimenter_full_name",
            nargs="+",
            default=[],
            help="Experimenter names (optional - extracted from data if available)"
        )
        parser.add_argument(
            "--subject_id",
            help="Subject identifier (optional - extracted from session data if available)"
        )
        parser.add_argument(
            "--notes",
            help="Additional session notes"
        )
        parser.add_argument(
            "--local_timezone",
            default="America/Los_Angeles",
            help="Local timezone for session times"
        )
        parser.add_argument(
            "--output_directory",
            help="Output directory (defaults to data_directory)"
        )
        parser.add_argument(
            "--output_filename",
            default="session_fip.json",
            help="Output filename"
        )

        parsed_args = parser.parse_args(args)

        return cls(
            data_directory=parsed_args.data_directory,
            experimenter_full_name=parsed_args.experimenter_full_name,
            subject_id=parsed_args.subject_id,
            notes=parsed_args.notes,
            local_timezone=parsed_args.local_timezone,
            output_directory=parsed_args.output_directory,
            output_filename=parsed_args.output_filename,
        )