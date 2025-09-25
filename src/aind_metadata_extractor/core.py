"""Core module for metadata extraction job settings."""

import argparse
import json
import logging
from pydantic_settings import (
    BaseSettings,
    EnvSettingsSource,
    InitSettingsSource,
    JsonConfigSettingsSource,
    PydanticBaseSettingsSource,
)
from pydantic import Field
from typing import Optional, Union, List, Type, Tuple
from pathlib import Path


class BaseExtractor:
    """Parent class for metadata extractors."""

    def _extract(self):
        """Load and extract metadata from input source."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def run_job(self):
        """Run the extraction job."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def write(self) -> None:
        """Save extraction results to a standardized JSON file."""
        if not hasattr(self, "metadata"):
            raise ValueError("No metadata found. Please run the job first.")

        job_settings = getattr(self, "job_settings", None)
        if job_settings is None or getattr(job_settings, "output_directory", None) is None:
            raise ValueError("No output directory specified in job settings.")

        job_settings.output_directory.mkdir(parents=True, exist_ok=True)

        # Generate filename from the module's parent directory name
        # e.g., aind_metadata_extractor.mesoscope.extractor -> "mesoscope"
        module_path = self.__class__.__module__
        module_parts = module_path.split(".")
        if len(module_parts) >= 2:
            folder_name = module_parts[-2]  # Get the parent folder name
        else:
            raise ValueError("Cannot determine folder name from module path.")

        output_path = job_settings.output_directory / f"{folder_name}.json"

        metadata = getattr(self, "metadata")
        with open(output_path, "w") as f:
            # Handle both pydantic models and plain dicts
            if hasattr(metadata, "model_dump"):
                json.dump(metadata.model_dump(), f, indent=4)
            else:
                json.dump(metadata, f, default=str, indent=4)

        logging.info(f"Metadata written to {output_path}")


class BaseJobSettings(BaseSettings):
    """Parent class for generating settings from a config file."""

    job_settings_name: str = Field(
        ...,
        description=("Literal name for job settings to make serialized class distinct."),
    )
    input_source: Optional[Union[Path, str, List[str], List[Path]]] = Field(
        default=None,
        description=("Location or locations of data sources to parse for metadata."),
    )
    output_directory: Optional[Union[Path, str]] = Field(
        default=None,
        description=("Location to metadata file data to. None to return object."),
    )

    user_settings_config_file: Optional[Union[Path, str]] = Field(
        default=None,
        repr=False,
        description="Optionally pull settings from a local config file.",
    )

    class Config:
        """Pydantic config to exclude field from displaying"""

        extra = "allow"
        exclude = {"user_settings_config_file"}

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: InitSettingsSource,
        env_settings: EnvSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        """
        Customize the order of settings sources, including JSON file.
        """
        config_file = init_settings.init_kwargs.get("user_settings_config_file")
        sources = [init_settings, env_settings]

        if isinstance(config_file, str):
            config_file = Path(config_file)

        if config_file and config_file.is_file():
            try:
                sources.append(JsonConfigSettingsSource(settings_cls, config_file))
            except Exception as e:
                logging.warning(f"Failed to load JSON config file {config_file}: {e}")
                raise

        return tuple(sources)

    @classmethod
    def from_args(cls, args: list):
        """
        Adds ability to construct settings from a list of arguments.
        Parameters
        ----------
        args : list
        A list of command line arguments to parse.
        """

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-j",
            "--job-settings",
            required=True,
            type=str,
            help=(
                r"""
                Custom settings defined by the user defined as a json
                 string. For example: -j
                 '{
                 "input_source":"/directory/to/read/from",
                 "output_directory":"/directory/to/write/to",
                 "job_settings_name": "Bergamo"}'
                """
            ),
        )
        job_args = parser.parse_args(args)
        job_settings_from_args = cls.model_validate_json(job_args.job_settings)
        return job_settings_from_args
