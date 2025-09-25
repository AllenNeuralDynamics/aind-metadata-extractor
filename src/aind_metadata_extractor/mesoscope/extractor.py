"""Mesoscope Extractor class"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Tuple, Union

import h5py as h5
import tifffile

from aind_metadata_extractor.core import BaseExtractor
from aind_metadata_extractor.mesoscope.job_settings import JobSettings
from aind_metadata_extractor.utils.camstim_sync.camstim import Camstim, CamstimSettings

from aind_metadata_extractor.models.mesoscope import MesoscopeExtractModel


class MesoscopeExtract(BaseExtractor):
    """Class to manage transforming mesoscope platform json and metadata into
    a mesoscope model model."""

    _STRUCTURE_LOOKUP_DICT = {
        385: "VISp",
        394: "VISam",
        402: "VISal",
        409: "VISl",
        417: "VISrl",
        533: "VISpm",
        312782574: "VISli",
    }

    # TODO: Deprecate this constructor. Use GenericEtl constructor instead
    def __init__(self, job_settings: Union[JobSettings, str]):
        """
        Class constructor for MesoscopeExtract.
        Parameters
        ----------
        job_settings: Union[JobSettings, str]
          Variables for a particular session
        """
        if isinstance(job_settings, str):
            job_settings_model = JobSettings.model_validate_json(job_settings)
        else:
            job_settings_model = job_settings
        if isinstance(job_settings_model.behavior_source, str):
            job_settings_model.behavior_source = Path(job_settings_model.behavior_source)
        camstim_output = job_settings_model.output_directory
        if job_settings_model.make_camsitm_dir:
            camstim_output = job_settings_model.output_directory / f"{job_settings_model.session_id}_behavior"
        self.job_settings = job_settings_model
        camstim_settings = CamstimSettings(
            input_source=self.job_settings.behavior_source,
            output_directory=camstim_output,
            session_id=self.job_settings.session_id,
            subject_id=self.job_settings.subject_id,
        )
        self.camstim = Camstim(camstim_settings)

    @staticmethod
    def _read_metadata(tiff_path: Path):
        """
        Calls tifffile.read_scanimage_metadata on the specified
        path and returns teh result. This method was factored
        out so that it could be easily mocked in unit tests.
        """

        with open(tiff_path, "rb") as tiff:
            file_handle = tifffile.FileHandle(tiff)
            file_contents = tifffile.read_scanimage_metadata(file_handle)
        return file_contents

    def _read_h5_metadata(self, h5_path: str):
        """Reads scanimage metadata from h5path

        Parameters
        ----------
        h5_path : str
            Path to h5 file

        Returns
        -------
        dict
        """
        data = h5.File(h5_path)
        try:
            file_contents = data["scanimage_metadata"][()].decode()
        except KeyError:
            logging.warning("No scanimage metadata found in h5 file. Returning image shape 512x512.")
            file_contents = '[{"SI.hRoiManager.pixelsPerLine": 512, "SI.hRoiManager.linesPerFrame": 512}]'  # noqa
        data.close()
        file_contents = json.loads(file_contents)
        return file_contents

    def _extract_behavior_metdata(self) -> dict:
        """Loads behavior metadata from the behavior json files
        Returns
        -------
        dict
            behavior video metadata
        """
        session_metadata = {}
        session_id = self.job_settings.session_id
        for ftype in sorted(list(self.job_settings.behavior_source.glob("*json"))):
            if (
                ("Behavior" in ftype.stem and session_id in ftype.stem)
                or ("Eye" in ftype.stem and session_id in ftype.stem)
                or ("Face" in ftype.stem and session_id in ftype.stem)
            ):
                with open(ftype, "r") as f:
                    session_metadata[ftype.stem] = json.load(f)
        return session_metadata

    def _extract_platform_metadata(self, session_metadata: dict) -> dict:
        """Parses the platform json file and returns the metadata

        Parameters
        ----------
        session_metadata : dict
            For session parsing

        Returns
        -------
        dict
            _description_
        """
        input_source = next(self.job_settings.input_source.glob("*platform.json"), "")
        if (isinstance(input_source, str) and input_source == "") or not input_source.exists():
            raise ValueError("No platform json file found in directory")
        with open(input_source, "r") as f:
            session_metadata["platform"] = json.load(f)

        return session_metadata

    def _extract_time_series_metadata(self) -> dict:
        """Grab time series metadata from TIFF or HDF5

        Returns
        -------
        dict
            timeseries metadata
        """
        timeseries = next(self.job_settings.input_source.glob("*timeseries*.tiff"), "")
        if timeseries:
            meta = self._read_metadata(timeseries)
        else:
            experiment_dir = list(self.job_settings.input_source.glob("ophys_experiment*"))[0]
            experiment_id = experiment_dir.name.split("_")[-1]
            timeseries = next(experiment_dir.glob(f"{experiment_id}.h5"))
            meta = self._read_h5_metadata(str(timeseries))

        return meta

    def _extract(self) -> MesoscopeExtractModel:
        """extract data from the platform json file and tiff file (in the
        future).
        If input source is a file, will extract the data from the file.
        The input source is a directory, will extract the data from the
        directory.

        Returns
        -------
        (dict, dict)
            The extracted data from the platform json file and the time series
        """
        # The pydantic models will validate that the user inputs a Path.
        # We can add validators there if we want to coerce strings to Paths.
        session_metadata = self._extract_behavior_metdata()
        session_metadata = self._extract_platform_metadata(session_metadata)
        meta = self._extract_time_series_metadata()
        epochs, session_type = self._camstim_epoch_and_session()
        user_settings = self.job_settings.model_dump()
        data = {
            "session_metadata": session_metadata,
            "camstim_epochs": epochs,
            "camstim_session_type": session_type,
            "time_series_header": meta,
            "job_settings": user_settings,
        }
        return MesoscopeExtractModel(
            tiff_header=data["time_series_header"],
            session_metadata=data["session_metadata"],
            camstim_epchs=data["camstim_epochs"],
            camstim_session_type=data["camstim_session_type"],
            job_settings=data["job_settings"],
        )

    def _camstim_epoch_and_session(self) -> Tuple[list, str]:
        """Get the camstim table and epochs

        Returnsd
        -------
        list
            The camstim table and epochs
        """
        if self.camstim.behavior:
            self.camstim.build_behavior_table()
        else:
            self.camstim.build_stimulus_table(modality="ophys")
        return self.camstim.epochs_from_stim_table(), self.camstim.session_type

    def run_job(self) -> dict:
        """
        Run the extraction job.
        """
        self.metadata = self._extract()
        return self.metadata.model_dump()

    @classmethod
    def from_args(cls, args: list):
        """
        Adds ability to construct settings from a list of arguments.
        Parameters
        ----------
        args : list
        A list of command line arguments to parse.
        """

        logging.warning("This method will be removed in future versions. " "Please use JobSettings.from_args instead.")

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-u",
            "--job-settings",
            required=True,
            type=json.loads,
            help=(
                """
                Custom settings defined by the user defined as a json
                 string. For example: -u
                 '{"experimenter_full_name":["John Smith","Jane Smith"],
                 "subject_id":"12345",
                 "session_start_time":"2023-10-10T10:10:10",
                 "session_end_time":"2023-10-10T18:10:10",
                 "project":"my_project"}
                """
            ),
        )
        job_args = parser.parse_args(args)
        job_settings_from_args = JobSettings(**job_args.job_settings)
        return cls(
            job_settings=job_settings_from_args,
        )


if __name__ == "__main__":
    sys_args = sys.argv[1:]
    main_job_settings = JobSettings.from_args(sys_args)
    metl = MesoscopeExtract(job_settings=main_job_settings)
    extracted_data = metl.run_job()
