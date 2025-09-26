"""
Ophys Indicator Benchmark Optogenetics Extractor
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Union

import pandas as pd

from aind_metadata_extractor.models.ophys_indicator_benchmark import (
    OphysIndicatorBenchmarkModel,
    OptoModel,
)
from aind_metadata_extractor.fip_legacy.job_settings import (
    JobSettings as FiberJobSettings 
) 
from aind_metadata_extractor.models.fip import FiberData

from aind_metadata_extractor.ophys_indictor_benchmark.job_settings import (
    JobSettings,
)

from aind_metadata_extractor.fip_legacy.extractor import FiberPhotometryExtractor

class OphysIndicatorBenchMarkExtractor:
    """Extractor for Ophys Benchmark Opto Metadata."""

    def __init__(self, job_settings: Union[str, JobSettings]):
        """Initialize the Ophys Benchmark extractor with job settings."""
        if isinstance(job_settings, str):
            if Path(job_settings).exists():
                with open(job_settings, "r") as f:
                    jobs_settings_params = json.load(f)
                    job_settings = json.dumps(jobs_settings_params)

            self.job_settings = JobSettings.model_validate_json(job_settings)
        else:
            self.job_settings = job_settings
        
        with open(self.job_settings.data_directory / "fiber_params.json", "r") as f:
            fiber_params = json.load(f)
            fiber_job_settings = FiberJobSettings(**fiber_params)
            self.fiber_data = FiberPhotometryExtractor(fiber_job_settings)


    def extract(self) -> OphysIndicatorBenchmarkModel:
        """Run extraction process"""
        opto_params = self._extract_opto_parameters()
        stimulus_epochs = self._extract_stimulus_epochs()
        opto_model = OptoModel(
            opto_metadata=opto_params, stimulus_epochs=stimulus_epochs
        )

        fiber_metadata = FiberData(**self.fiber_data.extract())
        return OphysIndicatorBenchmarkModel(opto_data=opto_model, fiber_data=fiber_metadata)

    def _extract_opto_parameters(self) -> dict:
        """Returns opto parameters"""
        return {
            "stimulus_name": self.job_settings.stimulus_name,
            "pulse_shape": self.job_settings.pulse_shape,
            "pulse_frequency": self.job_settings.pulse_frequency,
            "number_pulse_trains": self.job_settings.number_pulse_trains,
            "pulse_width": self.job_settings.pulse_width,
            "pulse_train_duration": self.job_settings.pulse_train_duration,
            "fixed_pulse_train_interval": self.job_settings.fixed_pulse_train_interval,
            "pulse_train_interval": self.job_settings.pulse_train_interval,
            "baseline_duration": self.job_settings.baseline_duration,
        }

    def _extract_stimulus_epochs(self) -> dict:
        """Extracts stimulus epoch information"""
        stim_csv_path = tuple(
            self.job_settings.data_directory.glob("Stim*.csv")
        )
        if not stim_csv_path:
            raise FileNotFoundError("No stim csv found. Check data")

        stim_df = pd.read_csv(stim_csv_path[0])
        filename = stim_csv_path[0].stem

        # Parse directly with the format
        start_time = datetime.strptime(filename, "Stim_%Y-%m-%dT%H_%M_%S")
        start_time = start_time + pd.to_timedelta(self.job_settings.baseline_duration, unit="s")
        # Compute end time
        end_time = start_time + pd.to_timedelta(
            stim_df["SoftwareTS"].max(), unit="us"
        )
        end_time = end_time.isoformat()

        return {
            "stimulus_start_time": start_time.isoformat(),
            "stimulus_end_time": end_time,
            "stimulus_name": "OptoStim",
            "stimulus_modalities": ["Optogenetics"],
            "configurations": {
                "wavelength": self.job_settings.wavelength,
                "power": self.job_settings.power,
            },
        }
