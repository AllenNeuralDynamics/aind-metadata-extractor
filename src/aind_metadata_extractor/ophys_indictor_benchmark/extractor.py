"""
Ophys Indicator Benchmark Optogenetics Extractor
"""
import pandas as pd
from typing import Union
from pathlib import Path
from aind_metadata_extractor.ophys_indictor_benchmark.job_settings import JobSettings
from aind_metadata_extractor.models.ophys_indicator_benchmark import OptoModel, OphysIndicatorBenchmarkModel

from datetime import datetime
from zoneinfo import ZoneInfo
from tzlocal import get_localzone

class OphysIndicatorBenchMarkExtractor:
    """Extractor for Ophys Benchmark Opto Metadata."""

    def __init__(self, job_settings: Union[str, JobSettings]):
        """Initialize the SmartSPIM extractor with job settings."""
        if isinstance(job_settings, str):
            self.job_settings = JobSettings.model_validate_json(job_settings)
        else:
            self.job_settings = job_settings
    
    def extract(self) -> OphysIndicatorBenchmarkModel:
        """Run extraction process"""
        opto_params = self._extract_opto_parameters()
        stimulus_epochs = self._extract_stimulus_epochs()
        opto_model = OptoModel(opto_metadata=opto_params, stimulus_epochs=stimulus_epochs)

        return OphysIndicatorBenchmarkModel(opto_data=opto_model)

    
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
            "baseline_duration": self.job_settings.baseline_duration
        }
    
    def _extract_stimulus_epochs(self) -> dict:
        """Extracts stimulus epoch information"""
        stim_csv_path = tuple(self.job_settings.data_directory.glob("Stim*.csv"))
        if not stim_csv_path:
            raise FileNotFoundError("No stim csv found. Check data")
        
        stim_df = pd.read_csv(stim_csv_path[0])

        tz = ZoneInfo("America/Los_Angeles")
        start_time = datetime.fromisoformat(
            stim_df["SoftwareTS"].iloc[0]
        ).replace(tzinfo=tz)
        end_time = datetime.fromisoformat(
            stim_df["SoftwareTS"].iloc[-1]
        ).replace(tzinfo=tz)

        return {
            "stimulus_start_time": start_time,
            "stimulus_end_time": end_time,
            "stimulus_name": "OptoStim",
            "stimulus_modalities": ["Optogenetics"],
            "configurations": {
                "wavelength": self.job_settings.wavelength,
                "power": self.job_settings.power
            }
        }


