import unittest
import tempfile
import pandas as pd
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

# Import your classes
from aind_metadata_extractor.ophys_indictor_benchmark.job_settings import (
    JobSettings
)
from aind_metadata_extractor.ophys_indictor_benchmark.extractor import (
    OphysIndicatorBenchMarkExtractor
)
from aind_metadata_extractor.models.ophys_indicator_benchmark import (
    OptoModel,
    OphysIndicatorBenchmarkModel
)

class TestOphysIndicatorBenchMarkExtractor(unittest.TestCase):

    def setUp(self):
        # Create temporary directory and stim CSV
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tmp_dir.name)

        # Sample stim CSV
        df = pd.DataFrame({
            "SoftwareTS": [
                "2025-09-17T12:00:00",
                "2025-09-17T12:01:00"
            ]
        })
        df.to_csv(self.data_dir / "Stim_example.csv", index=False)

        # Valid JobSettings
        self.job_settings = JobSettings(
            data_directory=self.data_dir,
            stimulus_name="LaserStim",
            pulse_shape="square",
            pulse_frequency=[20.0, 40.0],
            number_pulse_trains=[5, 10],
            pulse_width=[10, 20],
            pulse_train_duration=[1.0, 2.0],
            fixed_pulse_train_interval=True,
            pulse_train_interval=5.0,
            baseline_duration=10.0,
            wavelength=470,
            power=2.5,
            job_settings_name="Optogenetics"
        )

    def tearDown(self):
        self.tmp_dir.cleanup()

    # ---- Initialization tests ----
    def test_init_with_jobsettings(self):
        extractor = OphysIndicatorBenchMarkExtractor(self.job_settings)
        self.assertEqual(extractor.job_settings.stimulus_name, "LaserStim")

    def test_init_with_json_string(self):
        """Initialize extractor using a JobSettings JSON string."""
        job_json = self.job_settings.model_dump_json()
        extractor = OphysIndicatorBenchMarkExtractor(job_json)
        self.assertEqual(extractor.job_settings.stimulus_name, "LaserStim")
        self.assertEqual(extractor.job_settings.pulse_shape, "square")
        self.assertIsInstance(extractor.job_settings.pulse_frequency, list)

    # ---- Opto parameters ----
    def test_extract_opto_parameters(self):
        extractor = OphysIndicatorBenchMarkExtractor(self.job_settings)
        opto_params = extractor._extract_opto_parameters()
        self.assertIsInstance(opto_params, dict)
        self.assertEqual(opto_params["stimulus_name"], "LaserStim")
        self.assertEqual(opto_params["pulse_shape"], "square")

    # ---- Stimulus epochs ----
    def test_extract_stimulus_epochs(self):
        extractor = OphysIndicatorBenchMarkExtractor(self.job_settings)
        epochs = extractor._extract_stimulus_epochs()
        self.assertIn("stimulus_start_time", epochs)
        self.assertIn("stimulus_end_time", epochs)
        self.assertEqual(epochs["stimulus_name"], "OptoStim")
        self.assertEqual(epochs["stimulus_modalities"], ["Optogenetics"])
        self.assertEqual(epochs["configurations"]["wavelength"], 470)
        self.assertEqual(epochs["configurations"]["power"], 2.5)
        # Check that start_time is datetime
        self.assertIsInstance(epochs["stimulus_start_time"], datetime)

    # ---- Full extraction ----
    def test_extract_returns_model(self):
        extractor = OphysIndicatorBenchMarkExtractor(self.job_settings)
        result = extractor.extract()
        self.assertIsInstance(result, OphysIndicatorBenchmarkModel)
        self.assertIsInstance(result.opto_data, OptoModel)
        self.assertEqual(result.opto_data.opto_metadata["stimulus_name"], "LaserStim")

    # ---- File-not-found handling ----
    def test_extract_stimulus_epochs_file_not_found(self):
        for f in self.data_dir.glob("Stim*.csv"):
            f.unlink()  # remove CSV
        extractor = OphysIndicatorBenchMarkExtractor(self.job_settings)
        with self.assertRaises(FileNotFoundError):
            extractor._extract_stimulus_epochs()


if __name__ == "__main__": # pragma: no cover
    unittest.main()
