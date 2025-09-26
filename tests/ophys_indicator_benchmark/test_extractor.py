"""Tests for testing ophys benchmark extractor"""
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from aind_metadata_extractor.models.ophys_indicator_benchmark import (
    OphysIndicatorBenchmarkModel,
    OptoModel,
)
from aind_metadata_extractor.ophys_indictor_benchmark.extractor import (
    OphysIndicatorBenchMarkExtractor,
)
from aind_metadata_extractor.ophys_indictor_benchmark.job_settings import (
    JobSettings,
)


class TestOphysIndicatorBenchMarkExtractor(unittest.TestCase):
    """Class for Testing Extractor"""

    def setUp(self):
        """Set up for tests"""
        # Create temporary directory and stim CSV
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tmp_dir.name)

        # Sample stim CSV
        df = pd.DataFrame(
            {"SoftwareTime": [450, 455]}
        )
        df.to_csv(self.data_dir / "Stim_2025-08-01T17_48_50.csv", index=False)

        # Minimal valid fiber_params.json (required fields only)
        fiber_params = {
            "job_settings_name": "FiberPhotometry",
            "subject_id": "mouse123",
            "rig_id": "rigA",
            "iacuc_protocol": "IACUC-001",
            "notes": "test session",
            "data_directory": str(self.data_dir),
        }
        (self.data_dir / "fiber_params.json").write_text(json.dumps(fiber_params))

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
            job_settings_name="Optogenetics",
        )

    def test_init_with_jobsettings_path(self):
        """Initialize extractor using a JobSettings JSON file path."""
        # Dump JobSettings to a temporary JSON file
        job_path = self.data_dir / "job_settings.json"
        job_path.write_text(self.job_settings.model_dump_json())

        # Initialize extractor with the file path
        extractor = OphysIndicatorBenchMarkExtractor(str(job_path))
        self.assertEqual(extractor.job_settings.stimulus_name, "LaserStim")
        self.assertEqual(extractor.job_settings.pulse_shape, "square")
        self.assertIsInstance(extractor.job_settings.pulse_frequency, list)

    def tearDown(self):
        """Clean up"""
        self.tmp_dir.cleanup()

    # ---- Initialization tests ----
    def test_init_with_jobsettings(self):
        """Test with job settings"""
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
        """Test getting opto parameters"""
        extractor = OphysIndicatorBenchMarkExtractor(self.job_settings)
        opto_params = extractor._extract_opto_parameters()
        self.assertIsInstance(opto_params, dict)
        self.assertEqual(opto_params["stimulus_name"], "LaserStim")
        self.assertEqual(opto_params["pulse_shape"], "square")

    # ---- Stimulus epochs ----
    def test_extract_stimulus_epochs(self):
        """Test getting stimulus epochs"""
        extractor = OphysIndicatorBenchMarkExtractor(self.job_settings)
        epochs = extractor._extract_stimulus_epochs()
        self.assertIn("stimulus_start_time", epochs)
        self.assertIn("stimulus_end_time", epochs)
        self.assertEqual(epochs["stimulus_name"], "OptoStim")
        self.assertEqual(epochs["stimulus_modalities"], ["Optogenetics"])
        self.assertEqual(epochs["configurations"]["wavelength"], 470)
        self.assertEqual(epochs["configurations"]["power"], 2.5)
        # Check that start_time is datetime
        self.assertIsInstance(epochs["stimulus_start_time"], str)

    # ---- Full extraction ----
    def test_extract_returns_model(self):
        """Test full extraction"""
        extractor = OphysIndicatorBenchMarkExtractor(self.job_settings)
        result = extractor.extract()
        self.assertIsInstance(result, OphysIndicatorBenchmarkModel)
        self.assertIsInstance(result.opto_data, OptoModel)
        self.assertEqual(
            result.opto_data.opto_metadata["stimulus_name"], "LaserStim"
        )

    # ---- File-not-found handling ----
    def test_extract_stimulus_epochs_file_not_found(self):
        """Test file not found"""
        for f in self.data_dir.glob("Stim*.csv"):
            f.unlink()  # remove CSV
        extractor = OphysIndicatorBenchMarkExtractor(self.job_settings)
        with self.assertRaises(FileNotFoundError):
            extractor._extract_stimulus_epochs()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
