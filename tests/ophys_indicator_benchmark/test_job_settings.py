"""Tests job settings for Ophys Benchmark"""

import unittest
from pathlib import Path

from pydantic import ValidationError

from aind_metadata_extractor.ophys_indictor_benchmark.job_settings import (
    JobSettings,
)


class TestJobSettings(unittest.TestCase):
    """Class to test job settings"""

    def setUp(self):
        """Valid baseline dataset for tests."""
        self.valid_data = {
            "data_directory": Path("."),
            "stimulus_name": "laser_stim",
            "pulse_shape": "square",
            "pulse_frequency": [20.0, 40.0],
            "number_pulse_trains": [5, 10],
            "pulse_width": [10, 20],
            "pulse_train_duration": [1.0, 2.0],
            "fixed_pulse_train_interval": True,
            "pulse_train_interval": 5.0,
            "baseline_duration": 10.0,
            "wavelength": 470,
            "power": 2.5,
            "job_settings_name": "Optogenetics",
        }

    def test_valid_jobsettings(self):
        """Model should accept valid data."""
        settings = JobSettings(**self.valid_data)
        self.assertIsInstance(settings, JobSettings)
        self.assertEqual(settings.wavelength, 470)
        self.assertEqual(settings.power, 2.5)
        self.assertEqual(settings.pulse_frequency, [20.0, 40.0])

    def test_missing_required_field(self):
        """Missing a required field should raise ValidationError."""
        data = self.valid_data.copy()
        data.pop("stimulus_name")
        with self.assertRaises(ValidationError):
            JobSettings(**data)

    def test_optional_field_none(self):
        """pulse_train_interval may be None if fixed_pulse_train_interval is False."""
        data = self.valid_data.copy()
        data["fixed_pulse_train_interval"] = False
        data["pulse_train_interval"] = None
        settings = JobSettings(**data)
        self.assertIsNone(settings.pulse_train_interval)

    def test_invalid_types(self):
        """Invalid field types should raise ValidationError."""
        bad_cases = {
            "pulse_frequency": "not_a_list",
            "number_pulse_trains": [1, "oops"],
            "pulse_width": "wrong_type",
            "pulse_train_duration": [1, "bad"],
            "baseline_duration": "not_a_float",
            "wavelength": "not_an_int",
            "power": "not_a_float",
        }
        for field, bad_value in bad_cases.items():
            data = self.valid_data.copy()
            data[field] = bad_value
            with self.assertRaises(
                ValidationError, msg=f"Field {field} should fail"
            ):
                JobSettings(**data)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
