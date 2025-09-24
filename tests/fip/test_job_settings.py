"""Test cases for Fiber Photometry job settings."""

import unittest
from datetime import datetime
from pathlib import Path

from aind_metadata_extractor.fip.job_settings import JobSettings


class TestJobSettings(unittest.TestCase):
    """Test JobSettings model validation and functionality."""

    def setUp(self):
        """Set up test fixtures."""
        import tempfile
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create subdirectories for data and output
        self.data_dir = self.temp_path / "data"
        self.data_dir.mkdir()
        self.output_dir = self.temp_path / "output"
        self.output_dir.mkdir()

        self.valid_job_settings = {
            "data_directory": str(self.data_dir),
            "experimenter_full_name": ["Alice", "Bob"],
            "subject_id": "SUBJ001",
            "session_start_time": datetime(2023, 1, 1, 12, 0),
            "session_end_time": datetime(2023, 1, 1, 13, 0),
            "notes": "Test session",
            "local_timezone": "America/Los_Angeles",
            "output_directory": str(self.output_dir),
            "output_filename": "custom.json",
        }

    def test_minimal_required_fields(self):
        """Test creation with only required fields."""
        # Patch Path.is_dir to always return True for testing
        from unittest.mock import patch
        with patch("pathlib.Path.is_dir", return_value=True):
            settings = JobSettings(data_directory="C:/data")
            self.assertEqual(settings.data_directory, "C:/data")
            self.assertEqual(settings.job_settings_name, "FiberPhotometry")
            self.assertEqual(settings.local_timezone, "America/Los_Angeles")
            self.assertEqual(settings.output_filename, "session_fip.json")
            self.assertIsNone(settings.output_directory)
            self.assertEqual(settings.experimenter_full_name, [])

    def test_optional_fields(self):
        """Test creation with optional fields."""
        dt_start = datetime(2023, 1, 1, 12, 0)
        dt_end = datetime(2023, 1, 1, 13, 0)
        from unittest.mock import patch
        with patch("pathlib.Path.is_dir", return_value=True):
            settings = JobSettings(
                data_directory=Path("/tmp/data"),
                experimenter_full_name=["Alice", "Bob"],
                subject_id="SUBJ001",
                session_start_time=dt_start,
                session_end_time=dt_end,
                notes="Test session",
                local_timezone="UTC",
                output_directory=Path("/tmp/output"),
                output_filename="custom.json",
            )
            self.assertEqual(settings.experimenter_full_name, ["Alice", "Bob"])
            self.assertEqual(settings.subject_id, "SUBJ001")
            self.assertEqual(settings.session_start_time, dt_start)
            self.assertEqual(settings.session_end_time, dt_end)
            self.assertEqual(settings.notes, "Test session")
            self.assertEqual(settings.local_timezone, "UTC")
            self.assertEqual(settings.output_directory, Path("/tmp/output"))
            self.assertEqual(settings.output_filename, "custom.json")

    def test_defaults(self):
        """Test default values for optional fields."""
        from unittest.mock import patch
        with patch("pathlib.Path.is_dir", return_value=True):
            settings = JobSettings(data_directory="C:/data")
            self.assertEqual(settings.output_filename, "session_fip.json")
            self.assertEqual(settings.local_timezone, "America/Los_Angeles")
            self.assertIsNone(settings.output_directory)
            self.assertEqual(settings.experimenter_full_name, [])
        
    def test_validate_path_is_dir_error(self):
        """Test validation error when data_directory is not a directory."""
        from unittest.mock import patch
        with patch("pathlib.Path.is_dir", return_value=False):
            with self.assertRaises(ValueError) as context:
                JobSettings(data_directory="C:/not_a_dir")
            self.assertIn("is not a directory", str(context.exception))


if __name__ == "__main__":
    unittest.main()
