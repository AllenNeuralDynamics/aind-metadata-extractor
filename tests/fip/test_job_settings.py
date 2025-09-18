"""Test cases for Fiber Photometry job settings."""

import unittest
from datetime import datetime
from pathlib import Path

from aind_metadata_extractor.fip.job_settings import JobSettings


class TestJobSettings(unittest.TestCase):
    """Test JobSettings model validation and functionality."""

    def test_minimal_required_fields(self):
        """Test creation with only required fields."""
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

    def test_from_args(self):
        """Test from_args classmethod."""
        args = [
            "--data_directory",
            "/data",
            "--experimenter_full_name",
            "Alice",
            "Bob",
            "--subject_id",
            "SUBJ002",
            "--notes",
            "Session notes",
            "--local_timezone",
            "Europe/London",
            "--output_directory",
            "/output",
            "--output_filename",
            "output.json",
        ]
        settings = JobSettings.from_args(args)
        self.assertEqual(settings.data_directory, "/data")
        self.assertEqual(settings.experimenter_full_name, ["Alice", "Bob"])
        self.assertEqual(settings.subject_id, "SUBJ002")
        self.assertEqual(settings.notes, "Session notes")
        self.assertEqual(settings.local_timezone, "Europe/London")
        self.assertEqual(settings.output_directory, "/output")
        self.assertEqual(settings.output_filename, "output.json")

    def test_defaults(self):
        """Test default values for optional fields."""
        settings = JobSettings(data_directory="C:/data")
        self.assertEqual(settings.output_filename, "session_fip.json")
        self.assertEqual(settings.local_timezone, "America/Los_Angeles")
        self.assertIsNone(settings.output_directory)
        self.assertEqual(settings.experimenter_full_name, [])


if __name__ == "__main__":
    unittest.main()
