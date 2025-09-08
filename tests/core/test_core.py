"""Tests class and methods in core module"""

import json
import os
import unittest
from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock, patch

from aind_metadata_extractor.core import BaseJobSettings

RESOURCES_DIR = Path("tests/resources")
CONFIG_FILE_PATH = RESOURCES_DIR / "job_settings.json"
CONFIG_FILE_PATH_CORRUPT = RESOURCES_DIR / "job_settings_corrupt.txt"


class TestJobSettings(unittest.TestCase):
    """Tests JobSettings can be configured from json file."""

    class MockJobSettings(BaseJobSettings):
        """Mock class for testing purposes"""

        job_settings_name: Literal["mock_job"] = "mock_job"
        name: str
        id: int

    def test_load_from_config_file(self):
        """Test job settings can be loaded from config file."""

        job_settings = self.MockJobSettings(
            job_settings_name="mock_job",
            user_settings_config_file=CONFIG_FILE_PATH,
        )
        expected_settings_json = json.dumps(
            {
                "job_settings_name": "mock_job",
                "user_settings_config_file": str(CONFIG_FILE_PATH),
                "name": "Anna Apple",
                "id": 12345,
            }
        )
        round_trip = self.MockJobSettings.model_validate_json(expected_settings_json)
        self.assertEqual(round_trip.model_dump_json(), job_settings.model_dump_json())

    @patch("logging.warning")
    def test_load_from_config_file_json_error(self, mock_log_warn: MagicMock):
        """Test job settings raises an error when config file is corrupt"""

        with self.assertRaises(Exception):
            self.MockJobSettings(user_settings_config_file=CONFIG_FILE_PATH_CORRUPT)
        mock_log_warn.assert_called_once()

    def test_from_args(self):
        """Test job settings can be created from command line arguments."""
        args = ["-j", '{"job_settings_name": "mock_job", "name": "Test User", "id": 42}']

        job_settings = self.MockJobSettings.from_args(args)

        self.assertEqual(job_settings.job_settings_name, "mock_job")
        self.assertEqual(job_settings.name, "Test User")
        self.assertEqual(job_settings.id, 42)

    def test_from_args_missing_required_arg(self):
        """Test that from_args raises error when required argument is missing."""
        args = []  # Missing required -j argument

        with self.assertRaises(SystemExit):
            self.MockJobSettings.from_args(args)

    def test_from_args_invalid_json(self):
        """Test that from_args raises error when JSON is invalid."""
        args = ["-j", "invalid json"]

        with self.assertRaises(Exception):
            self.MockJobSettings.from_args(args)


if __name__ == "__main__":
    unittest.main()
