"""Tests class and methods in core module"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock, patch, mock_open

from aind_metadata_extractor.core import BaseJobSettings, BaseExtractor

RESOURCES_DIR = Path(os.path.dirname(os.path.realpath(__file__))) / ".." / "resources"
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


class TestBaseExtractor(unittest.TestCase):
    """Tests for BaseExtractor class"""

    class MockExtractor(BaseExtractor):
        """Mock extractor for testing purposes"""

        def __init__(self, job_settings=None, metadata=None):
            self.job_settings = job_settings
            self.metadata = metadata

        def _extract(self):
            """Mock implementation"""
            pass

        def run_job(self):
            """Mock implementation"""
            pass

    class MockJobSettings:
        """Mock job settings for testing"""

        def __init__(self, output_directory=None):
            self.output_directory = Path(output_directory) if output_directory else None

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_write_success_with_pydantic_model(self):
        """Test successful write with pydantic model metadata"""
        # Create mock metadata with model_dump method (pydantic model)
        mock_metadata = MagicMock()
        mock_metadata.model_dump.return_value = {"key": "value", "number": 42}

        # Create extractor with proper module path
        extractor = self.MockExtractor()
        extractor.__class__.__module__ = "aind_metadata_extractor.mesoscope.extractor"
        extractor.job_settings = self.MockJobSettings(self.temp_dir)
        extractor.metadata = mock_metadata

        # Mock open and json.dump
        with (
            patch("builtins.open", mock_open()) as mock_file,
            patch("json.dump") as mock_json_dump,
            patch("logging.info") as mock_log,
        ):

            extractor.write()

            # Verify file operations
            expected_path = self.temp_path / "mesoscope.json"
            mock_file.assert_called_once_with(expected_path, "w")
            mock_json_dump.assert_called_once_with(
                {"key": "value", "number": 42}, mock_file.return_value.__enter__.return_value, indent=4
            )
            mock_log.assert_called_once_with(f"Metadata written to {expected_path}")

    def test_write_success_with_dict_metadata(self):
        """Test successful write with dictionary metadata"""
        # Create dictionary metadata (no model_dump method)
        dict_metadata = {"key": "value", "list": [1, 2, 3]}

        # Create extractor with proper module path
        extractor = self.MockExtractor()
        extractor.__class__.__module__ = "aind_metadata_extractor.bergamo.extractor"
        extractor.job_settings = self.MockJobSettings(self.temp_dir)
        extractor.metadata = dict_metadata

        # Mock open and json.dump
        with (
            patch("builtins.open", mock_open()) as mock_file,
            patch("json.dump") as mock_json_dump,
            patch("logging.info") as mock_log,
        ):

            extractor.write()

            # Verify file operations
            expected_path = self.temp_path / "bergamo.json"
            mock_file.assert_called_once_with(expected_path, "w")
            mock_json_dump.assert_called_once_with(
                dict_metadata, mock_file.return_value.__enter__.return_value, default=str, indent=4
            )
            mock_log.assert_called_once_with(f"Metadata written to {expected_path}")

    def test_write_creates_output_directory(self):
        """Test that write creates output directory if it doesn't exist"""
        nested_dir = self.temp_path / "nested" / "directory"

        extractor = self.MockExtractor()
        extractor.__class__.__module__ = "aind_metadata_extractor.smartspim.extractor"
        extractor.job_settings = self.MockJobSettings(nested_dir)
        extractor.metadata = {"test": "data"}

        with patch("builtins.open", mock_open()), patch("json.dump"), patch("logging.info"):

            extractor.write()

            # Verify directory was created
            self.assertTrue(nested_dir.exists())
            self.assertTrue(nested_dir.is_dir())

    def test_write_no_metadata_raises_error(self):
        """Test that write raises ValueError if no metadata exists"""
        extractor = self.MockExtractor()
        extractor.__class__.__module__ = "aind_metadata_extractor.test.extractor"  # Valid module path
        extractor.job_settings = self.MockJobSettings(self.temp_dir)
        # Explicitly remove metadata attribute
        if hasattr(extractor, "metadata"):
            delattr(extractor, "metadata")

        with self.assertRaises(ValueError) as context:
            extractor.write()

        self.assertEqual(str(context.exception), "No metadata found. Please run the job first.")

    def test_write_no_job_settings_raises_error(self):
        """Test that write raises ValueError if no job_settings exists"""
        extractor = self.MockExtractor()
        extractor.metadata = {"test": "data"}
        # Don't set job_settings

        with self.assertRaises(ValueError) as context:
            extractor.write()

        self.assertEqual(str(context.exception), "No output directory specified in job settings.")

    def test_write_no_output_directory_raises_error(self):
        """Test that write raises ValueError if no output_directory in job_settings"""
        extractor = self.MockExtractor()
        extractor.job_settings = self.MockJobSettings(None)  # output_directory is None
        extractor.metadata = {"test": "data"}

        with self.assertRaises(ValueError) as context:
            extractor.write()

        self.assertEqual(str(context.exception), "No output directory specified in job settings.")

    def test_write_invalid_module_path_raises_error(self):
        """Test that write raises ValueError for invalid module path"""
        extractor = self.MockExtractor()
        extractor.__class__.__module__ = "invalid_module"  # Not enough parts
        extractor.job_settings = self.MockJobSettings(self.temp_dir)
        extractor.metadata = {"test": "data"}

        with self.assertRaises(ValueError) as context:
            extractor.write()

        self.assertEqual(str(context.exception), "Cannot determine folder name from module path.")

    def test_write_filename_generation(self):
        """Test that filename is correctly generated from module path"""
        test_cases = [
            ("aind_metadata_extractor.mesoscope.extractor", "mesoscope.json"),
            ("aind_metadata_extractor.bergamo.extractor", "bergamo.json"),
            ("aind_metadata_extractor.smartspim.extractor", "smartspim.json"),
            ("some.package.fip.module", "fip.json"),
        ]

        for module_path, expected_filename in test_cases:
            with self.subTest(module_path=module_path):
                extractor = self.MockExtractor()
                extractor.__class__.__module__ = module_path
                extractor.job_settings = self.MockJobSettings(self.temp_dir)
                extractor.metadata = {"test": "data"}

                with patch("builtins.open", mock_open()) as mock_file, patch("json.dump"), patch("logging.info"):

                    extractor.write()

                    expected_path = self.temp_path / expected_filename
                    mock_file.assert_called_with(expected_path, "w")

    def test_abstract_methods_not_implemented(self):
        """Test that abstract methods raise NotImplementedError"""
        extractor = BaseExtractor()

        with self.assertRaises(NotImplementedError) as context:
            extractor._extract()
        self.assertEqual(str(context.exception), "This method should be implemented by subclasses.")

        with self.assertRaises(NotImplementedError) as context:
            extractor.run_job()
        self.assertEqual(str(context.exception), "This method should be implemented by subclasses.")


if __name__ == "__main__":
    unittest.main()
