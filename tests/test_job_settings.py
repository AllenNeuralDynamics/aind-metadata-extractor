"""Test cases for SmartSPIM job settings."""

import unittest
from pathlib import Path

from aind_metadata_extractor.smartspim.job_settings import JobSettings


class TestJobSettings(unittest.TestCase):
    """Test JobSettings model validation and functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.valid_job_settings = {
            "subject_id": "804714",
            "metadata_service_path": "https://api.metadata.service.com/smartspim",
            "input_source": "/data/SmartSPIM_2025-08-19_15-03-00",
        }

    def test_job_settings_creation_with_valid_data(self):
        """Test creating JobSettings with valid data."""
        job_settings = JobSettings.model_validate(self.valid_job_settings)

        self.assertEqual(job_settings.job_settings_name, "SmartSPIM")
        self.assertEqual(job_settings.subject_id, "804714")
        self.assertEqual(
            job_settings.metadata_service_path,
            "https://api.metadata.service.com/smartspim"
        )
        self.assertEqual(
            job_settings.input_source,
            "/data/SmartSPIM_2025-08-19_15-03-00"
        )

    def test_job_settings_with_path_object(self):
        """Test JobSettings with Path object for input_source."""
        settings_data = self.valid_job_settings.copy()
        test_path = "/data/SmartSPIM_2025-08-19_15-03-00"
        settings_data["input_source"] = str(Path(test_path))

        job_settings = JobSettings.model_validate(settings_data)
        self.assertEqual(
            str(job_settings.input_source),
            str(Path(test_path))
        )

    def test_default_asi_filename(self):
        """Test default ASI filename."""
        job_settings = JobSettings.model_validate(self.valid_job_settings)
        self.assertEqual(
            job_settings.asi_filename,
            "derivatives/ASI_logging.txt"
        )

    def test_default_metadata_filename(self):
        """Test default metadata filename."""
        job_settings = JobSettings.model_validate(self.valid_job_settings)
        self.assertEqual(
            job_settings.mdata_filename_json,
            "derivatives/metadata.json"
        )

    def test_custom_asi_filename(self):
        """Test custom ASI filename."""
        settings_data = self.valid_job_settings.copy()
        settings_data["asi_filename"] = "custom/asi_log.txt"

        job_settings = JobSettings.model_validate(settings_data)
        self.assertEqual(job_settings.asi_filename, "custom/asi_log.txt")

    def test_custom_metadata_filename(self):
        """Test custom metadata filename."""
        settings_data = self.valid_job_settings.copy()
        settings_data["mdata_filename_json"] = "custom/metadata.json"

        job_settings = JobSettings.model_validate(settings_data)
        self.assertEqual(
            job_settings.mdata_filename_json,
            "custom/metadata.json"
        )

    def test_optional_slims_datetime(self):
        """Test optional SLIMS datetime field."""
        job_settings = JobSettings.model_validate(self.valid_job_settings)
        self.assertIsNone(job_settings.slims_datetime)

        # Test with datetime provided
        settings_data = self.valid_job_settings.copy()
        settings_data["slims_datetime"] = "2025-08-19T19:03:00Z"

        job_settings = JobSettings.model_validate(settings_data)
        self.assertEqual(job_settings.slims_datetime, "2025-08-19T19:03:00Z")

    def test_deprecated_raw_dataset_path(self):
        """Test deprecated raw_dataset_path field."""
        settings_data = self.valid_job_settings.copy()
        settings_data["raw_dataset_path"] = "/old/path/format"

        job_settings = JobSettings.model_validate(settings_data)
        self.assertEqual(job_settings.raw_dataset_path, "/old/path/format")

    def test_deprecated_processing_manifest_path(self):
        """Test deprecated processing_manifest_path field."""
        settings_data = self.valid_job_settings.copy()
        settings_data["processing_manifest_path"] = (
            "derivatives/old_manifest.json"
        )

        job_settings = JobSettings.model_validate(settings_data)
        self.assertEqual(
            job_settings.processing_manifest_path,
            "derivatives/old_manifest.json"
        )

    def test_missing_required_subject_id(self):
        """Test that missing subject_id raises validation error."""
        settings_data = self.valid_job_settings.copy()
        del settings_data["subject_id"]

        with self.assertRaises(ValueError):
            JobSettings.model_validate(settings_data)

    def test_missing_required_metadata_service_path(self):
        """Test that missing metadata_service_path raises validation error."""
        settings_data = self.valid_job_settings.copy()
        del settings_data["metadata_service_path"]

        with self.assertRaises(ValueError):
            JobSettings.model_validate(settings_data)

    def test_job_settings_name_literal(self):
        """Test that job_settings_name must be 'SmartSPIM'."""
        job_settings = JobSettings.model_validate(self.valid_job_settings)
        self.assertEqual(job_settings.job_settings_name, "SmartSPIM")

        # Test that it defaults to SmartSPIM even if not provided
        settings_data = self.valid_job_settings.copy()
        # job_settings_name should default to SmartSPIM

        job_settings = JobSettings.model_validate(settings_data)
        self.assertEqual(job_settings.job_settings_name, "SmartSPIM")

    def test_model_dump(self):
        """Test that model can be dumped to dictionary."""
        job_settings = JobSettings.model_validate(self.valid_job_settings)
        dumped = job_settings.model_dump()

        self.assertIsInstance(dumped, dict)
        self.assertEqual(dumped["subject_id"], "804714")
        self.assertEqual(dumped["job_settings_name"], "SmartSPIM")

    def test_model_validation_json(self):
        """Test model validation from JSON."""
        import json

        json_str = json.dumps(self.valid_job_settings)
        job_settings = JobSettings.model_validate_json(json_str)

        self.assertEqual(job_settings.subject_id, "804714")
        self.assertEqual(job_settings.job_settings_name, "SmartSPIM")


if __name__ == "__main__":
    unittest.main()
