"""Test cases for Mesoscope job settings."""

import unittest
import tempfile
from datetime import datetime
from pathlib import Path

from aind_metadata_extractor.pophys.mesoscope.job_settings import JobSettings


class TestJobSettings(unittest.TestCase):
    """Test JobSettings model validation and functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directories for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create subdirectories for input source, behavior source, and output
        self.input_dir = self.temp_path / "input"
        self.input_dir.mkdir()
        self.behavior_dir = self.temp_path / "behavior"
        self.behavior_dir.mkdir()
        self.output_dir = self.temp_path / "output"
        self.output_dir.mkdir()

        self.valid_job_settings = {
            "input_source": str(self.input_dir),
            "session_id": "12345",
            "behavior_source": str(self.behavior_dir),
            "output_directory": str(self.output_dir),
            "session_start_time": "2024-02-12T09:14:43",
            "session_end_time": "2024-02-12T10:14:43",
            "subject_id": "123456",
            "project": "TestProject",
            "experimenter_full_name": ["John Doe", "Jane Smith"],
        }

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_job_settings_creation_with_valid_data(self):
        """Test creating JobSettings with valid data."""
        job_settings = JobSettings.model_validate(self.valid_job_settings)

        self.assertEqual(job_settings.job_settings_name, "Mesoscope")
        self.assertEqual(job_settings.session_id, "12345")
        self.assertEqual(job_settings.subject_id, "123456")
        self.assertEqual(job_settings.project, "TestProject")
        self.assertEqual(job_settings.experimenter_full_name, ["John Doe", "Jane Smith"])

    def test_job_settings_with_path_objects(self):
        """Test JobSettings with Path objects."""
        settings_data = self.valid_job_settings.copy()
        settings_data["input_source"] = self.input_dir
        settings_data["behavior_source"] = self.behavior_dir
        settings_data["output_directory"] = self.output_dir

        job_settings = JobSettings.model_validate(settings_data)
        self.assertEqual(job_settings.input_source, self.input_dir)
        self.assertEqual(job_settings.behavior_source, self.behavior_dir)
        self.assertEqual(job_settings.output_directory, self.output_dir)

    def test_default_values(self):
        """Test default values for optional fields."""
        job_settings = JobSettings.model_validate(self.valid_job_settings)

        self.assertEqual(job_settings.job_settings_name, "Mesoscope")
        self.assertEqual(job_settings.iacuc_protocol, "2115")
        self.assertEqual(job_settings.magnification, "16x")
        self.assertEqual(job_settings.fov_coordinate_ml, 1.5)
        self.assertEqual(job_settings.fov_coordinate_ap, 1.5)
        self.assertEqual(job_settings.fov_reference, "Bregma")
        self.assertEqual(job_settings.mouse_platform_name, "disc")
        self.assertFalse(job_settings.make_camsitm_dir)
        self.assertIsNone(job_settings.optional_output)

    def test_custom_optional_values(self):
        """Test custom values for optional fields."""
        settings_data = self.valid_job_settings.copy()
        settings_data.update(
            {
                "iacuc_protocol": "2116",
                "magnification": "20x",
                "fov_coordinate_ml": 2.0,
                "fov_coordinate_ap": 2.5,
                "fov_reference": "Lambda",
                "mouse_platform_name": "platform",
                "make_camsitm_dir": True,
                "optional_output": str(self.output_dir / "optional"),
            }
        )

        job_settings = JobSettings.model_validate(settings_data)

        self.assertEqual(job_settings.iacuc_protocol, "2116")
        self.assertEqual(job_settings.magnification, "20x")
        self.assertEqual(job_settings.fov_coordinate_ml, 2.0)
        self.assertEqual(job_settings.fov_coordinate_ap, 2.5)
        self.assertEqual(job_settings.fov_reference, "Lambda")
        self.assertEqual(job_settings.mouse_platform_name, "platform")
        self.assertTrue(job_settings.make_camsitm_dir)

    def test_datetime_validation(self):
        """Test datetime field validation."""
        job_settings = JobSettings.model_validate(self.valid_job_settings)

        self.assertIsInstance(job_settings.session_start_time, datetime)
        self.assertIsInstance(job_settings.session_end_time, datetime)
        self.assertEqual(job_settings.session_start_time.year, 2024)
        self.assertEqual(job_settings.session_start_time.month, 2)
        self.assertEqual(job_settings.session_start_time.day, 12)

    def test_path_validation_success(self):
        """Test that valid directory paths pass validation."""
        # This should not raise any exceptions
        JobSettings.model_validate(self.valid_job_settings)

    def test_path_validation_failure_input_source(self):
        """Test that invalid input_source path raises validation error."""
        settings_data = self.valid_job_settings.copy()
        settings_data["input_source"] = str(self.temp_path / "nonexistent")

        with self.assertRaises(ValueError) as context:
            JobSettings.model_validate(settings_data)

        self.assertIn("is not a directory", str(context.exception))

    def test_path_validation_failure_behavior_source(self):
        """Test that invalid behavior_source path raises validation error."""
        settings_data = self.valid_job_settings.copy()
        settings_data["behavior_source"] = str(self.temp_path / "nonexistent")

        with self.assertRaises(ValueError) as context:
            JobSettings.model_validate(settings_data)

        self.assertIn("is not a directory", str(context.exception))

    def test_path_validation_failure_output_directory(self):
        """Test that invalid output_directory path raises validation error."""
        settings_data = self.valid_job_settings.copy()
        settings_data["output_directory"] = str(self.temp_path / "nonexistent")

        with self.assertRaises(ValueError) as context:
            JobSettings.model_validate(settings_data)

        self.assertIn("is not a directory", str(context.exception))

    def test_path_validation_with_file_instead_of_directory(self):
        """Test that file paths instead of directories raise validation error."""
        # Create a file instead of directory
        test_file = self.temp_path / "test_file.txt"
        test_file.write_text("test")

        settings_data = self.valid_job_settings.copy()
        settings_data["input_source"] = str(test_file)

        with self.assertRaises(ValueError) as context:
            JobSettings.model_validate(settings_data)

        self.assertIn("is not a directory", str(context.exception))

    def test_missing_required_fields(self):
        """Test that missing required fields raise validation errors."""
        required_fields = [
            "input_source",
            "session_id",
            "behavior_source",
            "output_directory",
            "session_start_time",
            "session_end_time",
            "subject_id",
            "project",
            "experimenter_full_name",
        ]

        for field in required_fields:
            with self.subTest(field=field):
                settings_data = self.valid_job_settings.copy()
                del settings_data[field]

                with self.assertRaises(ValueError):
                    JobSettings.model_validate(settings_data)

    def test_job_settings_name_literal(self):
        """Test that job_settings_name is correctly set to 'Mesoscope'."""
        job_settings = JobSettings.model_validate(self.valid_job_settings)
        self.assertEqual(job_settings.job_settings_name, "Mesoscope")

        # Test that it defaults to Mesoscope even if not provided
        settings_data = self.valid_job_settings.copy()
        # job_settings_name should default to Mesoscope
        job_settings = JobSettings.model_validate(settings_data)
        self.assertEqual(job_settings.job_settings_name, "Mesoscope")

    def test_experimenter_full_name_list(self):
        """Test that experimenter_full_name accepts lists of names."""
        # Test with single experimenter
        settings_data = self.valid_job_settings.copy()
        settings_data["experimenter_full_name"] = ["Alice Johnson"]

        job_settings = JobSettings.model_validate(settings_data)
        self.assertEqual(job_settings.experimenter_full_name, ["Alice Johnson"])

        # Test with multiple experimenters
        settings_data["experimenter_full_name"] = ["Alice Johnson", "Bob Smith", "Carol Brown"]

        job_settings = JobSettings.model_validate(settings_data)
        self.assertEqual(len(job_settings.experimenter_full_name), 3)
        self.assertIn("Alice Johnson", job_settings.experimenter_full_name)
        self.assertIn("Bob Smith", job_settings.experimenter_full_name)
        self.assertIn("Carol Brown", job_settings.experimenter_full_name)

    def test_model_dump(self):
        """Test that model can be dumped to dictionary."""
        job_settings = JobSettings.model_validate(self.valid_job_settings)
        dumped = job_settings.model_dump()

        self.assertIsInstance(dumped, dict)
        self.assertEqual(dumped["session_id"], "12345")
        self.assertEqual(dumped["job_settings_name"], "Mesoscope")
        self.assertEqual(dumped["subject_id"], "123456")
        self.assertEqual(dumped["project"], "TestProject")

    def test_model_validation_json(self):
        """Test model validation from JSON."""
        import json

        json_str = json.dumps(self.valid_job_settings, default=str)
        job_settings = JobSettings.model_validate_json(json_str)

        self.assertEqual(job_settings.session_id, "12345")
        self.assertEqual(job_settings.job_settings_name, "Mesoscope")
        self.assertEqual(job_settings.subject_id, "123456")

    def test_field_titles(self):
        """Test that field titles are properly set."""
        job_settings = JobSettings.model_validate(self.valid_job_settings)

        # Get field info from the model
        fields = job_settings.model_fields

        self.assertEqual(fields["input_source"].title, "Path to the input source")
        self.assertEqual(fields["session_id"].title, "ID of the session")
        self.assertEqual(fields["behavior_source"].title, "Path to the behavior source")
        self.assertEqual(fields["output_directory"].title, "Path to the output directory")
        self.assertEqual(fields["subject_id"].title, "ID of the subject")
        self.assertEqual(fields["project"].title, "Name of the project")

    def test_inheritance_from_base_job_settings(self):
        """Test that JobSettings properly inherits from BaseJobSettings."""
        from aind_metadata_extractor.core import BaseJobSettings

        job_settings = JobSettings.model_validate(self.valid_job_settings)
        self.assertIsInstance(job_settings, BaseJobSettings)

    def test_fov_coordinates_numeric_types(self):
        """Test that FOV coordinates accept numeric types."""
        settings_data = self.valid_job_settings.copy()
        settings_data.update({"fov_coordinate_ml": 2.5, "fov_coordinate_ap": 3.0})

        job_settings = JobSettings.model_validate(settings_data)
        self.assertEqual(job_settings.fov_coordinate_ml, 2.5)
        self.assertEqual(job_settings.fov_coordinate_ap, 3.0)

        # Test with integers
        settings_data.update({"fov_coordinate_ml": 2, "fov_coordinate_ap": 3})

        job_settings = JobSettings.model_validate(settings_data)
        self.assertEqual(job_settings.fov_coordinate_ml, 2.0)
        self.assertEqual(job_settings.fov_coordinate_ap, 3.0)


if __name__ == "__main__":
    unittest.main()
