"""Tests for fiber photometry legacy extractor."""

import unittest
from unittest.mock import patch
from pathlib import Path
from datetime import datetime
import tempfile
import re

from aind_metadata_extractor.fip_legacy.extractor import FiberPhotometryExtractor
from aind_metadata_extractor.fip_legacy.job_settings import JobSettings
from aind_metadata_extractor.models.fip import FiberData


class TestFiberPhotometryLegacyExtractor(unittest.TestCase):
    """Test class for legacy FiberPhotometryExtractor with file-based data access."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_dir = Path(self.temp_dir)

        # Create test job settings with all required fields
        self.job_settings = JobSettings(
            subject_id="mouse_001",
            rig_id="Rig_001",
            iacuc_protocol="IACUC-12345",
            notes="Test experiment notes",
            experimenter_full_name=["John Doe", "Jane Smith"],
            session_type="FIB",
            mouse_platform_name="Platform_A",
            active_mouse_platform=True,
            anaesthesia="isoflurane",
        )

        # Create sample test files
        self.create_test_files()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def create_test_files(self):
        """Create sample test files for testing."""
        # Create a sample CSV file with timestamp data
        csv_content = """timestamp,value
        2024-01-15 14:30:00,100.5
        2024-01-15 14:30:01,101.2
        2024-01-15 14:30:02,99.8
        """

        csv_file = self.test_data_dir / "2024-01-15_14-30-00_mouse_001_fibpho.csv"
        csv_file.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_file, "w") as f:
            f.write(csv_content)

        # Create additional test files with different patterns
        other_file = self.test_data_dir / "2024-01-15_14-30-00_mouse_001_other.txt"
        with open(other_file, "w") as f:
            f.write("Other data file")

    def test_extract_metadata_success(self):
        """Test successful metadata extraction from files."""
        extractor = FiberPhotometryExtractor(self.job_settings)

        with patch.object(extractor, "_extract_session_start_time", return_value=datetime(2024, 1, 15, 14, 30, 0)):
            result = extractor.extract_metadata()

        # Verify result
        self.assertIsInstance(result, FiberData)
        self.assertEqual(result.subject_id, "mouse_001")
        self.assertEqual(result.rig_id, "Rig_001")
        self.assertEqual(result.iacuc_protocol, "IACUC-12345")
        self.assertEqual(result.notes, "Test experiment notes")
        self.assertEqual(result.experimenter_full_name, ["John Doe", "Jane Smith"])
        self.assertEqual(result.session_type, "FIB")
        self.assertEqual(result.mouse_platform_name, "Platform_A")
        self.assertEqual(result.active_mouse_platform, True)
        self.assertEqual(result.anaesthesia, "isoflurane")
        self.assertEqual(result.session_start_time, datetime(2024, 1, 15, 14, 30, 0))

    def test_extract_metadata_minimal_settings(self):
        """Test metadata extraction with minimal job settings."""
        minimal_settings = JobSettings(
            subject_id="mouse_002", rig_id="Rig_002", iacuc_protocol="IACUC-67890", notes="Minimal test"
        )

        extractor = FiberPhotometryExtractor(minimal_settings)

        with patch.object(extractor, "_extract_session_start_time", return_value=datetime(2024, 1, 15, 15, 0, 0)):
            result = extractor.extract_metadata()

        # Verify required fields
        self.assertEqual(result.subject_id, "mouse_002")
        self.assertEqual(result.rig_id, "Rig_002")
        self.assertEqual(result.iacuc_protocol, "IACUC-67890")
        self.assertEqual(result.notes, "Minimal test")

        # Verify default values for optional fields
        self.assertEqual(result.experimenter_full_name, [])
        self.assertEqual(result.session_type, "FIB")
        self.assertIsNone(result.mouse_platform_name)
        self.assertFalse(result.active_mouse_platform)
        self.assertIsNone(result.anaesthesia)

    def test_extract_session_start_time_from_filename(self):
        """Test session start time extraction from filename."""
        extractor = FiberPhotometryExtractor(self.job_settings)

        # Test with the created file
        csv_file = self.test_data_dir / "2024-01-15_14-30-00_mouse_001_fibpho.csv"

        with patch("pathlib.Path.glob", return_value=[csv_file]):
            result = extractor._extract_session_start_time()

        expected_time = datetime(2024, 1, 15, 14, 30, 0)
        self.assertEqual(result, expected_time)

    def test_extract_session_start_time_no_files(self):
        """Test session start time extraction when no files found."""
        extractor = FiberPhotometryExtractor(self.job_settings)

        with patch("pathlib.Path.glob", return_value=[]):
            with self.assertRaises(FileNotFoundError) as context:
                extractor._extract_session_start_time()

            self.assertIn("No fiber photometry files found", str(context.exception))

    def test_extract_session_start_time_invalid_format(self):
        """Test handling of files with invalid timestamp format."""
        extractor = FiberPhotometryExtractor(self.job_settings)

        # Create file with invalid timestamp format
        invalid_file = self.test_data_dir / "invalid_timestamp_mouse_001_fibpho.csv"
        invalid_file.touch()

        with patch("pathlib.Path.glob", return_value=[invalid_file]):
            with self.assertRaises(ValueError) as context:
                extractor._extract_session_start_time()

            self.assertIn("Could not extract valid timestamp", str(context.exception))

    def test_regex_patterns(self):
        """Test regex patterns for date and mouse ID extraction."""
        # Test date regex
        date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}")

        # Valid date formats
        self.assertTrue(date_pattern.search("2024-01-15_14-30-00_mouse_001_fibpho.csv"))
        self.assertTrue(date_pattern.search("prefix_2023-12-25_09-15-30_suffix.txt"))

        # Invalid date formats
        self.assertIsNone(date_pattern.search("2024-1-15_14-30-00_mouse_001.csv"))
        self.assertIsNone(date_pattern.search("24-01-15_14-30-00_mouse_001.csv"))

        # Test mouse ID regex (assuming pattern matches the extractor's implementation)
        mouse_pattern = re.compile(r"mouse_\w+")

        self.assertTrue(mouse_pattern.search("2024-01-15_14-30-00_mouse_001_fibpho.csv"))
        self.assertTrue(mouse_pattern.search("prefix_mouse_abc123_suffix.txt"))
        self.assertIsNone(mouse_pattern.search("2024-01-15_14-30-00_subject_001_fibpho.csv"))

    def test_file_glob_pattern(self):
        """Test file globbing for fiber photometry files."""

        # Create additional test files
        fibpho_file1 = self.test_data_dir / "2024-01-15_14-30-00_mouse_001_fibpho.csv"
        other_file = self.test_data_dir / "2024-01-15_14-30-00_mouse_001_other.csv"

        # Files should be found by glob pattern
        files = list(self.test_data_dir.glob("*fibpho*"))
        self.assertIn(fibpho_file1, files)

        # Other files should not match the pattern
        fibpho_files = list(self.test_data_dir.glob("*fibpho*"))
        self.assertNotIn(other_file, fibpho_files)

    def test_job_settings_validation(self):
        """Test job settings validation."""
        # Test valid minimal settings
        valid_settings = JobSettings(
            subject_id="mouse_001", rig_id="Rig_001", iacuc_protocol="IACUC-12345", notes="Test notes"
        )
        self.assertEqual(valid_settings.subject_id, "mouse_001")
        self.assertEqual(valid_settings.job_settings_name, "FiberPhotometry")

        # Test missing required fields
        with self.assertRaises(ValueError):
            JobSettings(subject_id="mouse_001")  # Missing other required fields

    def test_job_settings_from_args(self):
        """Test job settings creation from command line arguments."""
        # Mock command line arguments
        test_args = [
            "--subject_id",
            "mouse_123",
            "--rig_id",
            "Rig_456",
            "--iacuc_protocol",
            "IACUC-789",
            "--notes",
            "Command line test",
            "--experimenter_full_name",
            "Alice",
            "Bob",
            "--session_type",
            "TEST",
            "--mouse_platform_name",
            "Platform_B",
            "--active_mouse_platform",
            "--anaesthesia",
            "ketamine",
        ]

        with patch("sys.argv", ["test"] + test_args):
            settings = JobSettings.from_args()

        self.assertEqual(settings.subject_id, "mouse_123")
        self.assertEqual(settings.rig_id, "Rig_456")
        self.assertEqual(settings.iacuc_protocol, "IACUC-789")
        self.assertEqual(settings.notes, "Command line test")
        self.assertEqual(settings.experimenter_full_name, ["Alice", "Bob"])
        self.assertEqual(settings.session_type, "TEST")
        self.assertEqual(settings.mouse_platform_name, "Platform_B")
        self.assertEqual(settings.active_mouse_platform, True)
        self.assertEqual(settings.anaesthesia, "ketamine")

    def test_extractor_initialization(self):
        """Test extractor initialization with job settings."""
        extractor = FiberPhotometryExtractor(self.job_settings)
        self.assertEqual(extractor.job_settings, self.job_settings)

        # Verify job settings are accessible
        self.assertEqual(extractor.job_settings.subject_id, "mouse_001")
        self.assertEqual(extractor.job_settings.rig_id, "Rig_001")

    def test_extract_session_start_time_multiple_files(self):
        """Test session start time extraction with multiple files."""
        extractor = FiberPhotometryExtractor(self.job_settings)

        # Create multiple files with different timestamps
        file1 = self.test_data_dir / "2024-01-15_14-30-00_mouse_001_fibpho.csv"
        file2 = self.test_data_dir / "2024-01-15_15-45-30_mouse_001_fibpho.csv"
        file3 = self.test_data_dir / "2024-01-15_09-15-00_mouse_001_fibpho.csv"

        file1.touch()
        file2.touch()
        file3.touch()

        with patch("pathlib.Path.glob", return_value=[file1, file2, file3]):
            result = extractor._extract_session_start_time()

        # Should return the earliest timestamp
        expected_time = datetime(2024, 1, 15, 9, 15, 0)
        self.assertEqual(result, expected_time)

    def test_extract_metadata_with_timing_error(self):
        """Test metadata extraction when timing extraction fails."""
        extractor = FiberPhotometryExtractor(self.job_settings)

        with patch.object(
            extractor, "_extract_session_start_time", side_effect=FileNotFoundError("No timing files found")
        ):
            with self.assertRaises(FileNotFoundError):
                extractor.extract_metadata()


if __name__ == "__main__":
    unittest.main()
