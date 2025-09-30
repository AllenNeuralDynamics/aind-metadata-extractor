"""Tests for fiber photometry legacy extractor."""

import unittest
from unittest.mock import patch
from pathlib import Path
from datetime import datetime
import tempfile
import re

from aind_metadata_extractor.fip_legacy.extractor import FiberPhotometryExtractor
from aind_metadata_extractor.fip_legacy.job_settings import JobSettings
from aind_metadata_extractor.models.fip_legacy import FiberData


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
            data_directory=str(self.test_data_dir),  # <-- Add this line
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

    def test_extract_no_data_files(self):
        """Test error when no data files are found."""

        job_settings = JobSettings(
            subject_id="mouse_001",
            rig_id="Rig_001",
            iacuc_protocol="IACUC-12345",
            notes="Test experiment notes",
            data_directory=str("testingdir"),  # Non-existent directory
        )

        extractor = FiberPhotometryExtractor(job_settings)

        with self.assertRaises(FileNotFoundError) as context:
            extractor.extract_metadata()
        self.assertIn("Data directory testingdir does not exist", str(context.exception))

    def test_extract_no_csv_files(self):
        """Test error when no CSV data files are found."""
        empty_dir = tempfile.mkdtemp()
        job_settings = JobSettings(
            subject_id="mouse_001",
            rig_id="Rig_001",
            iacuc_protocol="IACUC-12345",
            notes="Test experiment notes",
            data_directory=str(empty_dir),  # Empty directory
        )

        extractor = FiberPhotometryExtractor(job_settings)

        with self.assertRaises(FileNotFoundError) as context:
            extractor.extract_metadata()
        self.assertIn("No CSV data files found in ", str(context.exception))

    def test_extract(self):
        """Test the extract() method for full coverage."""
        extractor = FiberPhotometryExtractor(self.job_settings)

        # Prepare a minimal valid metadata dict for FiberData
        metadata = {
            "job_settings_name": "FiberPhotometry",
            "experimenter_full_name": ["John Doe", "Jane Smith"],
            "subject_id": "mouse_001",
            "rig_id": "Rig_001",
            "iacuc_protocol": "IACUC-12345",
            "notes": "Test experiment notes",
            "session_type": "FIB",
            "mouse_platform_name": "Platform_A",
            "active_mouse_platform": True,
            "anaesthesia": "isoflurane",
            "session_start_time": datetime(2024, 1, 15, 14, 30, 0),
            "session_end_time": datetime(2024, 1, 15, 15, 30, 0),
            "data_directory": str(self.test_data_dir),
        }

        with patch.object(extractor, "_extract_metadata_from_data_files", return_value=metadata):
            result = extractor.extract()
            self.assertIsInstance(result, dict)
            self.assertEqual(result["subject_id"], "mouse_001")
            self.assertEqual(result["rig_id"], "Rig_001")
            self.assertEqual(result["experimenter_full_name"], ["John Doe", "Jane Smith"])
            self.assertEqual(result["notes"], "Test experiment notes")
            self.assertEqual(result["session_type"], "FIB")
            self.assertEqual(result["mouse_platform_name"], "Platform_A")
            self.assertEqual(result["active_mouse_platform"], True)
            self.assertEqual(result["anaesthesia"], "isoflurane")

    def test_extract_metadata_minimal_settings(self):
        """Test metadata extraction with minimal job settings."""
        # Provide a valid data_directory to avoid FileNotFoundError
        minimal_settings = JobSettings(
            subject_id="mouse_002",
            rig_id="Rig_002",
            iacuc_protocol="IACUC-67890",
            notes="Minimal test",
            data_directory=str(self.test_data_dir),
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
        data_files = [csv_file]

        result = extractor._extract_session_start_time(data_files)

        expected_time = datetime(2024, 1, 15, 14, 30, 0)
        self.assertEqual(result, expected_time)

    def test_extract_session_start_time_invalid_format(self):
        """Test handling of files with invalid timestamp format."""
        extractor = FiberPhotometryExtractor(self.job_settings)

        # Create file with invalid timestamp format
        invalid_file = self.test_data_dir / "invalid_timestamp_mouse_001_fibpho.csv"
        invalid_file.touch()

        data_files = [invalid_file]
        with self.assertRaises(ValueError) as context:
            extractor._extract_session_start_time(data_files)

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

    def test_extractor_initialization(self):
        """Test extractor initialization with job settings."""
        extractor = FiberPhotometryExtractor(self.job_settings)
        self.assertEqual(extractor.job_settings, self.job_settings)

        # Verify job settings are accessible
        self.assertEqual(extractor.job_settings.subject_id, "mouse_001")
        self.assertEqual(extractor.job_settings.rig_id, "Rig_001")

    def test_extract_metadata_with_timing_error(self):
        """Test metadata extraction when timing extraction fails."""
        extractor = FiberPhotometryExtractor(self.job_settings)

        with patch.object(
            extractor, "_extract_session_start_time", side_effect=FileNotFoundError("No timing files found")
        ):
            with self.assertRaises(FileNotFoundError):
                extractor.extract_metadata()

    def test_extract_metadata_from_data_files_invalid_path_type(self):
        """Test ValueError when data_directory is not a string or Path."""
        extractor = FiberPhotometryExtractor(self.job_settings)
        extractor.job_settings.data_directory = 12345  # Invalid type
        with self.assertRaises(ValueError) as context:
            extractor._extract_metadata_from_data_files()
        self.assertIn("data_directory must be a valid path", str(context.exception))

    def test_extract_metadata_from_data_files_missing_directory(self):
        """Test FileNotFoundError when directory does not exist."""
        extractor = FiberPhotometryExtractor(self.job_settings)
        extractor.job_settings.data_directory = "nonexistent_dir"
        with self.assertRaises(FileNotFoundError) as context:
            extractor._extract_metadata_from_data_files()
        self.assertIn("Data directory nonexistent_dir does not exist", str(context.exception))

    def test_extract_metadata_from_data_files_no_data_files(self):
        """Test FileNotFoundError when no data files are found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = FiberPhotometryExtractor(self.job_settings)
            extractor.job_settings.data_directory = tmpdir
            # Remove all files
            with self.assertRaises(FileNotFoundError) as context:
                extractor._extract_metadata_from_data_files()
            self.assertIn("No data files found in", str(context.exception))

    def test_extract_metadata_from_data_files_hardware_configs(self):
        """Test extraction of hardware configs from data_streams."""
        extractor = FiberPhotometryExtractor(self.job_settings)
        extractor.job_settings.data_streams = [
            {"light_sources": ["ls1", "ls2"], "detectors": ["det1"], "fiber_connections": ["fc1"]}
        ]
        # Patch file finding and timing
        with (
            patch.object(extractor, "_extract_session_timing", return_value=(None, None)),
            patch.object(extractor, "_extract_timestamps", return_value=[]),
            patch("pathlib.Path.glob", return_value=[Path(__file__)]),
        ):
            result = extractor._extract_metadata_from_data_files()
            self.assertEqual(result["light_source_configs"], ["ls1", "ls2"])
            self.assertEqual(result["detector_configs"], ["det1"])
            self.assertEqual(result["fiber_configs"], ["fc1"])

    def test_extract_metadata_from_data_files_no_hardware_configs(self):
        """Test extraction when data_streams is empty."""
        extractor = FiberPhotometryExtractor(self.job_settings)
        extractor.job_settings.data_streams = []
        # Patch file finding and timing
        with (
            patch.object(extractor, "_extract_session_timing", return_value=(None, None)),
            patch.object(extractor, "_extract_timestamps", return_value=[]),
            patch("pathlib.Path.glob", return_value=[Path(__file__)]),
        ):
            result = extractor._extract_metadata_from_data_files()
            self.assertEqual(result["light_source_configs"], [])
            self.assertEqual(result["detector_configs"], [])
            self.assertEqual(result["fiber_configs"], [])

    def test_extract_timestamps_with_timestamp_column(self):
        """Test _extract_timestamps with a timestamp column."""
        extractor = FiberPhotometryExtractor(self.job_settings)
        # Create a CSV file with a timestamp column
        csv_file = self.test_data_dir / "timestamp_test.csv"
        with open(csv_file, "w") as f:
            f.write("timestamp,value\n2024-01-15 14:30:00,1\n2024-01-15 14:30:01,2\n")
        result = extractor._extract_timestamps([csv_file])
        self.assertEqual(result, None)

    def test_extract_timestamps_no_timestamp_column(self):
        """Test _extract_timestamps with no timestamp column."""
        extractor = FiberPhotometryExtractor(self.job_settings)
        csv_file = self.test_data_dir / "no_timestamp.csv"
        with open(csv_file, "w") as f:
            f.write("value\n1\n2\n3\n")
        result = extractor._extract_timestamps([csv_file])
        self.assertEqual(result, None)

    def test_extract_session_end_time_no_timestamps(self):
        """Test _extract_session_end_time returns None when no timestamps."""
        extractor = FiberPhotometryExtractor(self.job_settings)
        result = extractor._extract_session_end_time([])
        self.assertIsNone(result)

    def test_extract_session_timing_no_files(self):
        """Test _extract_session_timing returns (None, None) when no files."""
        extractor = FiberPhotometryExtractor(self.job_settings)
        result = extractor._extract_session_timing([])
        self.assertEqual(result, (None, None))

    def test_extract_timestamps_file_read_error(self):
        """Test _extract_timestamps skips unreadable files."""
        extractor = FiberPhotometryExtractor(self.job_settings)
        # Patch pandas.read_csv to raise error
        with patch("pandas.read_csv", side_effect=Exception("fail")):
            result = extractor._extract_timestamps([Path("fake.csv")])
            self.assertEqual(result, None)

    def test_extract_session_timing_with_timestamp_column(self):
        """Test _extract_session_timing with timestamp column."""
        extractor = FiberPhotometryExtractor(self.job_settings)
        csv_file = self.test_data_dir / "timing_test.csv"
        with open(csv_file, "w") as f:
            f.write("timestamp,value\n2024-01-15 14:30:00,1\n2024-01-15 14:31:00,2\n")
        start, end = extractor._extract_session_timing([csv_file])
        self.assertEqual(start, datetime(2024, 1, 15, 14, 30, 0))
        self.assertEqual(end, datetime(2024, 1, 15, 14, 31, 0))

    def test_extract_session_timing_no_timestamp_column(self):
        """Test _extract_session_timing falls back to filename."""
        extractor = FiberPhotometryExtractor(self.job_settings)
        csv_file = self.test_data_dir / "2024-01-15_14-30-00_mouse_001_fibpho.csv"
        with open(csv_file, "w") as f:
            f.write("value\n1\n2\n")
        start, end = extractor._extract_session_timing([csv_file])
        self.assertEqual(start, datetime(2024, 1, 15, 14, 30, 0))
        self.assertIsNone(end)

    def test_extract_timing_from_filename(self):
        """Test _extract_timing_from_filename returns correct start time."""
        extractor = FiberPhotometryExtractor(self.job_settings)
        csv_file = self.test_data_dir / "2024-01-15_14-30-00_mouse_001_fibpho.csv"
        start, end = extractor._extract_timing_from_filename(csv_file)
        self.assertEqual(start, datetime(2024, 1, 15, 14, 30, 0))
        self.assertIsNone(end)

    def test_extract_timing_from_filename_no_match(self):
        """Test _extract_timing_from_filename returns None if no match."""
        extractor = FiberPhotometryExtractor(self.job_settings)
        csv_file = self.test_data_dir / "no_date_here.csv"
        start, end = extractor._extract_timing_from_filename(csv_file)
        self.assertIsNone(start)
        self.assertIsNone(end)


if __name__ == "__main__":
    unittest.main()
