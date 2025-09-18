"""Tests for fiber photometry contract-based extractor."""

import unittest
from unittest.mock import Mock, patch
from pathlib import Path
from datetime import datetime
import tempfile

from aind_metadata_extractor.fip.extractor import FiberPhotometryExtractor
from aind_metadata_extractor.fip.job_settings import JobSettings
from aind_metadata_extractor.models.fip import FiberData


class TestFiberPhotometryExtractor(unittest.TestCase):
    """Test class for FiberPhotometryExtractor with contract-based data access."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_dir = Path(self.temp_dir)

        # Create test job settings
        self.job_settings = JobSettings(data_directory=str(self.test_data_dir))

        # Mock contract data
        self.mock_dataset = Mock()
        self.mock_dataset.metadata = {
            "mouse_platform_name": "Platform_A",
            "active_mouse_platform": True,
            "iacuc_protocol": "IACUC-12345",
            "notes": "Test experiment",
            "subject_id": "mouse_001",
            "rig_id": "Rig_001",
            "experimenter_full_name": ["John Doe", "Jane Smith"],
            "session_type": "FIB",
            "anaesthesia": "isoflurane",
        }

        # Mock data stream
        self.mock_data_stream = Mock()
        self.mock_data_stream.name = "fibpho_405_470"
        self.mock_data_stream.timing_stream = "timestamps"

        # Mock timing data
        self.mock_timing_data = Mock()
        self.mock_timing_data.index = ["session_start_time"]
        self.mock_timing_data.loc = {"session_start_time": datetime(2024, 1, 15, 14, 30, 0)}

        self.mock_dataset.data_streams = [self.mock_data_stream]

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    @patch("aind_metadata_extractor.fip.extractor.aind_physiology_fip.data_contract.dataset")
    def test_extract_metadata_success(self, mock_dataset_func):
        """Test successful metadata extraction from contract."""
        # Setup mock
        mock_dataset_func.return_value = self.mock_dataset

        # Create extractor and extract metadata
        extractor = FiberPhotometryExtractor(self.job_settings)

        with (
            patch.object(extractor, "_get_data_stream", return_value=self.mock_data_stream),
            patch.object(extractor, "_extract_index", return_value=self.mock_timing_data),
        ):

            result = extractor.extract_metadata()

        # Verify result
        self.assertIsInstance(result, FiberData)
        self.assertEqual(result.mouse_platform_name, "Platform_A")
        self.assertEqual(result.active_mouse_platform, True)
        self.assertEqual(result.iacuc_protocol, "IACUC-12345")
        self.assertEqual(result.notes, "Test experiment")
        self.assertEqual(result.subject_id, "mouse_001")
        self.assertEqual(result.rig_id, "Rig_001")
        self.assertEqual(result.experimenter_full_name, ["John Doe", "Jane Smith"])
        self.assertEqual(result.session_type, "FIB")
        self.assertEqual(result.anaesthesia, "isoflurane")
        self.assertEqual(result.session_start_time, datetime(2024, 1, 15, 14, 30, 0))

    @patch("aind_metadata_extractor.fip.extractor.aind_physiology_fip.data_contract.dataset")
    def test_extract_metadata_contract_missing(self, mock_dataset_func):
        """Test handling when contract is missing."""
        mock_dataset_func.side_effect = FileNotFoundError("Contract not found")

        extractor = FiberPhotometryExtractor(self.job_settings)

        with self.assertRaises(FileNotFoundError):
            extractor.extract_metadata()

    @patch("aind_metadata_extractor.fip.extractor.aind_physiology_fip.data_contract.dataset")
    def test_extract_metadata_missing_fields(self, mock_dataset_func):
        """Test handling of missing optional fields in contract."""
        # Create dataset with minimal metadata
        minimal_dataset = Mock()
        minimal_dataset.metadata = {
            "subject_id": "mouse_001",
            "rig_id": "Rig_001",
            "iacuc_protocol": "IACUC-12345",
            "notes": "Test experiment",
        }
        minimal_dataset.data_streams = [self.mock_data_stream]

        mock_dataset_func.return_value = minimal_dataset

        extractor = FiberPhotometryExtractor(self.job_settings)

        with (
            patch.object(extractor, "_get_data_stream", return_value=self.mock_data_stream),
            patch.object(extractor, "_extract_index", return_value=self.mock_timing_data),
        ):

            result = extractor.extract_metadata()

        # Verify required fields are present and optional fields have defaults
        self.assertEqual(result.subject_id, "mouse_001")
        self.assertEqual(result.rig_id, "Rig_001")
        self.assertIsNone(result.mouse_platform_name)
        self.assertFalse(result.active_mouse_platform)
        self.assertEqual(result.experimenter_full_name, [])

    def test_get_data_stream_success(self):
        """Test successful data stream retrieval."""
        extractor = FiberPhotometryExtractor(self.job_settings)

        # Mock multiple data streams
        stream1 = Mock()
        stream1.name = "other_stream"
        stream2 = Mock()
        stream2.name = "fibpho_405_470"
        stream3 = Mock()
        stream3.name = "another_stream"

        dataset = Mock()
        dataset.data_streams = [stream1, stream2, stream3]

        result = extractor._get_data_stream(dataset)
        self.assertEqual(result, stream2)

    def test_get_data_stream_not_found(self):
        """Test data stream not found."""
        extractor = FiberPhotometryExtractor(self.job_settings)

        # Mock data streams without the target
        stream1 = Mock()
        stream1.name = "other_stream"
        stream2 = Mock()
        stream2.name = "different_stream"

        dataset = Mock()
        dataset.data_streams = [stream1, stream2]

        with self.assertRaises(ValueError) as context:
            extractor._get_data_stream(dataset)

        self.assertIn("Data stream 'fibpho_405_470' not found", str(context.exception))

    def test_get_data_stream_empty_list(self):
        """Test handling of empty data streams list."""
        extractor = FiberPhotometryExtractor(self.job_settings)

        dataset = Mock()
        dataset.data_streams = []

        with self.assertRaises(ValueError) as context:
            extractor._get_data_stream(dataset)

        self.assertIn("Data stream 'fibpho_405_470' not found", str(context.exception))

    @patch("aind_metadata_extractor.fip.extractor.pd.read_csv")
    def test_extract_index_success(self, mock_read_csv):
        """Test successful index extraction."""
        # Mock CSV data
        mock_df = Mock()
        mock_df.index = ["session_start_time", "session_end_time"]
        mock_df.loc = {"session_start_time": datetime(2024, 1, 15, 14, 30, 0)}
        mock_read_csv.return_value = mock_df

        extractor = FiberPhotometryExtractor(self.job_settings)

        # Mock data stream with timing configuration
        data_stream = Mock()
        data_stream.timing_stream = "timestamps"
        data_stream.timing_stream_config = {"index_key": "session_start_time"}

        # Mock dataset directory structure
        test_csv_path = self.test_data_dir / "timestamps.csv"
        test_csv_path.parent.mkdir(parents=True, exist_ok=True)
        test_csv_path.touch()

        result = extractor._extract_index(data_stream, self.test_data_dir)

        self.assertEqual(result, mock_df)
        mock_read_csv.assert_called_once_with(test_csv_path, index_col=0)

    def test_extract_index_file_not_found(self):
        """Test handling when timing CSV file is not found."""
        extractor = FiberPhotometryExtractor(self.job_settings)

        data_stream = Mock()
        data_stream.timing_stream = "missing_timestamps"

        with self.assertRaises(FileNotFoundError):
            extractor._extract_index(data_stream, self.test_data_dir)

    def test_extract_metadata_from_contract_success(self):
        """Test successful metadata extraction from contract object."""
        extractor = FiberPhotometryExtractor(self.job_settings)

        with (
            patch.object(extractor, "_get_data_stream", return_value=self.mock_data_stream),
            patch.object(extractor, "_extract_index", return_value=self.mock_timing_data),
        ):

            result = extractor._extract_metadata_from_contract(self.mock_dataset)

        self.assertIsInstance(result, FiberData)
        self.assertEqual(result.subject_id, "mouse_001")
        self.assertEqual(result.session_start_time, datetime(2024, 1, 15, 14, 30, 0))

    def test_extract_metadata_from_contract_timing_extraction_error(self):
        """Test handling of timing extraction errors."""
        extractor = FiberPhotometryExtractor(self.job_settings)

        # Mock data stream that will cause timing extraction to fail
        failing_data_stream = Mock()
        failing_data_stream.timing_stream = "invalid_timing"

        with (
            patch.object(extractor, "_get_data_stream", return_value=failing_data_stream),
            patch.object(extractor, "_extract_index", side_effect=FileNotFoundError("Timing file not found")),
        ):

            with self.assertRaises(FileNotFoundError):
                extractor._extract_metadata_from_contract(self.mock_dataset)

    def test_job_settings_validation(self):
        """Test job settings validation."""
        # Test valid settings
        valid_settings = JobSettings(data_directory="/path/to/data")
        self.assertEqual(str(valid_settings.data_directory), "/path/to/data")
        self.assertEqual(valid_settings.job_settings_name, "FiberPhotometry")

        # Test missing required field
        with self.assertRaises(ValueError):
            JobSettings()

    def test_extractor_initialization(self):
        """Test extractor initialization with job settings."""
        extractor = FiberPhotometryExtractor(self.job_settings)
        self.assertEqual(extractor.job_settings, self.job_settings)
        self.assertEqual(extractor.job_settings.data_directory, str(self.test_data_dir))


if __name__ == "__main__":
    unittest.main()
