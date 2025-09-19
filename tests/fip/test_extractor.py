"""Tests for fiber photometry contract-based extractor."""

import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
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
        self.job_settings = JobSettings(
            data_directory=str(self.test_data_dir),
            experimenter_full_name=["John Doe", "Jane Smith"],
            subject_id="mouse_001",
            rig_id="Rig_001",
            mouse_platform_name="Platform_A",
            active_mouse_platform=True,
            iacuc_protocol="IACUC-12345",
            notes="Test experiment",
            session_type="FIB",
            anaesthesia="isoflurane",
        )

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    @patch("aind_metadata_extractor.fip.extractor.dataset")
    def test_extract_success(self, mock_dataset):
        """Test successful extraction using contract."""
        # Mock contract dataset and streams
        mock_stream = MagicMock()
        mock_stream.name = "green"
        mock_stream.reader_params.index = "ReferenceTime"
        mock_stream.reader_params.path = self.test_data_dir / "green.csv"
        mock_stream.read.return_value = MagicMock(
            columns=["ReferenceTime"],
            __getitem__=lambda self, key: [933019.553312, 933754.601152],
            index=MagicMock(empty=False, min=lambda: 933019.553312, max=lambda: 933754.601152)
        )

        mock_dataset.return_value._data = [mock_stream]
        mock_dataset.return_value.data_streams = [mock_stream]

        extractor = FiberPhotometryExtractor(self.job_settings)
        with patch.object(extractor, "_get_data_stream", return_value=mock_stream):
            with patch.object(extractor, "_extract_timing_from_csv", return_value={
                "start_time": 933019.553312,
                "end_time": 933754.601152
            }):
                with patch.object(extractor, "_extract_data_files", return_value={
                    "data_files": [str(self.test_data_dir / "green.csv")]
                }):
                    with patch.object(extractor, "_extract_hardware_config", return_value={
                        "rig_config": {"rig_name": "Rig_001"},
                        "session_config": {"session_type": "FIB"}
                    }):
                        fiber_data = extractor.extract()
        self.assertIn("session_start_time", fiber_data)
        self.assertIn("session_end_time", fiber_data)
        self.assertIn("data_files", fiber_data)
        self.assertIn("rig_config", fiber_data)
        self.assertIn("session_config", fiber_data)
        self.assertEqual(fiber_data["rig_config"]["rig_name"], "Rig_001")
        self.assertEqual(fiber_data["session_config"]["session_type"], "FIB")

    @patch("aind_metadata_extractor.fip.extractor.dataset")
    def test_extract_basic_metadata_fallback(self, mock_dataset):
        """Test fallback to basic metadata extraction."""
        extractor = FiberPhotometryExtractor(self.job_settings)
        with patch.object(extractor, "_extract_timing_from_csv", side_effect=Exception("fail")):
            with patch.object(extractor, "_extract_data_files", return_value={"data_files": []}):
                with patch.object(extractor, "_extract_hardware_config", return_value={}):
                    result = extractor._extract_basic_metadata()
        self.assertIn("start_time", result)
        self.assertIn("end_time", result)
        self.assertIn("data_files", result)
        self.assertIn("rig_config", result)
        self.assertIn("session_config", result)

    def test_save_to_file(self):
        """Test saving FiberData to file."""
        extractor = FiberPhotometryExtractor(self.job_settings)
        fiber_data = FiberData(
            job_settings_name="FiberPhotometry",
            experimenter_full_name=["John Doe"],
            subject_id="mouse_001",
            rig_id="Rig_001",
            mouse_platform_name="Platform_A",
            active_mouse_platform=True,
            data_streams=[],
            session_type="FIB",
            iacuc_protocol="IACUC-12345",
            notes="Test experiment",
            anaesthesia="isoflurane",
            protocol_id=[],
            data_directory=str(self.test_data_dir),
            data_files=[str(self.test_data_dir / "green.csv")],
            rig_config={"rig_name": "Rig_001"},
            session_config={"session_type": "FIB"},
            local_timezone="America/Los_Angeles",
            output_directory=str(self.test_data_dir),
            output_filename="session_fip.json",
        )
        output_path = extractor.save_to_file(fiber_data)
        self.assertTrue(Path(output_path).exists())
        with open(output_path, "r") as f:
            data = f.read()
        self.assertIn("FiberPhotometry", data)
        self.assertIn("Rig_001", data)


if __name__ == "__main__":
    unittest.main()
