"""Tests for fiber photometry contract-based extractor."""

import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
from datetime import datetime

from aind_metadata_extractor.fip.extractor import FiberPhotometryExtractor
from aind_metadata_extractor.fip.job_settings import JobSettings
from aind_metadata_extractor.models.fip import FIPDataModel


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
            session_start_time=datetime(2023, 1, 1, 12, 0),
            session_end_time=datetime(2023, 1, 1, 13, 0),
            animal_weight_post=25.0,
            animal_weight_prior=24.5,
            rig_config={
                "rig_name": "Rig_001",
            },
            session_config={
                "session_type": "FIB"
            },
            output_directory=str(self.test_data_dir),
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
            index=MagicMock(empty=False, min=lambda: 933019.553312, max=lambda: 933754.601152),
        )

        mock_dataset.return_value._data = [mock_stream]
        mock_dataset.return_value.data_streams = [mock_stream]

        extractor = FiberPhotometryExtractor(self.job_settings)
        with patch.object(extractor, "_get_data_stream", return_value=mock_stream):
            with patch.object(
                extractor,
                "_extract_timing_from_csv",
                return_value={"start_time": 933019.553312, "end_time": 933754.601152},
            ):
                with patch.object(
                    extractor,
                    "_extract_data_files",
                    return_value={"data_files": [str(self.test_data_dir / "green.csv")]},
                ):
                    with patch.object(
                        extractor,
                        "_extract_hardware_config",
                        return_value={"rig_config": {"rig_name": "Rig_001"}, "session_config": {"session_type": "FIB"}},
                    ):
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
        """Test saving FIPDataModel to file."""
        extractor = FiberPhotometryExtractor(self.job_settings)
        fiber_data = FIPDataModel(
            job_settings_name="FIP",
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
            session_start_time=datetime(2023, 1, 1, 12, 0),
            session_end_time=datetime(2023, 1, 1, 13, 0),
            animal_weight_post=25.0,
            animal_weight_prior=24.5,
            output_directory=str(self.test_data_dir),
            output_filename="session_fip.json",
        )
        output_path = extractor.save_to_file(fiber_data)
        self.assertTrue(Path(output_path).exists())
        with open(output_path, "r") as f:
            data = f.read()
        self.assertIn("FIP", data)
        self.assertIn("Rig_001", data)

    def test_extract_index(self):
        """Test _extract_index method for green, red, and default fallback."""
        extractor = FiberPhotometryExtractor(self.job_settings)

        # Mock green stream with index
        green_stream = MagicMock()
        green_stream.reader_params.index = "GreenIndex"
        with patch.object(
            extractor, "_get_data_stream", side_effect=lambda color: green_stream if color == "green" else None
        ):
            result = extractor._extract_index()
            self.assertEqual(result["index_key"], "GreenIndex")

        # Mock green stream without index, red stream with index
        green_stream_no_index = MagicMock()
        green_stream_no_index.reader_params.index = None
        red_stream = MagicMock()
        red_stream.reader_params.index = "RedIndex"
        with patch.object(
            extractor,
            "_get_data_stream",
            side_effect=lambda color: (
                green_stream_no_index if color == "green" else red_stream if color == "red" else None
            ),
        ):
            result = extractor._extract_index()
            self.assertEqual(result["index_key"], "RedIndex")

        # Both streams missing index, should fallback to default
        green_stream_no_index = MagicMock()
        green_stream_no_index.reader_params.index = None
        red_stream_no_index = MagicMock()
        red_stream_no_index.reader_params.index = None
        with patch.object(
            extractor,
            "_get_data_stream",
            side_effect=lambda color: (
                green_stream_no_index if color == "green" else red_stream_no_index if color == "red" else None
            ),
        ):
            result = extractor._extract_index()
            self.assertEqual(result["index_key"], "ReferenceTime")

    def test_extract_timing_from_csv(self):
        """Test _extract_timing_from_csv for green, red, and fallback cases."""
        extractor = FiberPhotometryExtractor(self.job_settings)

        # Case 1: Green stream has index key in columns
        green_stream = MagicMock()
        green_data = MagicMock()
        green_data.columns = ["ReferenceTime"]
        # Mock the Series returned by green_data["ReferenceTime"]
        reference_time_series = MagicMock()
        reference_time_series.min.return_value = 1
        reference_time_series.max.return_value = 2
        green_data.__getitem__.side_effect = lambda key: (
            reference_time_series if key == "ReferenceTime" else MagicMock()
        )
        green_data.index = MagicMock(empty=False, min=lambda: 1, max=lambda: 2)
        green_stream.read.return_value = green_data

        with (
            patch.object(
                extractor, "_get_data_stream", side_effect=lambda color: green_stream if color == "green" else None
            ),
            patch.object(extractor, "_extract_index", return_value={"index_key": "ReferenceTime"}),
        ):
            timing = extractor._extract_timing_from_csv()
            self.assertEqual(timing["start_time"], 1)
            self.assertEqual(timing["end_time"], 2)

        # Case 2: Green stream missing, red stream has index key in columns
        red_stream = MagicMock()
        red_data = MagicMock()
        red_data.columns = ["ReferenceTime"]
        reference_time_series_red = MagicMock()
        reference_time_series_red.min.return_value = 3
        reference_time_series_red.max.return_value = 4
        red_data.__getitem__.side_effect = lambda key: (
            reference_time_series_red if key == "ReferenceTime" else MagicMock()
        )
        red_data.index = MagicMock(empty=False, min=lambda: 3, max=lambda: 4)
        red_stream.read.return_value = red_data

        with (
            patch.object(
                extractor, "_get_data_stream", side_effect=lambda color: red_stream if color == "red" else None
            ),
            patch.object(extractor, "_extract_index", return_value={"index_key": "ReferenceTime"}),
        ):
            timing = extractor._extract_timing_from_csv()
            self.assertEqual(timing["start_time"], 3)
            self.assertEqual(timing["end_time"], 4)

        # Case 3: Red stream has no index key in columns but has index
        red_data_no_col = MagicMock()
        red_data_no_col.columns = []
        red_data_no_col.index = MagicMock(empty=False, min=lambda: 5, max=lambda: 6)
        red_stream.read.return_value = red_data_no_col

        with (
            patch.object(
                extractor, "_get_data_stream", side_effect=lambda color: red_stream if color == "red" else None
            ),
            patch.object(extractor, "_extract_index", return_value={"index_key": "ReferenceTime"}),
        ):
            timing = extractor._extract_timing_from_csv()
            self.assertEqual(timing["start_time"], 5)
            self.assertEqual(timing["end_time"], 6)

        # Case 4: Both streams missing, should fallback to now
        with (
            patch.object(extractor, "_get_data_stream", return_value=None),
            patch.object(extractor, "_extract_index", return_value={"index_key": "ReferenceTime"}),
        ):
            timing = extractor._extract_timing_from_csv()
            self.assertTrue(isinstance(timing["start_time"], datetime))
            self.assertTrue(isinstance(timing["end_time"], datetime))

    def test_extract_data_files(self):
        """Test _extract_data_files method for existing and non-existing files."""
        extractor = FiberPhotometryExtractor(self.job_settings)

        # Case 1: Both files exist
        green_stream = MagicMock()
        green_stream.reader_params.path = self.test_data_dir / "green.csv"
        red_stream = MagicMock()
        red_stream.reader_params.path = self.test_data_dir / "red.csv"

        # Actually create the files so Path.exists() returns True
        Path(green_stream.reader_params.path).touch()
        Path(red_stream.reader_params.path).touch()

        with patch.object(
            extractor,
            "_get_data_stream",
            side_effect=lambda color: green_stream if color == "green" else red_stream if color == "red" else None,
        ):
            result = extractor._extract_data_files()
            self.assertIn(str(self.test_data_dir / "green.csv"), result["data_files"])
            self.assertIn(str(self.test_data_dir / "red.csv"), result["data_files"])

        # Case 2: Files do not exist
        green_stream.reader_params.path = self.test_data_dir / "missing_green.csv"
        red_stream.reader_params.path = self.test_data_dir / "missing_red.csv"
        with patch.object(
            extractor,
            "_get_data_stream",
            side_effect=lambda color: green_stream if color == "green" else red_stream if color == "red" else None,
        ):
            result = extractor._extract_data_files()
            self.assertNotIn(str(self.test_data_dir / "missing_green.csv"), result["data_files"])
            self.assertNotIn(str(self.test_data_dir / "missing_red.csv"), result["data_files"])

    def test_extract_hardware_config(self):
        """Test _extract_hardware_config for rig and session streams."""
        extractor = FiberPhotometryExtractor(self.job_settings)

        # Mock rig stream with model_dump
        rig_stream = MagicMock()
        rig_data = MagicMock()
        rig_data.model_dump.return_value = {"rig_name": "Rig_001"}
        rig_stream.read.return_value = rig_data

        # Mock session stream with model_dump
        session_stream = MagicMock()
        session_data = MagicMock()
        session_data.model_dump.return_value = {"session_type": "FIB"}
        session_stream.read.return_value = session_data

        with patch.object(
            extractor,
            "_get_data_stream",
            side_effect=lambda name: (
                rig_stream if name == "rig_input" else session_stream if name == "session_input" else None
            ),
        ):
            result = extractor._extract_hardware_config()
            self.assertEqual(result["rig_config"]["rig_name"], "Rig_001")
            self.assertEqual(result["session_config"]["session_type"], "FIB")

    def test_get_data_stream(self):
        """Test _get_data_stream returns correct stream or None."""
        extractor = FiberPhotometryExtractor(self.job_settings)
        # Mock dataset with two streams
        green_stream = MagicMock()
        green_stream.name = "green"
        red_stream = MagicMock()
        red_stream.name = "red"
        extractor._dataset = MagicMock()
        extractor._dataset._data = [green_stream, red_stream]

        # Should return green stream
        result = extractor._get_data_stream("green")
        self.assertIs(result, green_stream)

        # Should return red stream
        result = extractor._get_data_stream("red")
        self.assertIs(result, red_stream)

        # Should return None for missing stream
        result = extractor._get_data_stream("blue")
        self.assertIsNone(result)

    @patch("aind_metadata_extractor.fip.extractor.dataset")
    def test_extract_full_contract(self, mock_dataset):
        """Test extract method for full contract-based extraction."""
        extractor = FiberPhotometryExtractor(self.job_settings)

        mock_dataset_instance = MagicMock()
        mock_dataset.return_value = mock_dataset_instance

        # Patch _extract_metadata_from_contract to return valid metadata
        metadata = {
            "job_settings_name": "FiberPhotometry",
            "experimenter_full_name": ["John Doe", "Jane Smith"],
            "subject_id": "mouse_001",
            "rig_id": "Rig_001",
            "mouse_platform_name": "Platform_A",
            "active_mouse_platform": True,
            "iacuc_protocol": "IACUC-12345",
            "notes": "Test experiment",
            "session_type": "FIB",
            "anaesthesia": "isoflurane",
            "session_start_time": 933019.553312,
            "session_end_time": 933754.601152,
            "data_files": [str(self.test_data_dir / "green.csv")],
            "rig_config": {"rig_name": "Rig_001"},
            "session_config": {"session_type": "FIB"},
            "local_timezone": "America/Los_Angeles",
            "output_directory": str(self.test_data_dir),
            "output_filename": "session_fip.json",
        }
        with patch.object(extractor, "_extract_metadata_from_contract", return_value=metadata):
            result = extractor.extract()
        self.assertIsInstance(result, dict)
        self.assertEqual(result["subject_id"], "mouse_001")
        self.assertEqual(result["rig_id"], "Rig_001")
        self.assertEqual(result["rig_config"]["rig_name"], "Rig_001")
        self.assertEqual(result["session_config"]["session_type"], "FIB")

    def test_extract_metadata_from_contract(self):
        """Test _extract_metadata_from_contract aggregates all metadata."""
        extractor = FiberPhotometryExtractor(self.job_settings)
        extractor.dataset = MagicMock()  # Simulate contract dataset

        timing_data = {"start_time": 1, "end_time": 2}
        files_data = {"data_files": [str(self.test_data_dir / "green.csv")]}
        hardware_data = {"rig_config": {"rig_name": "Rig_001"}, "session_config": {"session_type": "FIB"}}

        with (
            patch.object(extractor, "_extract_timing_from_csv", return_value=timing_data),
            patch.object(extractor, "_extract_data_files", return_value=files_data),
            patch.object(extractor, "_extract_hardware_config", return_value=hardware_data),
        ):
            metadata = extractor._extract_metadata_from_contract()

        self.assertEqual(metadata["start_time"], 1)
        self.assertEqual(metadata["end_time"], 2)
        self.assertIn(str(self.test_data_dir / "green.csv"), metadata["data_files"])
        self.assertEqual(metadata["rig_config"]["rig_name"], "Rig_001")
        self.assertEqual(metadata["session_config"]["session_type"], "FIB")


if __name__ == "__main__":
    unittest.main()
