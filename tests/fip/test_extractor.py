"""Tests for fiber photometry contract-based extractor."""

import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
from datetime import datetime

from aind_metadata_extractor.fip.extractor import FiberPhotometryExtractor
from aind_metadata_extractor.fip.job_settings import JobSettings
from aind_metadata_extractor.models.fip import FIPDataModel
from contraqctor.contract import FilePathBaseParam


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
            ethics_review_id="IACUC-12345",
            notes="Test experiment",
            session_type="FIB",
            session_start_time=datetime(2023, 1, 1, 12, 0),
            session_end_time=datetime(2023, 1, 1, 13, 0),
            animal_weight_post=25.0,
            animal_weight_prior=24.5,
            rig_config={
                "rig_name": "Rig_001",
            },
            session_config={"session_type": "FIB"},
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
                    return_value=({"rig_name": "Rig_001"}, {"session_type": "FIB"}),
                ):
                    fiber_data = extractor.extract()
        self.assertIn("session_start_time", fiber_data)
        self.assertIn("session_end_time", fiber_data)
        self.assertIn("data_files", fiber_data)
        self.assertIn("rig_config", fiber_data)
        self.assertIn("session_config", fiber_data)
        self.assertEqual(fiber_data["rig_config"]["rig_name"], "Rig_001")
        self.assertEqual(fiber_data["session_config"]["session_type"], "FIB")


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
            ethics_review_id="IACUC-12345",
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


    def test_extract_timing_from_csv(self):
        """Test _extract_timing_from_csv for green, red, and fallback cases."""
        extractor = FiberPhotometryExtractor(self.job_settings)

        # --- Case 1: Green stream with valid CpuTime ---
        green_stream = MagicMock()
        green_data = MagicMock()
        green_data.columns = ["CpuTime"]
        green_data.empty = False
        cpu_time_series = MagicMock()
        cpu_time_series.iloc = ["2023-01-01T04:00:00+00:00", "2023-01-01T05:00:00+00:00"]
        green_data.__getitem__.side_effect = lambda key: cpu_time_series if key == "CpuTime" else MagicMock()
        green_stream.read.return_value = green_data

        extractor._dataset = MagicMock()
        extractor._dataset.__getitem__.side_effect = lambda name: green_stream if name == "camera_green_iso_metadata" else MagicMock()

        timing = extractor._extract_timing_from_csv()
        self.assertEqual(timing[0].hour, 20)
        self.assertEqual(timing[1].hour, 21)

        # --- Case 2: Both streams missing, should error ---
        empty_stream = MagicMock()
        empty_data = MagicMock()
        empty_data.columns = []
        empty_data.empty = True
        empty_stream.read.return_value = empty_data
        extractor._dataset.__getitem__.side_effect = lambda name: empty_stream

        with self.assertRaises(ValueError) as context:
            extractor._extract_timing_from_csv()
        self.assertIn(
            "Could not extract session timing from camera metadata. "
            "Expected to find CpuTime column in camera_green_iso_metadata.csv or camera_red_metadata.csv. "
            "Please verify that camera metadata files exist in the data directory.",
            str(context.exception),
        )

    def test_extract_data_files(self):
        """Test _extract_data_files method for existing and non-existing files."""
        extractor = FiberPhotometryExtractor(self.job_settings)

        # Case 1: Both files exist
        green_stream = MagicMock()
        green_stream.reader_params = FilePathBaseParam(path=self.test_data_dir / "green.csv")
        red_stream = MagicMock()
        red_stream.reader_params = FilePathBaseParam(path=self.test_data_dir / "red.csv")

        Path(green_stream.reader_params.path).touch()
        Path(red_stream.reader_params.path).touch()

        extractor._dataset = MagicMock()
        extractor._dataset.__getitem__.side_effect = lambda name: green_stream if "green" in name else red_stream

        result = extractor._extract_data_files()
        self.assertIn(str(self.test_data_dir / "green.csv"), result)
        self.assertIn(str(self.test_data_dir / "red.csv"), result)

        # Case 2: Files do not exist
        green_stream.reader_params = FilePathBaseParam(path=self.test_data_dir / "missing_green.csv")
        red_stream.reader_params = FilePathBaseParam(path=self.test_data_dir / "missing_red.csv")
        result = extractor._extract_data_files()
        self.assertNotIn(str(self.test_data_dir / "missing_green.csv"), result)
        self.assertNotIn(str(self.test_data_dir / "missing_red.csv"), result)

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
        session_data.model_dump.return_value = {
            "session_type": "FIB",
            "experimenter_full_name": ["John Doe", "Jane Smith"],
        }
        session_stream.read.return_value = session_data

        result = extractor._extract_hardware_config()
        self.assertEqual(result["rig_config"]["rig_name"], "Rig_001")
        self.assertEqual(result["session_config"]["session_type"], "FIB")
        self.assertEqual(result["session_config"]["experimenter_full_name"], ["John Doe", "Jane Smith"])

    def test_extract_hardware_config_no_rig_dump(self):
        """Test _extract_hardware_config raises error if model_dump is missing."""
        extractor = FiberPhotometryExtractor(self.job_settings)

        # Mock rig stream without model_dump
        rig_stream = MagicMock()
        rig_data = MagicMock()
        if hasattr(rig_data, "model_dump"):
            del rig_data.model_dump
        rig_stream.read.return_value = rig_data


        with self.assertRaises(AttributeError) as context:
            extractor._extract_hardware_config()
        self.assertIn("Rig data must have a 'model_dump' method", str(context.exception))

    def test_extract_hardware_no_session_dump(self):
        """Test _extract_hardware_config raises error if session model_dump is missing."""
        extractor = FiberPhotometryExtractor(self.job_settings)

        # Mock session stream without model_dump
        session_stream = MagicMock()
        session_data = MagicMock()
        if hasattr(session_data, "model_dump"):
            del session_data.model_dump
        session_stream.read.return_value = session_data


        with self.assertRaises(AttributeError) as context:
            extractor._extract_hardware_config()
        self.assertIn("Session data must have a 'model_dump' method", str(context.exception))

    @patch("aind_metadata_extractor.fip.extractor.dataset")
    def test_extract_full_contract(self, mock_dataset):
        """Test extract method for full contract-based extraction."""
        extractor = FiberPhotometryExtractor(self.job_settings)

        mock_dataset_instance = MagicMock()
        mock_dataset.return_value = mock_dataset_instance

        # Patch _extract_metadata_from_contract to return valid metadata
        metadata = {
            "job_settings_name": "FiberPhotometry",
            "subject_id": "mouse_001",
            "rig_id": "Rig_001",
            "mouse_platform_name": "Platform_A",
            "active_mouse_platform": True,
            "ethics_review_id": "IACUC-12345",
            "notes": "Test experiment",
            "session_type": "FIB",
            "anaesthesia": "isoflurane",
            "session_start_time": 933019.553312,
            "session_end_time": 933754.601152,
            "data_files": [str(self.test_data_dir / "green.csv")],
            "rig_config": {"rig_name": "Rig_001"},
            "session_config": {
                "session_type": "FIB",
                "experimenter": ["John Doe", "Jane Smith"],
                "subject": "mouse_001",
            },
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
        self.assertEqual(result["experimenter_full_name"], ["John Doe", "Jane Smith"])

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
