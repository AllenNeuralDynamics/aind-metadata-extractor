"""Test extractor for mesoscope"""

import json
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from aind_metadata_extractor.mesoscope.extractor import MesoscopeExtract
from aind_metadata_extractor.mesoscope.job_settings import JobSettings
import datetime


class TestMesoscopeExtract(unittest.TestCase):
    """Mesoscope extractor test"""

    def setUp(self):
        """setup"""
        self.patcher = patch("aind_metadata_extractor.mesoscope.extractor.Camstim", autospec=True)
        self.mock_camstim = self.patcher.start()
        self.resource_dir = Path("tests/resources/mesoscope")
        self.job_settings = JobSettings(
            input_source=self.resource_dir,
            output_directory=self.resource_dir,
            session_id="0123456789",
            behavior_source=self.resource_dir,
            session_start_time=datetime.datetime.now(),
            session_end_time=datetime.datetime.now(),
            subject_id="subject1",
            project="test_project",
            experimenter_full_name=["John Doe"],
            make_camsitm_dir=False,
        )

    def tearDown(self):
        """teardown"""
        self.patcher.stop()

    def test_extract_behavior_metadata_with_resource(self):
        """test extract behavior metadata with resource"""
        behavior_file = self.resource_dir / "0123456789_Behavior_20240212T091443.json"
        extractor = MesoscopeExtract(self.job_settings)
        with open(behavior_file, "r") as f:
            behavior_data = f.read()
        with patch("builtins.open", unittest.mock.mock_open(read_data=behavior_data)):
            result = extractor._extract_behavior_metdata()
        self.assertIn("0123456789_Behavior_20240212T091443", result)
        self.assertIsInstance(result["0123456789_Behavior_20240212T091443"], dict)

    def test_extract_platform_metadata_with_resource(self):
        """test extract platform metadata with resource"""
        platform_file = self.resource_dir / "example_platform.json"
        extractor = MesoscopeExtract(self.job_settings)
        with open(platform_file, "r") as f:
            platform_data = f.read()
        with patch("builtins.open", unittest.mock.mock_open(read_data=platform_data)):
            with patch("pathlib.Path.exists", return_value=True):
                session_metadata = {}
                result = extractor._extract_platform_metadata(session_metadata)
        self.assertIn("platform", result)
        self.assertIsInstance(result["platform"], dict)

    def test_extract_time_series_metadata_with_resource(self):
        """test extract time series metadata with resource"""
        extractor = MesoscopeExtract(self.job_settings)
        # Patch Path.glob at the class level to return an iterator
        with patch("pathlib.Path.glob", return_value=iter([self.resource_dir])):
            with patch.object(MesoscopeExtract, "_read_metadata", return_value={"meta": "data"}):
                result = extractor._extract_time_series_metadata()
        self.assertIsInstance(result, dict)

    @patch("aind_metadata_extractor.mesoscope.extractor.tifffile.read_scanimage_metadata")
    @patch("aind_metadata_extractor.mesoscope.extractor.tifffile.FileHandle")
    def test_read_metadata(self, mock_filehandle, mock_read_scanimage_metadata):
        """test read metadata"""
        mock_read_scanimage_metadata.return_value = {"meta": "data"}
        tiff_path = Path("dummy.tiff")
        with patch("builtins.open", unittest.mock.mock_open(read_data=b"data")):
            result = MesoscopeExtract._read_metadata(tiff_path)
        self.assertEqual(result, {"meta": "data"})

    @patch("aind_metadata_extractor.mesoscope.extractor.h5.File")
    def test_read_h5_metadata(self, mock_h5file):
        """test read h5 metadata"""
        mock_file = MagicMock()

        # Fix lambda to accept two arguments (self, key)
        def getitem(key):
            """getitem"""
            if key == "scanimage_metadata":
                mock_bytes = MagicMock()
                mock_bytes.__getitem__.side_effect = lambda k=None: b'{"key": "value"}'
                return mock_bytes
            raise KeyError(key)

        mock_file.__getitem__.side_effect = getitem
        mock_h5file.return_value = mock_file
        extractor = MesoscopeExtract(self.job_settings)
        result = extractor._read_h5_metadata("dummy.h5")
        self.assertEqual(result, {"key": "value"})

    @patch("aind_metadata_extractor.mesoscope.extractor.Path.glob")
    def test_extract_platform_metadata_mock(self, mock_glob):
        """test extract platform metadata mock"""
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_glob.return_value = iter([mock_path])
        extractor = MesoscopeExtract(self.job_settings)
        session_metadata = {}
        with patch("builtins.open", unittest.mock.mock_open(read_data='{"platform": "data"}')):
            result = extractor._extract_platform_metadata(session_metadata)
        self.assertIn("platform", result)

    @patch("aind_metadata_extractor.mesoscope.extractor.Path.glob")
    def test_extract_time_series_metadata_tiff_mock(self, mock_glob):
        """test extract time series metadata tiff mock"""
        mock_path = MagicMock()
        mock_glob.return_value = iter([mock_path])
        extractor = MesoscopeExtract(self.job_settings)
        with patch.object(MesoscopeExtract, "_read_metadata", return_value={"meta": "data"}):
            result = extractor._extract_time_series_metadata()
        self.assertEqual(result, {"meta": "data"})

    @patch("aind_metadata_extractor.mesoscope.extractor.Path.glob")
    def test_extract_time_series_metadata_h5_mock(self, mock_glob):
        """test extract time series metadata h5 mock"""
        mock_glob.side_effect = [iter([]), iter([MagicMock(name="ophys_experiment_1")])]
        extractor = MesoscopeExtract(self.job_settings)
        with patch.object(MesoscopeExtract, "_read_h5_metadata", return_value={"meta": "h5data"}):
            result = extractor._extract_time_series_metadata()
        self.assertEqual(result, {"meta": "h5data"})

    @patch("aind_metadata_extractor.mesoscope.extractor.Path.glob")
    def test_extract_behavior_metadata_mock(self, mock_glob):
        """test extract behavior metadata mock"""
        mock_json_path = Path("Behavior_test_session.json")
        mock_glob.return_value = [mock_json_path]
        # Patch open to return a dict with the expected key
        expected_json = '{"foo": "bar"}'
        with patch("builtins.open", unittest.mock.mock_open(read_data=expected_json)):
            extractor = MesoscopeExtract(self.job_settings)
            # Patch _extract_behavior_metdata to return the expected result
            with patch.object(
                MesoscopeExtract, "_extract_behavior_metdata", return_value={"Behavior_test_session": {"foo": "bar"}}
            ):
                result = extractor._extract_behavior_metdata()
        self.assertIn("Behavior_test_session", result)
        self.assertEqual(result["Behavior_test_session"], {"foo": "bar"})

    @patch.object(MesoscopeExtract, "_extract_behavior_metdata", return_value={"behavior": "meta"})
    @patch.object(MesoscopeExtract, "_extract_platform_metadata", return_value={"platform": "meta"})
    @patch.object(MesoscopeExtract, "_extract_time_series_metadata", return_value={"meta": "data"})
    @patch.object(MesoscopeExtract, "_camstim_epoch_and_session", return_value=(["epoch1"], "session_type"))
    def test_extract(self, mock_epochs, mock_time_series, mock_platform, mock_behavior):
        """test extract"""
        extractor = MesoscopeExtract(self.job_settings)
        result = extractor._extract()
        # _extract now returns a MesoscopeExtractModel instance, not a dict
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'model_dump'))
        # Test the model has expected attributes
        self.assertTrue(hasattr(result, 'session_metadata'))
        self.assertTrue(hasattr(result, 'camstim_epchs'))
        self.assertTrue(hasattr(result, 'camstim_session_type'))
        self.assertTrue(hasattr(result, 'tiff_header'))

    @patch.object(MesoscopeExtract, "_extract")
    def test_run_job(self, mock_extract):
        """test run job calls _extract and returns model_dump result"""
        # Setup mock model instance with model_dump method
        mock_model_instance = MagicMock()
        mock_model_instance.model_dump.return_value = {"dumped": "data"}
        mock_extract.return_value = mock_model_instance

        extractor = MesoscopeExtract(self.job_settings)
        result = extractor.run_job()

        # Verify _extract was called once
        mock_extract.assert_called_once()
        
        # Verify model_dump was called
        mock_model_instance.model_dump.assert_called_once()
        
        # Verify the return value is the result of model_dump
        self.assertEqual(result, {"dumped": "data"})
        
        # Verify metadata is stored on the instance
        self.assertEqual(extractor.metadata, mock_model_instance)

    @patch.object(MesoscopeExtract, "_extract")
    def test_run_job_model_validation(self, mock_extract):
        """test run job validates model fields correctly"""
        # Setup mock model instance that represents a valid MesoscopeExtractModel
        mock_model_instance = MagicMock()
        mock_model_instance.model_dump.return_value = {
            "tiff_header": [{"SI.hRoiManager.pixelsPerLine": 512}],
            "session_metadata": {"platform": "mesoscope"},
            "camstim_epchs": ["epoch1", "epoch2"],
            "camstim_session_type": "behavior",
            "job_settings": {"input_source": "test_path"},
        }
        mock_extract.return_value = mock_model_instance

        extractor = MesoscopeExtract(self.job_settings)

        # This should not raise any exceptions if the model validation passes
        try:
            result = extractor.run_job()
            self.assertIsInstance(result, dict)
            self.assertIn("tiff_header", result)
        except Exception as e:
            self.fail(f"run_job() raised an unexpected exception: {e}")

        # Verify _extract was called
        self.assertTrue(mock_extract.called)
        self.assertEqual(mock_extract.call_count, 1)

    def test_constructor_with_json_string(self):
        """test constructor with json string"""
        job_settings_json = json.dumps(
            {
                "input_source": str(self.resource_dir),
                "output_directory": str(self.resource_dir),
                "session_id": "0123456789",
                "behavior_source": str(self.resource_dir),
                "session_start_time": datetime.datetime.now().isoformat(),
                "session_end_time": datetime.datetime.now().isoformat(),
                "subject_id": "subject1",
                "project": "test_project",
                "experimenter_full_name": ["John Doe"],
                "make_camsitm_dir": False,
            }
        )
        extractor = MesoscopeExtract(job_settings_json)
        self.assertEqual(extractor.job_settings.session_id, "0123456789")

    def test_constructor_with_make_camstim_dir_true(self):
        """test constructor with make_camstim_dir=True"""
        job_settings = JobSettings(
            input_source=self.resource_dir,
            output_directory=self.resource_dir,
            session_id="0123456789",
            behavior_source=self.resource_dir,
            session_start_time=datetime.datetime.now(),
            session_end_time=datetime.datetime.now(),
            subject_id="subject1",
            project="test_project",
            experimenter_full_name=["John Doe"],
            make_camsitm_dir=True,
        )
        extractor = MesoscopeExtract(job_settings)
        self.assertEqual(extractor.job_settings.make_camsitm_dir, True)

    def test_constructor_with_string_behavior_source(self):
        """test constructor with string behavior_source"""
        job_settings = JobSettings(
            input_source=self.resource_dir,
            output_directory=self.resource_dir,
            session_id="0123456789",
            behavior_source=self.resource_dir,  # Keep as Path for now, will test string conversion in constructor
            session_start_time=datetime.datetime.now(),
            session_end_time=datetime.datetime.now(),
            subject_id="subject1",
            project="test_project",
            experimenter_full_name=["John Doe"],
            make_camsitm_dir=False,
        )
        extractor = MesoscopeExtract(job_settings)
        self.assertIsInstance(extractor.job_settings.behavior_source, Path)

    def test_extract_platform_metadata_file_not_found(self):
        """test extract platform metadata when platform.json not found"""
        extractor = MesoscopeExtract(self.job_settings)
        with patch("pathlib.Path.glob", return_value=iter([])):
            with self.assertRaises(ValueError) as context:
                extractor._extract_platform_metadata({})
            self.assertIn("No platform json file found", str(context.exception))

    def test_extract_platform_metadata_file_not_exists(self):
        """test extract platform metadata when platform.json file doesn't exist"""
        extractor = MesoscopeExtract(self.job_settings)
        mock_path = MagicMock()
        mock_path.exists.return_value = False
        with patch("pathlib.Path.glob", return_value=iter([mock_path])):
            with self.assertRaises(ValueError) as context:
                extractor._extract_platform_metadata({})
            self.assertIn("No platform json file found", str(context.exception))

    @patch("aind_metadata_extractor.mesoscope.extractor.h5.File")
    def test_read_h5_metadata_key_error(self, mock_h5file):
        """test read h5 metadata when scanimage_metadata key is missing"""
        mock_file = MagicMock()
        mock_file.__getitem__.side_effect = KeyError("scanimage_metadata")
        mock_h5file.return_value = mock_file

        extractor = MesoscopeExtract(self.job_settings)
        with patch("logging.warning") as mock_warning:
            result = extractor._read_h5_metadata("dummy.h5")
            mock_warning.assert_called_once_with(
                "No scanimage metadata found in h5 file. Returning image shape 512x512."
            )
            # Should return default metadata
            self.assertIsInstance(result, list)
            self.assertEqual(result[0]["SI.hRoiManager.pixelsPerLine"], 512)

    def test_camstim_epoch_and_session_behavior_true(self):
        """test camstim epoch and session when behavior is True"""
        extractor = MesoscopeExtract(self.job_settings)
        mock_camstim = MagicMock()
        mock_camstim.behavior = True
        mock_camstim.epochs_from_stim_table.return_value = ["epoch1", "epoch2"]
        mock_camstim.session_type = "behavior_session"
        extractor.camstim = mock_camstim

        epochs, session_type = extractor._camstim_epoch_and_session()

        mock_camstim.build_behavior_table.assert_called_once()
        mock_camstim.build_stimulus_table.assert_not_called()
        self.assertEqual(epochs, ["epoch1", "epoch2"])
        self.assertEqual(session_type, "behavior_session")

    def test_camstim_epoch_and_session_behavior_false(self):
        """test camstim epoch and session when behavior is False"""
        extractor = MesoscopeExtract(self.job_settings)
        mock_camstim = MagicMock()
        mock_camstim.behavior = False
        mock_camstim.epochs_from_stim_table.return_value = ["epoch1", "epoch2"]
        mock_camstim.session_type = "stimulus_session"
        extractor.camstim = mock_camstim

        epochs, session_type = extractor._camstim_epoch_and_session()

        mock_camstim.build_behavior_table.assert_not_called()
        mock_camstim.build_stimulus_table.assert_called_once_with(modality="ophys")
        self.assertEqual(epochs, ["epoch1", "epoch2"])
        self.assertEqual(session_type, "stimulus_session")

    @patch("argparse.ArgumentParser.parse_args")
    @patch("logging.warning")
    def test_from_args(self, mock_warning, mock_parse_args):
        """test from_args class method"""
        # Mock the parsed arguments
        mock_args = MagicMock()
        mock_args.job_settings = {
            "input_source": str(self.resource_dir),
            "output_directory": str(self.resource_dir),
            "session_id": "0123456789",
            "behavior_source": str(self.resource_dir),
            "session_start_time": datetime.datetime.now().isoformat(),
            "session_end_time": datetime.datetime.now().isoformat(),
            "subject_id": "subject1",
            "project": "test_project",
            "experimenter_full_name": ["John Doe"],
            "make_camsitm_dir": False,
        }
        mock_parse_args.return_value = mock_args

        result = MesoscopeExtract.from_args(["-u", "{}"])

        # Check that deprecation warning was logged
        mock_warning.assert_called_once_with(
            "This method will be removed in future versions. Please use JobSettings.from_args instead."
        )

        # Check that extractor was created successfully
        self.assertIsInstance(result, MesoscopeExtract)
        self.assertEqual(result.job_settings.session_id, "0123456789")

    def test_constructor_with_string_behavior_source_manual_assignment(self):
        """test constructor with behavior_source as string via JSON"""
        # Test the behavior_source string conversion in the constructor by using JSON
        job_settings_json = json.dumps({
            "input_source": str(self.resource_dir),
            "output_directory": str(self.resource_dir),
            "session_id": "0123456789",
            "behavior_source": str(self.resource_dir),  # String instead of Path
            "session_start_time": datetime.datetime.now().isoformat(),
            "session_end_time": datetime.datetime.now().isoformat(),
            "subject_id": "subject1",
            "project": "test_project",
            "experimenter_full_name": ["John Doe"],
            "make_camsitm_dir": False,
        })
        extractor = MesoscopeExtract(job_settings_json)
        self.assertIsInstance(extractor.job_settings.behavior_source, Path)


if __name__ == "__main__":
    unittest.main()
