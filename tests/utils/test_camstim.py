"""Test the camstim.py module"""

import unittest
from datetime import datetime as dt
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from aind_metadata_extractor.utils.camstim_sync.camstim import Camstim, CamstimSettings


class TestCamstim(unittest.TestCase):
    """Test camstim.py"""

    @classmethod
    @patch("pathlib.Path.rglob")
    @patch("aind_metadata_extractor.utils.camstim_sync.camstim.Camstim._get_sync_times")
    @patch("aind_metadata_extractor.utils.camstim_sync.camstim.Camstim.get_session_uuid")
    @patch("aind_metadata_extractor.utils.camstim_sync.camstim.Camstim._is_behavior")
    @patch("aind_metadata_extractor.utils.camstim_sync.pkl_utils.load_pkl")
    @patch("aind_metadata_extractor.utils.camstim_sync.sync_utils.load_sync")
    @patch("aind_metadata_extractor.utils.camstim_sync.pkl_utils.get_fps")
    @patch("aind_metadata_extractor.utils.camstim_sync.pkl_utils.get_stage")
    def setUpClass(
        cls,
        mock_get_stage: MagicMock,
        mock_get_fps: MagicMock,
        mock_load_sync: MagicMock,
        mock_load_pkl: MagicMock,
        mock_is_behavior: MagicMock,
        mock_session_uuid: MagicMock,
        mock_sync_times: MagicMock,
        mock_rglob: MagicMock,
    ) -> None:
        """Set up the test suite"""
        mock_get_fps.return_value = 30.0
        mock_load_sync.return_value = {}
        mock_load_pkl.return_value = {
            "fps": 30.0,
            "items": {
                "behavior": {
                    "params": {
                        "stage": "stage",
                    }
                }
            },
        }
        mock_is_behavior.return_value = True
        mock_session_uuid.return_value = "1234"
        mock_sync_times.return_value = (
            dt(2024, 11, 1, 15, 41, 32, 920082),
            dt(2024, 11, 1, 15, 41, 50, 648629),
        )
        mock_get_stage.return_value = "stage"
        mock_rglob.return_value = iter([Path("some/path/file.pkl")])
        cls.camstim = Camstim(
            CamstimSettings(
                input_source="some/path",
                output_directory="some/other/path",
                session_id="1234567890",
                subject_id="123456",
            )
        )
        cls.camstim_settings = CamstimSettings(
            input_source="some/path",
            output_directory="some/other/path",
            session_id="1234567890",
            subject_id="123456",
        )

    @patch("aind_metadata_extractor.utils.camstim_sync.sync_utils.get_ophys_stimulus_timestamps")  # noqa
    @patch("aind_metadata_extractor.utils.camstim_sync.behavior_utils.from_stimulus_file")  # noqa
    @patch("pandas.DataFrame.to_csv")
    def test_build_behavior_table(
        self,
        mock_to_csv: MagicMock,
        mock_from_stimulus_file: MagicMock,
        mock_get_ophys_stimulus_timestamps: MagicMock,
    ):
        """Test the build_behavior_table method"""
        # Mock the return values
        mock_get_ophys_stimulus_timestamps.return_value = [1, 2, 3]
        mock_from_stimulus_file.return_value = [pd.DataFrame({"a": [1, 2, 3]})]

        # Call the method
        self.camstim.build_behavior_table()

        # Assert the calls
        mock_get_ophys_stimulus_timestamps.assert_called_once_with(self.camstim.sync_data, self.camstim.pkl_path)
        mock_from_stimulus_file.assert_called_once_with(self.camstim.pkl_path, [1, 2, 3])
        mock_to_csv.assert_called_once_with(self.camstim.stim_table_path, index=False)

    @patch("aind_metadata_extractor.utils.camstim_sync.stim_utils.extract_frame_times_from_vsync")  # noqa
    @patch("aind_metadata_extractor.utils.camstim_sync.stim_utils.extract_frame_times_from_photodiode")  # noqa
    @patch("aind_metadata_extractor.utils.camstim_sync.stim_utils.create_stim_table")
    @patch("aind_metadata_extractor.utils.camstim_sync.naming_utils.map_column_names")
    @patch("pandas.DataFrame.to_csv")
    @patch("aind_metadata_extractor.utils.camstim_sync.stim_utils.seconds_to_frames")
    @patch("aind_metadata_extractor.utils.camstim_sync.pkl_utils.get_stimuli")
    @patch("aind_metadata_extractor.utils.camstim_sync.stim_utils.extract_blocks_from_stim")  # noqa
    @patch("aind_metadata_extractor.utils.camstim_sync.camstim.Camstim.get_stim_table_seconds")
    def test_build_stimulus_table(
        self,
        mock_get_stim_table_seconds: MagicMock,
        mock_extract_blocks_from_stim: MagicMock,
        mock_get_stimuli: MagicMock,
        mock_seconds_to_frames: MagicMock,
        mock_to_csv: MagicMock,
        mock_map_column_names: MagicMock,
        mock_create_stim_table: MagicMock,
        mock_extract_frame_times_from_photodiode: MagicMock,
        mock_extract_frame_times_from_vsync: MagicMock,
    ):
        """Test the build_stimulus_table method"""
        # Mock the return values
        mock_get_stim_table_seconds.return_value = [pd.DataFrame({"a": [1, 2, 3]})]
        mock_extract_blocks_from_stim.return_value = [1, 2, 3]
        mock_get_stimuli.return_value = {"stuff": "things"}
        mock_seconds_to_frames.return_value = np.array([1, 2, 3])
        mock_extract_frame_times_from_photodiode.return_value = [0.1, 0.2, 0.3]
        mock_extract_frame_times_from_vsync.return_value = [0.1, 0.2, 0.3]
        mock_create_stim_table.return_value = pd.DataFrame({"a": [1, 2, 3]})
        mock_map_column_names.return_value = pd.DataFrame({"a": [1, 2, 3]})

        # Override behavior for this test - build_stimulus_table requires non-behavior data
        self.camstim.behavior = False

        # Call the method
        self.camstim.build_stimulus_table()

        # Assert the calls
        mock_extract_frame_times_from_vsync.assert_called_once()
        mock_extract_frame_times_from_photodiode.assert_called_once()
        mock_create_stim_table.assert_called_once()
        mock_map_column_names.assert_called_once()
        mock_to_csv.assert_called_once_with(self.camstim.stim_table_path, index=False)

    def test_extract_stim_epochs(self):
        """Test the extract_stim_epochs method"""
        # Create a mock stimulus table
        data = {
            "start_time": [0, 1, 2, 3, 4],
            "stop_time": [1, 2, 3, 4, 5],
            "stim_name": ["stim1", "stim1", "stim2", "stim2", "stim3"],
            "stim_type": ["type1", "type1", "type2", "type2", "type3"],
            "frame": [0, 1, 2, 3, 4],
            "param1": ["a", "a", "b", "b", "c"],
            "param2": [1, 1, 2, 2, 3],
        }
        stim_table = pd.DataFrame(data)

        # Expected output - should include all epochs
        expected_epochs = [
            ["stim1", 0, 2, {"param1": {"a"}, "param2": {1}}, set()],
            ["stim2", 2, 4, {"param1": {"b"}, "param2": {2}}, set()],
            ["stim3", 4, 5, {"param1": {"c"}, "param2": {3}}, set()],
        ]

        # Call the method
        epochs = self.camstim.extract_stim_epochs(stim_table)

        # Assert the result
        self.assertEqual(epochs, expected_epochs)

    def test_extract_stim_epochs_with_images_and_movies(self):
        """Test the extract_stim_epochs method with images and movies"""
        # Create a mock stimulus table with images and movies
        data = {
            "start_time": [0, 1, 2, 3, 4],
            "stop_time": [1, 2, 3, 4, 5],
            "stim_name": ["image1", "image1", "movie1", "movie1", "stim3"],
            "stim_type": ["type1", "type1", "type2", "type2", "type3"],
            "frame": [0, 1, 2, 3, 4],
            "param1": ["a", "a", "b", "b", "c"],
            "param2": [1, 1, 2, 2, 3],
        }
        stim_table = pd.DataFrame(data)

        # Expected output - all epochs should be returned, but only image/movie should have template names
        expected_epochs = [
            ["image1", 0, 2, {"param1": {"a"}, "param2": {1}}, {"image1"}],
            ["movie1", 2, 4, {"param1": {"b"}, "param2": {2}}, {"movie1"}],
            ["stim3", 4, 5, {"param1": {"c"}, "param2": {3}}, set()],
        ]
        # Call the method
        epochs = self.camstim.extract_stim_epochs(stim_table)
        # Assert the result
        self.assertEqual(epochs, expected_epochs)

    @patch("aind_metadata_extractor.utils.camstim_sync.camstim.Camstim.extract_stim_epochs")
    @patch("pandas.read_csv")
    def test_epochs_from_stim_table(self, mock_read_csv: MagicMock, mock_extract_stim_epochs: MagicMock):
        """Test the epochs_from_stim_table method"""
        # Mock the return values
        mock_read_csv.return_value = pd.DataFrame(
            {
                "start_time": [0, 1, 2],
                "stop_time": [1, 2, 3],
                "stim_name": ["stim1", "stim2", "stim3"],
                "stim_type": ["type1", "type2", "type3"],
                "frame": [0, 1, 2],
                "param1": ["a", "b", "c"],
                "param2": [1, 2, 3],
            }
        )
        mock_extract_stim_epochs.return_value = [
            ["stim1", 0, 1, {"param1": {"a"}, "param2": {1}}, set()],
            ["stim2", 1, 2, {"param1": {"b"}, "param2": {2}}, set()],
            ["stim3", 2, 3, {"param1": {"c"}, "param2": {3}}, set()],
        ]

        # Call the method
        schema_epochs = self.camstim.epochs_from_stim_table()

        # Assert the result
        self.assertEqual(len(schema_epochs), 3)
        self.assertEqual(schema_epochs[0]["stimulus_name"], "stim1")
        self.assertEqual(schema_epochs[1]["stimulus_name"], "stim2")
        self.assertEqual(schema_epochs[2]["stimulus_name"], "stim3")

    @patch("aind_metadata_extractor.utils.camstim_sync.stim_utils.convert_frames_to_seconds")  # noqa
    @patch("aind_metadata_extractor.utils.camstim_sync.naming_utils.collapse_columns")
    @patch("aind_metadata_extractor.utils.camstim_sync.naming_utils.drop_empty_columns")
    @patch("aind_metadata_extractor.utils.camstim_sync.naming_utils.standardize_movie_numbers")
    @patch("aind_metadata_extractor.utils.camstim_sync.naming_utils.add_number_to_shuffled_movie")  # noqa
    @patch("aind_metadata_extractor.utils.camstim_sync.naming_utils.map_stimulus_names")
    def test_get_stim_table_seconds(
        self,
        mock_map_stimulus_names: MagicMock,
        mock_add_number_to_shuffled_movie: MagicMock,
        mock_standardize_movie_numbers: MagicMock,
        mock_drop_empty_columns: MagicMock,
        mock_collapse_columns: MagicMock,
        mock_convert_frames_to_seconds: MagicMock,
    ):
        """Test the get_stim_table_seconds method"""
        # Mock the return values
        mock_convert_frames_to_seconds.return_value = pd.DataFrame({"a": [1, 2, 3]})
        mock_collapse_columns.return_value = pd.DataFrame({"a": [1, 2, 3]})
        mock_drop_empty_columns.return_value = pd.DataFrame({"a": [1, 2, 3]})
        mock_standardize_movie_numbers.return_value = pd.DataFrame({"a": [1, 2, 3]})
        mock_add_number_to_shuffled_movie.return_value = pd.DataFrame({"a": [1, 2, 3]})
        mock_map_stimulus_names.return_value = pd.DataFrame({"a": [1, 2, 3]})

        # Call the method
        stim_table_sweeps = pd.DataFrame({"frame": [1, 2, 3]})
        frame_times = [0.1, 0.2, 0.3]
        name_map = {"old_name": "new_name"}

        result = self.camstim.get_stim_table_seconds(stim_table_sweeps, frame_times, name_map)
        # Assert the calls
        mock_convert_frames_to_seconds.assert_called_once_with(stim_table_sweeps, frame_times, 30.0, True)
        mock_collapse_columns.assert_called_once()
        mock_drop_empty_columns.assert_called_once()
        mock_standardize_movie_numbers.assert_called_once()
        mock_add_number_to_shuffled_movie.assert_called_once()
        mock_map_stimulus_names.assert_called_once()

        # Assert the result
        expected_result = pd.DataFrame({"a": [1, 2, 3]})
        pd.testing.assert_frame_equal(result, expected_result)

    def test_extract_whole_session_epoch(self):
        """
        Test that extract_whole_session_epoch returns the correct start and
        stop times for a mock stim_table DataFrame.
        """
        # Create a mock stim_table DataFrame
        stim_table = pd.DataFrame(
            {
                "start_time": [0.0, 10.0, 20.0],
                "stop_time": [5.0, 15.0, 25.0],
            }
        )
        camstim = self.camstim
        # Call the method
        result = camstim.extract_whole_session_epoch(stim_table)
        # Check that the result is as expected (tuple of (start, stop))
        self.assertEqual(result[0], stim_table.start_time.iloc[0])
        self.assertEqual(result[1], stim_table.stop_time.iloc[-1])

    def test_get_session_uuid(self):
        """Test the get_session_uuid method"""
        # Test that it accesses the session_uuid from pkl_data
        expected_uuid = "test-uuid-123"

        with patch("aind_metadata_extractor.utils.camstim_sync.pkl_utils.load_pkl") as mock_load_pkl:
            mock_load_pkl.return_value = {"session_uuid": expected_uuid}
            result = self.camstim.get_session_uuid()
            self.assertEqual(result, expected_uuid)
            mock_load_pkl.assert_called_once_with(self.camstim.pkl_path)

    def test_get_session_type_foraging(self):
        """Test _get_session_type when behavior is False (foraging data)"""
        # Set up foraging data structure
        original_behavior = self.camstim.behavior
        self.camstim.behavior = False
        self.camstim.pkl_data = {"items": {"foraging": {"params": {"stage": "foraging_stage"}}}}

        result = self.camstim._get_session_type()
        self.assertEqual(result, "foraging_stage")

        # Restore original behavior
        self.camstim.behavior = original_behavior

    def test_is_behavior_false(self):
        """Test _is_behavior when no behavior data exists"""
        # Create test data without behavior section
        test_pkl_data = {"items": {"foraging": {"params": {"stage": "test"}}}}

        # Temporarily replace pkl_data
        original_pkl_data = self.camstim.pkl_data
        self.camstim.pkl_data = test_pkl_data

        result = self.camstim._is_behavior()
        self.assertFalse(result)

        # Restore original data
        self.camstim.pkl_data = original_pkl_data

    @patch("aind_metadata_extractor.utils.camstim_sync.pkl_utils.get_stage")
    @patch("aind_metadata_extractor.utils.camstim_sync.pkl_utils.get_fps")
    @patch("aind_metadata_extractor.utils.camstim_sync.pkl_utils.load_pkl")
    @patch("aind_metadata_extractor.utils.camstim_sync.sync_utils.get_stop_time")
    @patch("aind_metadata_extractor.utils.camstim_sync.sync_utils.get_start_time")
    @patch("aind_metadata_extractor.utils.camstim_sync.sync_utils.load_sync")
    @patch("pathlib.Path.glob")
    @patch("pathlib.Path.rglob")
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.is_dir")
    def test_init_directory_creation(
        self,
        mock_is_dir,
        mock_mkdir,
        mock_rglob,
        mock_glob,
        mock_load_sync,
        mock_get_start_time,
        mock_get_stop_time,
        mock_load_pkl,
        mock_get_fps,
        mock_get_stage,
    ):
        """Test that __init__ creates output directory if it doesn't exist"""
        from datetime import datetime as dt

        # Mock setup
        mock_is_dir.return_value = False  # Directory doesn't exist
        mock_rglob.return_value = iter([Path("test.pkl")])
        mock_glob.return_value = iter([Path("test.h5")])
        mock_load_pkl.return_value = {"items": {"behavior": {"params": {"stage": "test"}}}, "session_uuid": "test"}
        mock_get_fps.return_value = 30.0
        mock_get_stage.return_value = "test"
        mock_load_sync.return_value = {}
        mock_get_start_time.return_value = dt.now()
        mock_get_stop_time.return_value = dt.now()

        # Create Camstim instance
        settings = CamstimSettings(
            input_source=Path("test_input"),
            output_directory=Path("test_output"),
            session_id="test123",
            subject_id="test_subject",
        )

        # Create Camstim instance which should trigger directory creation
        Camstim(settings)
        mock_mkdir.assert_called_once_with(parents=True)

    def test_extract_stim_epochs_spontaneous_skip(self):
        """Test that extract_stim_epochs skips spontaneous stimuli"""
        # Create test data with spontaneous stimulus
        data = {
            "start_time": [0, 1, 2],
            "stop_time": [1, 2, 3],
            "stim_name": ["stim1", "spontaneous", "stim2"],
            "param1": ["a", "b", "c"],
        }
        stim_table = pd.DataFrame(data)

        epochs = self.camstim.extract_stim_epochs(stim_table)

        # Should only have stim1 and stim2, spontaneous should be skipped
        self.assertEqual(len(epochs), 2)
        self.assertEqual(epochs[0][0], "stim1")
        self.assertEqual(epochs[1][0], "stim2")

    def test_summarize_epoch_params_large_set(self):
        """Test _summarize_epoch_params with large parameter sets (>1000 values)"""
        # Create test data with a column that has >1000 unique values
        large_data = list(range(1001))  # 1001 unique values
        data = {
            "start_time": large_data,
            "stop_time": large_data,
            "stim_name": ["test"] * 1001,
            "large_param": large_data,  # This will trigger the >1000 condition
        }
        stim_table = pd.DataFrame(data)

        current_epoch = ["test", 0, 1000, {}, set()]

        # Call the method
        self.camstim._summarize_epoch_params(stim_table, current_epoch, 0, 1001)

        # Check that the large parameter set was handled correctly
        self.assertEqual(current_epoch[3]["large_param"], ["Error: over 1000 values"])

    def test_summarize_epoch_params_empty_set(self):
        """Test _summarize_epoch_params with empty parameter sets"""
        # Create test data where dropna() results in empty sets
        data = {
            "start_time": [0, 1],
            "stop_time": [1, 2],
            "stim_name": ["test", "test"],
            "empty_param": [None, None],  # All NaN values
            "valid_param": ["a", "b"],  # Valid values
        }
        stim_table = pd.DataFrame(data)

        current_epoch = ["test", 0, 2, {}, set()]

        # Call the method
        self.camstim._summarize_epoch_params(stim_table, current_epoch, 0, 2)

        # Empty param should not be added, valid param should be added
        self.assertNotIn("empty_param", current_epoch[3])
        self.assertIn("valid_param", current_epoch[3])
        self.assertEqual(current_epoch[3]["valid_param"], {"a", "b"})

    @patch("aind_metadata_extractor.utils.camstim_sync.stim_utils.extract_frame_times_with_delay")
    @patch("aind_metadata_extractor.utils.camstim_sync.stim_utils.extract_frame_times_from_vsync")
    @patch("aind_metadata_extractor.utils.camstim_sync.stim_utils.extract_frame_times_from_photodiode")
    @patch("aind_metadata_extractor.utils.camstim_sync.stim_utils.create_stim_table")
    @patch("aind_metadata_extractor.utils.camstim_sync.naming_utils.map_column_names")
    @patch("pandas.DataFrame.to_csv")
    @patch("aind_metadata_extractor.utils.camstim_sync.stim_utils.seconds_to_frames")
    @patch("aind_metadata_extractor.utils.camstim_sync.pkl_utils.get_stimuli")
    @patch("aind_metadata_extractor.utils.camstim_sync.stim_utils.extract_blocks_from_stim")
    @patch("aind_metadata_extractor.utils.camstim_sync.camstim.Camstim.get_stim_table_seconds")
    def test_build_stimulus_table_ophys_modality(
        self,
        mock_get_stim_table_seconds: MagicMock,
        mock_extract_blocks_from_stim: MagicMock,
        mock_get_stimuli: MagicMock,
        mock_seconds_to_frames: MagicMock,
        mock_to_csv: MagicMock,
        mock_map_column_names: MagicMock,
        mock_create_stim_table: MagicMock,
        mock_extract_frame_times_from_photodiode: MagicMock,
        mock_extract_frame_times_from_vsync: MagicMock,
        mock_extract_frame_times_with_delay: MagicMock,
    ):
        """Test build_stimulus_table with ophys modality to cover lines 207-213 and 248"""
        # Mock the return values
        mock_get_stim_table_seconds.return_value = [
            pd.DataFrame({"a": [1, 2, 3]}),
            pd.DataFrame({"b": [4, 5, 6]}),  # Two tables to trigger vsync_table save
        ]
        mock_extract_blocks_from_stim.return_value = [1, 2, 3]
        mock_get_stimuli.return_value = {"stuff": "things"}
        mock_seconds_to_frames.return_value = np.array([1, 2, 3])
        mock_extract_frame_times_from_photodiode.return_value = [0.1, 0.2, 0.3]
        mock_extract_frame_times_from_vsync.return_value = np.array([0.15, 0.25, 0.35])
        mock_extract_frame_times_with_delay.return_value = 0.05  # Delay value
        mock_create_stim_table.return_value = pd.DataFrame({"a": [1, 2, 3]})
        mock_map_column_names.return_value = pd.DataFrame({"a": [1, 2, 3]})

        # Override behavior for this test
        self.camstim.behavior = False

        # Call the method with ophys modality
        self.camstim.build_stimulus_table(modality="ophys")

        # Assert the ophys-specific calls were made
        mock_extract_frame_times_with_delay.assert_called_once_with(self.camstim.sync_data)
        # extract_frame_times_from_vsync is called twice: once for ophys and once for vsync_times
        self.assertEqual(mock_extract_frame_times_from_vsync.call_count, 2)

        # Assert that to_csv was called twice (for both stim_table and vsync_table)
        self.assertEqual(mock_to_csv.call_count, 2)

    def test_extract_stim_epochs_with_image_set(self):
        """Test extract_stim_epochs with non-empty image_set to cover line 323"""
        # Create test data with image_set column
        data = {
            "start_time": [0, 1],
            "stop_time": [1, 2],
            "stim_name": ["test1", "test2"],
            "image_set": ["images_set", ""],  # One with image_set, one without
            "param1": ["a", "b"],
        }
        stim_table = pd.DataFrame(data)

        epochs = self.camstim.extract_stim_epochs(stim_table)

        # Should have 2 epochs
        self.assertEqual(len(epochs), 2)
        # First epoch should have "images_set" in template since it contains "image"
        self.assertEqual(epochs[0][4], {"test1"})
        # Second epoch should have empty template set
        self.assertEqual(epochs[1][4], set())


if __name__ == "__main__":
    unittest.main()
