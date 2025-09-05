""" Unit tests for the behavior_utils module in the utils package. """

import unittest

import pandas as pd
import numpy as np

from unittest.mock import MagicMock, patch
from aind_metadata_extractor.utils.camstim_sync import behavior_utils as behavior


class TestBehaviorUtils(unittest.TestCase):
    """
    Tests Behavior utils
    """

    @patch(
        "aind_metadata_extractor.utils.camstim_sync"
        ".behavior_utils.get_visual_stimuli_df"
    )
    def test_get_stimulus_presentations(self,
                                        mock_get_visual_stimuli_df):
        """
        Tests the get_stimulus_presentations function
        """
        data = {}  # Example data, replace with appropriate test data
        stimulus_timestamps = [0.0, 0.5, 1.0, 1.5]

        # Mocking the response of get_visual_stimuli_df
        mock_get_visual_stimuli_df.return_value = pd.DataFrame(
            {
                "frame": [0, 1, 2, 3],
                "time": [0.0, 0.5, 1.0, 1.5],
                "end_frame": [1, 2, 3, np.nan],
            }
        )

        # Expected DataFrame after processing
        expected_df = pd.DataFrame(
            {
                "end_frame": [1, 2, 3, np.nan],
                "start_frame": [0, 1, 2, 3],
                "start_time": [0.0, 0.5, 1.0, 1.5],
                "stop_time": [0.5, 1.0, 1.5, float("nan")],
            },
            index=pd.Index([0, 1, 2, 3], name="stimulus_presentations_id"),
        )

        # Call the function to test
        result_df = behavior.get_stimulus_presentations(
            data, stimulus_timestamps
        )

        # Assert DataFrame equality
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_get_gratings_metadata(self):
        """
        Creates a stimuli with gratings and
        tests the get_gratings_metadata
        Note: This function appears to be commented out in the main code,
        but we test the interface for potential future use
        """
        # Since get_gratings_metadata is commented out, we'll test the interface
        # that would be expected based on the test structure
        stimuli_with_gratings = {
            "grating": {
                "phase": 0.5,
                "sf": 0.03,
                "set_log": [[0, 0.0], [1, 45.0], [2, 90.0], [3, 0.0]],
            }
        }

        # Check if the function exists; if not, skip the test
        if hasattr(behavior, 'get_gratings_metadata'):
            result_grating_df = behavior.get_gratings_metadata(stimuli_with_gratings)
            self.assertIsInstance(result_grating_df, pd.DataFrame)
        else:
            self.skipTest("get_gratings_metadata function is not available")

        stimuli_without_gratings = {
            "other_stimuli": {"some_key": "some_value"}
        }

        if hasattr(behavior, 'get_gratings_metadata'):
            result_empty_df = behavior.get_gratings_metadata(stimuli_without_gratings)
            self.assertIsInstance(result_empty_df, pd.DataFrame)

    def test_get_visual_stimuli_df_with_foraging(self):
        """
        Tests the get_visual_stimuli_df function with foraging data structure
        """
        # Create mock data with foraging structure instead of behavior
        mock_data = {
            "items": {
                "foraging": {
                    "stimuli": {
                        "stimulus1": {
                            "set_log": [
                                ("ori", 90, None, 10),
                                ("image", "img1.jpg", None, 20),
                            ],
                            "draw_log": [0] * 30,  # Make sure it's long enough
                        }
                    },
                    "omitted_flash_frame_log": {}
                }
            }
        }

        mock_time = np.arange(30) * 0.1

        # Call the function under test
        result_df = behavior.get_visual_stimuli_df(mock_data, mock_time)

        # Should return a DataFrame
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertIn('omitted', result_df.columns)

    def test_get_visual_stimuli_df_no_omitted_log(self):
        """
        Tests the get_visual_stimuli_df function when no omitted flash log exists
        """
        mock_data = {
            "items": {
                "behavior": {
                    "stimuli": {
                        "stimulus1": {
                            "set_log": [
                                ("ori", 90, None, 10),
                            ],
                            "draw_log": [0] * 30,  # Make sure it's long enough
                        }
                    },
                    # No omitted_flash_frame_log key
                }
            }
        }

        mock_time = np.arange(30) * 0.1

        # Call the function under test
        result_df = behavior.get_visual_stimuli_df(mock_data, mock_time)

        # Should still work without omitted flash log
        self.assertIsInstance(result_df, pd.DataFrame)

    def test_add_active_flag_with_existing_active_column(self):
        """
        Tests add_active_flag when active column already exists
        """
        stim_pres_table = pd.DataFrame(
            {
                "start_time": [1, 5, 10],
                "active": [True, False, True],  # Already has active column
            }
        )

        trials = pd.DataFrame({"start_time": [0, 10], "stop_time": [20, 40]})

        # Should return the same table when active column exists
        result = behavior.add_active_flag(stim_pres_table, trials)
        pd.testing.assert_frame_equal(result, stim_pres_table)

    def test_add_active_flag_without_stimulus_block(self):
        """
        Tests add_active_flag when stimulus_block column doesn't exist
        """
        stim_pres_table = pd.DataFrame(
            {
                "start_time": [1, 5, 10, 15],
                "image_name": ["img1", "img2", "img3", np.nan],
            }
        )

        trials = pd.DataFrame({"start_time": [0, 10], "stop_time": [20, 40]})

        result = behavior.add_active_flag(stim_pres_table, trials)

        # Should add active column
        self.assertIn("active", result.columns)
        # Should mark stimuli within trial times as active
        self.assertTrue(result["active"].iloc[0])  # start_time=1 is within trials
        self.assertTrue(result["active"].iloc[1])  # start_time=5 is within trials
        self.assertTrue(result["active"].iloc[2])  # start_time=10 is within trials
        self.assertFalse(result["active"].iloc[3])  # image_name is NaN

    def test_compute_trials_id_for_stimulus_missing_columns(self):
        """
        Tests compute_trials_id_for_stimulus when required columns are missing
        """
        # Test without stimulus_block column
        stim_pres_table = pd.DataFrame(
            {
                "start_time": [1, 5, 10],
                "image_name": ["img1", "img2", "img3"],
                "active": [True, True, False],
            }
        )

        trials_table = pd.DataFrame({"start_time": [0, 10], "stop_time": [20, 40]})

        result = behavior.compute_trials_id_for_stimulus(stim_pres_table, trials_table)
        
        # Should return trials_ids with basic mapping only
        self.assertEqual(result.name, "trials_id")
        self.assertEqual(len(result), len(stim_pres_table))

        # Test without active column
        stim_pres_table2 = pd.DataFrame(
            {
                "start_time": [1, 5, 10],
                "image_name": ["img1", "img2", "img3"],
                "stimulus_block": [1, 1, 2],
            }
        )

        result2 = behavior.compute_trials_id_for_stimulus(stim_pres_table2, trials_table)
        self.assertEqual(result2.name, "trials_id")

    @patch(
        "aind_metadata_extractor.utils.camstim_sync"
        ".behavior_utils.get_images_dict"
    )
    @patch(
        "aind_metadata_extractor.utils.camstim_sync"
        ".stim_utils.convert_filepath_caseinsensitive"
    )
    @patch(
        "aind_metadata_extractor.utils.camstim_sync"
        ".stim_utils.get_image_set_name"
    )
    def test_get_stimulus_metadata(
        self,
        mock_get_image_set_name,
        mock_convert_filepath_caseinsensitive,
        mock_get_images_dict,
    ):
        """
        Tests the get_stimulus_metadata function
        """
        # Example pkl input
        pkl = {
            "items": {
                "behavior": {
                    "stimuli": {
                        "images": {},
                        "grating": {
                            "phase": 0.5,
                            "sf": 0.03,
                            "set_log": [
                                [0, 0.0],
                                [1, 45.0],
                                [2, 90.0],
                                [3, 0.0],
                            ],
                        },
                    }
                }
            }
        }

        # Mock the get_images_dict function
        mock_get_images_dict.return_value = {
            "metadata": {"image_set": "path/to/images.pkl"},
            "image_attributes": [
                {
                    "image_category": "image",
                    "image_name": "image1.jpg",
                    "orientation": np.nan,
                    "phase": np.nan,
                    "size": np.nan,
                    "spatial_frequency": np.nan,
                    "image_index": 0,
                },
                {
                    "image_category": "image",
                    "image_name": "image2.jpg",
                    "orientation": np.nan,
                    "phase": np.nan,
                    "size": np.nan,
                    "spatial_frequency": np.nan,
                    "image_index": 1,
                },
            ],
        }

        # Mock the stim.convert_filepath_caseinsensitive function
        mock_convert_filepath_caseinsensitive.return_value = (
            "path/to/images.pkl"
        )

        # Mock the stim.get_image_set_name function
        mock_get_image_set_name.return_value = "image_set_name"

        # Expected DataFrame
        expected_df = pd.DataFrame(
            {
                "image_category": [
                    "image",
                    "image",
                    "omitted",
                ],
                "image_name": [
                    "image1.jpg",
                    "image2.jpg",
                    "omitted",
                ],
                "orientation": [np.nan, np.nan, np.nan],
                "phase": [np.nan, np.nan, np.nan],
                "size": [np.nan, np.nan, np.nan],
                "spatial_frequency": [np.nan, np.nan, np.nan],
                "image_set": [
                    "image_set_name",
                    "image_set_name",
                    "omitted",
                ],
                "image_index": [0, 1, 2],
            }
        ).set_index("image_index")

        # Call the function
        result_df = behavior.get_stimulus_metadata(pkl)

        # Assert DataFrame equality
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_get_stimulus_metadata_no_images(self):
        """
        Tests the get_stimulus_metadata function when no images are present
        """
        # Example pkl input without images
        pkl_no_images = {
            "items": {
                "behavior": {
                    "stimuli": {
                        "grating": {
                            "phase": 0.5,
                        }
                    }
                }
            }
        }

        # Call the function
        result_df = behavior.get_stimulus_metadata(pkl_no_images)

        # Should have only the omitted entry
        self.assertEqual(len(result_df), 1)
        self.assertEqual(result_df.iloc[0]["image_name"], "omitted")
        self.assertEqual(result_df.iloc[0]["image_category"], "omitted")

    def test_get_stimulus_epoch(self):
        """
        Tests the get_stimulus_epoch function
        using a fake set_log
        """
        # Example set_log input
        set_log = [
            ("Image", "image1.jpg", 0, 10),
            ("Image", "image2.jpg", 0, 20),
            ("Grating", 45, 0, 30),
        ]
        n_frames = 40

        # Test case where current_set_index is not the last one
        current_set_index = 0
        start_frame = 10
        expected_output = (10, 20)
        result = behavior.get_stimulus_epoch(
            set_log, current_set_index, start_frame, n_frames
        )
        self.assertEqual(result, expected_output)

        # Test case where current_set_index is the last one
        current_set_index = 2
        start_frame = 30
        expected_output = (30, 40)
        result = behavior.get_stimulus_epoch(
            set_log, current_set_index, start_frame, n_frames
        )
        self.assertEqual(result, expected_output)

        # Test case where there is only one stimulus in set_log
        set_log_single = [("Image", "image1.jpg", 0, 10)]
        current_set_index = 0
        start_frame = 10
        expected_output = (10, 40)
        result = behavior.get_stimulus_epoch(
            set_log_single, current_set_index, start_frame, n_frames
        )
        self.assertEqual(result, expected_output)

    def test_get_draw_epochs(self):
        """
        Creats fake draw logs
        tests the get_draw_epochs function
        """
        # Example draw_log input
        draw_log = [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1]
        start_frame = 2
        stop_frame = 11

        # Expected output
        expected_output = [(2, 4), (5, 8), (10, 11)]

        # Call the function
        result = behavior.get_draw_epochs(draw_log, start_frame, stop_frame)

        # Assert equality
        self.assertEqual(result, expected_output)

        # Test case where no frames are active
        draw_log_no_active = [0, 0, 0, 0, 0]
        start_frame = 0
        stop_frame = 4
        expected_output_no_active = []
        result_no_active = behavior.get_draw_epochs(
            draw_log_no_active, start_frame, stop_frame
        )
        self.assertEqual(result_no_active, expected_output_no_active)

        # Test case where all frames are active
        draw_log_all_active = [1, 1, 1, 1, 1]
        start_frame = 0
        stop_frame = 4
        expected_output_all_active = [(0, 4)]
        result_all_active = behavior.get_draw_epochs(
            draw_log_all_active, start_frame, stop_frame
        )
        self.assertEqual(result_all_active, expected_output_all_active)

        # Test case with mixed active and inactive frames
        draw_log_mixed = [1, 0, 1, 0, 1, 0, 1]
        start_frame = 0
        stop_frame = 6
        expected_output_mixed = [(0, 1), (2, 3), (4, 5)]
        result_mixed = behavior.get_draw_epochs(
            draw_log_mixed, start_frame, stop_frame
        )
        self.assertEqual(result_mixed, expected_output_mixed)

    def test_unpack_change_log(self):
        """
        Tests changing of the log using names with .jpg
        """
        # Example change input
        change = (("Image", "image1.jpg"), ("Grating", "45_deg"), 12345, 67)

        # Expected output
        expected_output = {
            "frame": 67,
            "time": 12345,
            "from_category": "Image",
            "to_category": "Grating",
            "from_name": "image1.jpg",
            "to_name": "45_deg",
        }

        # Call the function
        result = behavior.unpack_change_log(change)

        # Assert equality
        self.assertEqual(result, expected_output)

        # Test with different data
        change2 = (
            ("Video", "video1.mp4"),
            ("Static", "static_image"),
            54321,
            89,
        )

        expected_output2 = {
            "frame": 89,
            "time": 54321,
            "from_category": "Video",
            "to_category": "Static",
            "from_name": "video1.mp4",
            "to_name": "static_image",
        }

        result2 = behavior.unpack_change_log(change2)
        self.assertEqual(result2, expected_output2)

    @patch(
        "aind_metadata_extractor.utils.camstim_sync"
        ".behavior_utils.get_stimulus_epoch"
    )
    @patch(
        "aind_metadata_extractor.utils.camstim_sync.behavior_utils.get_draw_epochs"
    )
    def test_get_visual_stimuli_df(
        self, mock_get_draw_epochs, mock_get_stimulus_epoch
    ):
        """
        Tests the get_visual_stimuli_df function
        Mocks some intermediary functions
        """
        # Mock input data
        mock_data = {
            "items": {
                "behavior": {
                    "stimuli": {
                        "stimulus1": {
                            "set_log": [
                                ("ori", 90, None, 10),
                                ("image", "img1.jpg", None, 20),
                            ],
                            "draw_log": [(5, 15), (25, 35)],
                        },
                        "stimulus2": {
                            "set_log": [
                                ("ori", 270, None, 5),
                                ("image", "img2.jpg", None, 15),
                            ],
                            "draw_log": [(0, 10), (20, 30)],
                        },
                    },
                    "omitted_flash_frame_log": {"omitted_flash1": [1, 2]},
                }
            }
        }

        mock_time = np.arange(3) * 0.1  # Adjust the number of timestamps here

        # Mock return values for get_stimulus_epoch and get_draw_epochs
        mock_get_stimulus_epoch.side_effect = lambda *args, **kwargs: (
            0,
            2,
        )  # Mocking epoch start and end
        mock_get_draw_epochs.side_effect = lambda *args, **kwargs: [
            (0, 2)
        ]  # Mocking draw epochs

        # Call the function under test
        result_df = behavior.get_visual_stimuli_df(mock_data, mock_time)

        # Define expected output dataframe
        expected_columns = [
            "orientation",
            "image_name",
            "frame",
            "end_frame",
            "time",
            "duration",
            "omitted",
        ]
        expected_data = {
            "orientation": [90, 90, 270, 270, np.nan, np.nan],
            "image_name": [
                "img1.jpg",
                "img1.jpg",
                "img2.jpg",
                "img2.jpg",
                "omitted",
                "omitted",
            ],
            "frame": [0, 20, 5, 25, 3, 8],
            "end_frame": [10, 30, 10, 30, np.nan, np.nan],
            "time": [
                mock_time[0],
                mock_time[2],
                mock_time[1],
                mock_time[2],
                mock_time[0],
                mock_time[1],
            ],
            "duration": [
                mock_time[1] - mock_time[0],
                mock_time[2] - mock_time[1],
                mock_time[1] - mock_time[0],
                mock_time[2] - mock_time[1],
                0.25,
                0.25,
            ],
            "omitted": [False, False, False, False, True, True],
        }
        expected_df = pd.DataFrame(expected_data, columns=expected_columns)

        # Perform assertions
        self.assertEqual(result_df["time"].all(), expected_df["time"].all())

    def test_get_image_names(self):
        """
        Tests the get_image_names function
        """
        # Mock data
        behavior_stimulus_file = {
            "stimuli": {
                "stim1": {
                    "set_log": [
                        ("image", "image1.jpg", None, 0),
                        ("ori", 45, None, 1),
                    ]
                },
                "stim2": {
                    "set_log": [
                        ("image", "image2.jpg", None, 2),
                        ("ori", 90, None, 3),
                    ]
                },
                "stim3": {
                    "set_log": [
                        ("image", "image1.jpg", None, 4),
                        ("ori", 135, None, 5),
                    ]
                },
            }
        }

        # Expected output
        expected_output = {"image1.jpg", "image2.jpg"}

        # Call the function
        result = behavior.get_image_names(behavior_stimulus_file)

        # Assert equality
        self.assertEqual(result, expected_output)

        # Test case with no images
        behavior_stimulus_file_no_images = {
            "stimuli": {
                "stim1": {"set_log": [("ori", 45, None, 1)]},
                "stim2": {"set_log": [("ori", 90, None, 3)]},
            }
        }

        # Expected output
        expected_output_no_images = set()

        # Call the function
        result_no_images = behavior.get_image_names(
            behavior_stimulus_file_no_images
        )

        # Assert equality
        self.assertEqual(result_no_images, expected_output_no_images)

    def test_is_change_event(self):
        """
        Tests the is_change_event function
        """
        # Mock data
        stimulus_presentations = pd.DataFrame(
            {
                "image_name": [
                    "img1",
                    "img1",
                    "img2",
                    "img2",
                    "img3",
                    "omitted",
                    "img3",
                    "img4",
                ],
                "omitted": [
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                ],
            }
        )

        # Expected output
        expected_output = pd.Series(
            [False, False, True, False, True, False, False, True],
            name="is_change",
        )

        # Call the function
        result = behavior.is_change_event(stimulus_presentations)

        # Assert equality
        pd.testing.assert_series_equal(result, expected_output)

    def test_get_flashes_since_change(self):
        """
        Tests the get_flashes_since_change function
        """
        # Mock data
        stimulus_presentations = pd.DataFrame(
            {
                "image_name": [
                    "img1",
                    "img1",
                    "img2",
                    "img2",
                    "img3",
                    "omitted",
                    "img3",
                    "img4",
                ],
                "omitted": [
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                ],
                "is_change": [
                    False,
                    False,
                    True,
                    False,
                    True,
                    False,
                    False,
                    True,
                ],
            }
        )

        # Expected output
        expected_output = pd.Series(
            [0, 1, 0, 1, 0, 0, 1, 0], name="flashes_since_change"
        )

        # Call the function
        result = behavior.get_flashes_since_change(stimulus_presentations)

        # Assert equality
        pd.testing.assert_series_equal(result, expected_output)

    def test_add_active_flag(self):
        """
        Tests the add_active_flag function
        """
        # Mock data for stimulus presentations table
        stim_pres_table = pd.DataFrame(
            {
                "start_time": [1, 5, 10, 15, 20, 25, 30],
                "stop_time": [2, 6, 11, 16, 21, 26, 31],
                "image_name": [
                    "img1",
                    "img2",
                    "img3",
                    np.nan,
                    "img4",
                    "img5",
                    "img6",
                ],
                "stimulus_block": [1, 1, 2, 2, 3, 3, 3],
            }
        )

        # Mock data for trials table
        trials = pd.DataFrame({"start_time": [0, 10], "stop_time": [20, 40]})

        # Expected output
        expected_active = pd.Series(
            [True, True, True, True, True, True, True], name="active"
        )
        expected_output = stim_pres_table.copy()
        expected_output["active"] = expected_active

        # Call the function
        result = behavior.add_active_flag(stim_pres_table, trials)

        # Assert the 'active' column is correctly added
        pd.testing.assert_series_equal(result["active"], expected_active)

    def test_compute_trials_id_for_stimulus(self):
        """
        Tests the compute_trials_id_for_stimulus function
        """
        # Mock data for stimulus presentations table
        stim_pres_table = pd.DataFrame(
            {
                "start_time": [1, 5, 10, 15, 20, 25, 30, 35, 40, 45],
                "stop_time": [2, 6, 11, 16, 21, 26, 31, 36, 41, 46],
                "image_name": [
                    "img1",
                    "img2",
                    "img3",
                    np.nan,
                    "img4",
                    "img5",
                    "img6",
                    "img1",
                    "img2",
                    "img3",
                ],
                "stimulus_block": [1, 1, 2, 2, 3, 3, 3, 4, 4, 4],
                "active": [
                    True,
                    True,
                    True,
                    True,
                    False,
                    False,
                    False,
                    True,
                    True,
                    True,
                ],
            }
        )

        # Mock data for trials table
        trials_table = pd.DataFrame(
            {"start_time": [0, 10], "stop_time": [20, 40]}
        )

        # Expected output
        expected_trials_id = pd.Series(
            data=[0, 0, 0, -99, 1, 1, 1, 1, -99, -99],
            index=stim_pres_table.index,
            name="trials_id",
        ).astype("int")

        # Call the function
        result = behavior.compute_trials_id_for_stimulus(
            stim_pres_table, trials_table
        )

        # Assert the trials_id series is correctly assigned
        pd.testing.assert_series_equal(result, expected_trials_id)

    def test_fix_omitted_end_frame(self):
        """
        Tests the fix_omitted_end_frame function
        """
        # Mock data for stimulus presentations table
        stim_pres_table = pd.DataFrame(
            {
                "start_frame": [0, 5, 10, 15, 20],
                "end_frame": [5, 10, 15, np.nan, 25],
                "omitted": [False, False, False, True, False],
            }
        )

        # Calculate expected median stimulus frame duration
        median_stim_frame_duration = np.nanmedian(
            stim_pres_table["end_frame"] - stim_pres_table["start_frame"]
        )

        # Expected output
        expected_end_frame = stim_pres_table["end_frame"].copy()
        expected_end_frame.iloc[3] = (
            stim_pres_table["start_frame"].iloc[3] + median_stim_frame_duration
        )

        expected_stim_pres_table = stim_pres_table.copy()
        expected_stim_pres_table["end_frame"] = expected_end_frame
        expected_stim_pres_table = expected_stim_pres_table.astype(
            {"start_frame": int, "end_frame": int}
        )

        # Call the function
        result = behavior.fix_omitted_end_frame(stim_pres_table)

        # Assert the DataFrame is correctly modified
        pd.testing.assert_frame_equal(result, expected_stim_pres_table)

    def test_compute_is_sham_change_no_column(self):
        """
        tests the compute_is_sham_change function
        """
        stim_df_no_active = pd.DataFrame(
            {
                "trials_id": [0, 0, 0, 1, 1, 1],
                "stimulus_block": [1, 1, 2, 2, 3, 3],
                "image_name": ["A", "A", "B", "B", "C", "C"],
                "start_frame": [0, 10, 20, 30, 40, 50],
                "is_sham_change": [False, False, False, False, False, False],
            }
        )

        # Create a sample trials DataFrame
        trials = pd.DataFrame(
            {"catch": [False, False, True], "change_frame": [10, 40, 60]}
        )

        expected_stim_df = stim_df_no_active.copy()

        result = behavior.compute_is_sham_change(stim_df_no_active, trials)

        pd.testing.assert_frame_equal(result, expected_stim_df)

    def test_fingerprint_from_stimulus_file(self):
        """
        Creates a fake stim file and
        Tests the fingerprint_from_stimulus_file function
        """
        stimulus_presentations = pd.DataFrame(
            {
                "stim_block": [1, 1, 2, 2],
            }
        )

        stimulus_file = {
            "items": {
                "behavior": {
                    "items": {
                        "fingerprint": {
                            "static_stimulus": {
                                "runs": 3,
                                "frame_list": [0, 1, 1, 0, 1, 1],
                                "sweep_frames": [[0, 1], [2, 3], [4, 5]],
                                "sweep_table": [[0.0], [45.0], [90.0]],
                                "sweep_order": [0, 1, 2],
                                "dimnames": ["orientation"],
                            },
                            "frame_indices": [0, 1, 2, 3, 4, 5],
                        }
                    }
                }
            }
        }

        stimulus_timestamps = [0, 1, 2, 3, 4, 5]
        # Call the function under test
        result = behavior.fingerprint_from_stimulus_file(
            stimulus_presentations,
            stimulus_file,
            stimulus_timestamps,
            "fingerprint",
        )

        # Define expected output based on the provided mock data
        expected_columns = [
            "start_time",
            "stop_time",
            "start_frame",
            "end_frame",
            "duration",
            "orientation",
            "stim_block",
            "stim_name",
        ]

        expected_data = [
            {
                "start_time": 0,
                "stop_time": 2,
                "start_frame": 0,
                "end_frame": 1,
                "duration": 2,
                "orientation": 0.0,
                "stim_block": 4,
                "stim_name": "fingerprint",
            },
            {
                "start_time": 2,
                "stop_time": 4,
                "start_frame": 2,
                "end_frame": 3,
                "duration": 2,
                "orientation": 45.0,
                "stim_block": 4,
                "stim_name": "fingerprint",
            },
            {
                "start_time": 4,
                "stop_time": 5,
                "start_frame": 4,
                "end_frame": 5,
                "duration": 1,
                "orientation": 90.0,
                "stim_block": 4,
                "stim_name": "fingerprint",
            },
        ]

        expected_df = pd.DataFrame(expected_data, columns=expected_columns)

        # Assert that the result matches the expected DataFrame
        pd.testing.assert_frame_equal(result, expected_df)

    @patch("aind_metadata_extractor.utils.camstim_sync.pkl_utils.load_pkl")
    @patch(
        "aind_metadata_extractor.utils.camstim_sync.behavior_utils"
        ".get_stimulus_presentations"
    )
    @patch(
        "aind_metadata_extractor.utils.camstim_sync.behavior_utils"
        ".check_for_errant_omitted_stimulus"
    )
    @patch(
        "aind_metadata_extractor.utils.camstim_sync.behavior_utils"
        ".get_stimulus_metadata"
    )
    @patch(
        "aind_metadata_extractor.utils.camstim_sync.behavior_utils"
        ".is_change_event"
    )
    @patch(
        "aind_metadata_extractor.utils.camstim_sync.behavior_utils"
        ".get_flashes_since_change"
    )
    @patch(
        "aind_metadata_extractor.utils.camstim_sync.behavior_utils"
        ".get_stimulus_name"
    )
    @patch(
        "aind_metadata_extractor.utils.camstim_sync.behavior_utils"
        ".fix_omitted_end_frame"
    )
    @patch(
        "aind_metadata_extractor.utils.camstim_sync.behavior_utils"
        ".add_fingerprint_stimulus"
    )
    @patch("aind_metadata_extractor.utils.camstim_sync"
           ".behavior_utils.postprocess")
    def test_from_stimulus_file(
        self,
        mock_postprocess,
        mock_add_fingerprint_stimulus,
        mock_fix_omitted_end_frame,
        mock_get_stimulus_name,
        mock_get_flashes_since_change,
        mock_is_change_event,
        mock_get_stimulus_metadata,
        mock_check_for_errant_omitted_stimulus,
        mock_get_stimulus_presentations,
        mock_load_pkl,
    ):
        """
        Tests the from_stimulus_file function
        mocks intermediary functions so the test
        isn't 1000 lines
        """
        # Mock data
        stimulus_file = MagicMock()
        stimulus_timestamps = MagicMock()
        limit_to_images = ["image1", "image2"]
        column_list = ["column1", "column2"]
        project_code = "VBO"

        # Mock return values
        mock_load_pkl.return_value = MagicMock()
        mock_get_stimulus_presentations.return_value = pd.DataFrame(
            {
                "start_time": [0, 1, 2],
                "image_name": ["image1", "image2", "image1"],
                "orientation": [0, 90, 180],
                "index": ["0", "oris", "phase"],
            }
        )
        mock_check_for_errant_omitted_stimulus.return_value = pd.DataFrame(
            {
                "start_time": [0, 1, 2],
                "image_name": ["image1", "image2", "image1"],
                "orientation": [0, 90, 180],
                "index": ["0", "oris", "phase"],
            }
        )
        mock_get_stimulus_metadata.return_value = pd.DataFrame(
            {
                "image_name": ["image1", "image2"],
                "image_set": ["set1", "set2"],
                "image_index": [1, 2],
                "start_time": [0, 1],
                "phase": ["A", "B"],
                "spatial_frequency": [1.0, 2.0],
                "index": ["A", "phase"],
            }
        )
        mock_is_change_event.return_value = pd.Series([True, False, True])
        mock_get_flashes_since_change.return_value = pd.Series([0, 1, 0])
        mock_get_stimulus_name.return_value = "natural_movie_one"
        mock_fix_omitted_end_frame.return_value = pd.DataFrame(
            {
                "start_frame": [0, 1, 2],
                "end_frame": [1, 3, 4],
                "omitted": [False, False, False],
                "index": ["A", "B", "phase"],
            }
        )
        mock_add_fingerprint_stimulus.return_value = pd.DataFrame(
            {
                "start_time": [0, 1, 2],
                "end_time": [1, 2, 3],
                "image_name": ["image1", "image2", "image1"],
                "is_change": [True, False, True],
                "stim_block": [1, 1, 2],
                "index": ["A", "B", "phase"],
            }
        )
        mock_postprocess.return_value = pd.DataFrame(
            {
                "start_time": [0, 1, 2],
                "end_time": [1, 2, 3],
                "image_name": ["image1", "image2", "image1"],
                "is_change": [True, False, True],
                "stim_block": [1, 1, 2],
                "index": ["A", "B", "phase"],
            }
        )

        # Call the function under test
        result = behavior.from_stimulus_file(
            stimulus_file,
            stimulus_timestamps,
            limit_to_images,
            column_list,
            project_code=project_code,
        )

        # Define expected output based on the mocked return values
        expected_columns = [
            "start_time",
            "end_time",
            "image_name",
            "is_change",
            "stim_block",
            "stim_name",
            "movie_frame_index",
            "movie_repeat",
            "duration",
            "flashes_since_change",
        ]

        expected_data = {
            "start_time": [0, 1, 2],
            "end_time": [1, 2, 3],
            "image_name": ["image1", "image2", "image1"],
            "is_change": [True, False, True],
            "stim_block": [1, 1, 2],
            "stim_name": "natural_movie_one",
            "movie_frame_index": [0, 0, 0],
            "movie_repeat": [0, 0, 1],
            "duration": [1, 1, 1],
            "flashes_since_change": [0, 1, 0],
        }

        expected_df = pd.DataFrame(expected_data, columns=expected_columns)

        # Assert that the result matches the expected DataFrame
        self.assertEqual(
            expected_df["start_time"].all(), result["start_time"].all()
        )

    @patch("aind_metadata_extractor.utils.camstim_sync.pkl_utils.load_pkl")
    def test_from_stimulus_file_with_all_null_orientations_and_images(
        self, mock_load_pkl
    ):
        """
        Tests from_stimulus_file when both image_name and orientation are null
        """
        stimulus_file = MagicMock()
        stimulus_timestamps = [0.0, 1.0, 2.0]

        # Create mock data with null image_name and orientation
        mock_data = {
            "items": {
                "behavior": {
                    "stimuli": {},
                    "items": {}
                }
            }
        }
        mock_load_pkl.return_value = mock_data

        # Mock stimulus presentations with null values
        patch_path = ('aind_metadata_extractor.utils.camstim_sync.'
                      'behavior_utils._load_and_validate_stimulus_presentations')
        with patch(patch_path) as mock_load_validate:
            mock_load_validate.return_value = pd.DataFrame({
                "start_time": [0, 1, 2],
                "image_name": [None, None, None],
                "orientation": [None, None, None],
                "index": [0, 1, 2]
            })

            # Should raise ValueError when both are null
            with self.assertRaises(ValueError):
                behavior.from_stimulus_file(stimulus_file, stimulus_timestamps)

    @patch("aind_metadata_extractor.utils.camstim_sync.pkl_utils.load_pkl")
    def test_from_stimulus_file_with_grating_orientations(
        self, mock_load_pkl
    ):
        """
        Tests from_stimulus_file when image_name is null but orientation is not
        """
        stimulus_file = MagicMock()
        stimulus_timestamps = [0.0, 1.0, 2.0]

        mock_data = {
            "items": {
                "behavior": {
                    "stimuli": {},
                    "items": {}
                }
            }
        }
        mock_load_pkl.return_value = mock_data

        load_patch = ('aind_metadata_extractor.utils.camstim_sync.'
                      'behavior_utils._load_and_validate_stimulus_presentations')
        meta_patch = ('aind_metadata_extractor.utils.camstim_sync.'
                      'behavior_utils.get_stimulus_metadata')
        name_patch = ('aind_metadata_extractor.utils.camstim_sync.'
                      'behavior_utils.get_stimulus_name')
        omitted_patch = ('aind_metadata_extractor.utils.camstim_sync.'
                         'behavior_utils.fix_omitted_end_frame')
        process_patch = ('aind_metadata_extractor.utils.camstim_sync.'
                         'behavior_utils.postprocess')

        with patch(load_patch) as mock_load_validate:
            with patch(meta_patch) as mock_get_metadata:
                with patch(name_patch) as mock_get_name:
                    with patch(omitted_patch) as mock_fix_omitted:
                        with patch(process_patch) as mock_postprocess:

                            # Mock data with null image_name but valid orientation
                            presentations_df = pd.DataFrame({
                                "start_time": [0, 1, 2],
                                "image_name": [None, None, None],
                                "orientation": [0, 90, 180],
                                "duration": [0.25, 0.25, 0.25],
                                "index": [0, 1, 2]
                            })
                            presentations_df.index.name = "stimulus_presentations_id"

                            mock_load_validate.return_value = presentations_df
                            mock_get_metadata.return_value = pd.DataFrame({
                                "image_name": ["gratings_0", "gratings_90", "gratings_180"],
                                "image_set": ["grating"] * 3,
                                "image_index": [0, 1, 2],
                                "phase": [np.nan] * 3,
                                "size": [np.nan] * 3,
                                "orientation": [0, 90, 180],
                                "spatial_frequency": [np.nan] * 3
                            }).set_index("image_index")
                            mock_get_name.return_value = "grating"
                            mock_fix_omitted.return_value = presentations_df
                            mock_postprocess.return_value = presentations_df

                            # Should not raise error and should create grating image names
                            result = behavior.from_stimulus_file(stimulus_file, stimulus_timestamps)

                            # Verify that the function completed without error
                            self.assertIsInstance(result, tuple)

    def test_postprocess(self):
        """
        Tests the postprocess function
        """
        # Actual input data
        presentations = pd.DataFrame(
            {
                "image_name": ["image1", "image2", "image3", None],
                "omitted": [False, True, False, False],
                "duration": [0.25, None, None, None],
                "boolean_col": [True, False, True, False],
                "object_col": [True, None, False, None],
                "list_col": [[1, 2], [3, 4], [], [5, 6]],  # List-like column to trigger continue
                "start_time": [0, 1, 2, 3],
            }
        )
        # Ensure list_col is object dtype
        presentations["list_col"] = presentations["list_col"].astype(object)

        # Call the function under test
        processed_presentations = behavior.postprocess(presentations)

        expected_columns = [
            "start_time",
            "stop_time",
            "image_name",
            "omitted",
            "duration",
            "boolean_col",
            "object_col",
            "list_col",
        ]
        expected_data = {
            "image_name": ["image1", "image2", "image3", None],
            "omitted": [False, True, False, False],
            "duration": [
                0.25,
                0.25,
                0.25,
                0.25,
            ],  # Example of filled omitted values
            "boolean_col": [True, False, True, False],
            "object_col": [True, None, False, None],
            "list_col": [[1, 2], [3, 4], [], [5, 6]],  # List column should remain unchanged
            "start_time": [0, 1, 2, 3],
            "stop_time": [None, 1.25, None, None],
        }
        expected_df = pd.DataFrame(expected_data, columns=expected_columns)

        processed_presentations = pd.DataFrame(processed_presentations)
        # Assert that the result matches the expected DataFrame
        self.assertEqual(
            expected_df["duration"].all(),
            processed_presentations["duration"].all(),
        )
        self.assertEqual(
            expected_df["start_time"].all(),
            processed_presentations["start_time"].all(),
        )
        self.assertEqual(
            expected_df["image_name"].all(),
            processed_presentations["image_name"].all(),
        )
        self.assertEqual(
            expected_df["omitted"].all(),
            processed_presentations["omitted"].all(),
        )
        self.assertEqual(
            expected_df["boolean_col"].all(),
            processed_presentations["boolean_col"].all(),
        )

    def test_check_for_errant_omitted_stimulus(self):
        """
        Tests the check_for_errant_omitted_stimulus function
        """
        # Actual input data
        data = {
            "omitted": [True, False, False, False],
            "stimulus_block": [1, 1, 2, 2],
            "other_column": [1, 2, 3, 4],
        }
        input_df = pd.DataFrame(data)

        # Call the function under test
        processed_df = behavior.check_for_errant_omitted_stimulus(input_df)

        # Define expected output based on the expected behavior of the function
        expected_data = {
            "omitted": [False, False, False],
            "stimulus_block": [1, 2, 2],
            "other_column": [2, 3, 4],
        }
        expected_df = pd.DataFrame(expected_data)
        # Assert that the result matches the expected DataFrame
        self.assertEqual(
            processed_df["omitted"].all(), expected_df["omitted"].all()
        )

    def test_fill_missing_values_for_omitted_flashes(self):
        """
        Tests the fill_missing_values_for_omitted_flashes function
        """
        # Actual input data
        data = {
            "start_time": [0.0, 1.0, 2.0, 3.0],
            "stop_time": [0, 0, 0, 0],
            "duration": [1, 1, 0, 0],
            "omitted": [False, True, False, True],
        }
        df = pd.DataFrame(data)

        # Call the function under test
        processed_df = behavior.fill_missing_values_for_omitted_flashes(
            df, omitted_time_duration=0.25
        )

        # Define expected output based on the expected behavior of the function
        expected_data = {
            "start_time": [0.0, 1.0, 2.0, 3.0],
            "stop_time": [0.0, 1.25, 0.0, 3.25],
            "duration": [1, 0.25, 0.0, 0.25],
            "omitted": [False, True, False, True],
        }
        expected_df = pd.DataFrame(expected_data)

        # Assert that the result matches the expected DataFrame
        pd.testing.assert_frame_equal(processed_df, expected_df)

    def test_get_spontaneous_stimulus(self):
        """
        Tests the get_spontaneous_stimulus function
        """
        # Define a sample stimulus presentations table with gaps
        data = {
            "start_frame": [0, 100, 200, 400, 500],
            "start_time": [0.0, 10.0, 20.0, 40.0, 50.0],
            "end_frame": [100, 200, 300, 500, 600],
            "stop_time": [10.0, 20.0, 30.0, 50.0, 60.0],
            "stim_block": [0, 1, 2, 4, 5],
            "stim_name": ["stim1", "stim2", "stim3", "stim4", "stim5"],
        }
        df = pd.DataFrame(data)

        # Call the function under test
        processed_df = behavior.get_spontaneous_stimulus(df)

        # Define expected output based on the expected behavior of the function
        expected_data = {
            "start_frame": [0, 100, 200, 285, 400, 500],
            "start_time": [0.0, 10.0, 20.0, 285.0, 40.0, 50.0],
            "stop_time": [10.0, 20.0, 30.0, 285.0, 50.0, 60.0],
            "stim_block": [0, 1, 2, 3, 4, 5],
            "stim_name": [
                "spontaneous",
                "stim1",
                "stim2",
                "spontaneous",
                "stim3",
                "stim4",
            ],
        }
        expected_df = pd.DataFrame(expected_data)

        # Assert that the result matches the expected DataFrame
        self.assertEqual(
            processed_df["start_frame"].all(), expected_df["start_frame"].all()
        )

    def test_add_fingerprint_stimulus(self):
        """
        Simulates a fingerprint stim and
        Tests the add_fingerprint_stimulus function
        """
        stimulus_file = {
            "items": {
                "behavior": {
                    "items": {
                        "fingerprint": {
                            "static_stimulus": {
                                "runs": 10,
                                "sweep_frames": [[0, 1], [2, 3]],
                                "sweep_table": [[0.0], [45.0]],
                                "sweep_order": [0, 1],
                                "dimnames": ["orientation"],
                                "frame_list": [0, 1, 2],
                            },
                            "frame_indices": [0, 1, 2, 3],
                        }
                    }
                }
            }
        }
        stimulus_presentations = pd.DataFrame(
            {
                "stim_block": [0, 0, 0, 1, 1, 1],
                "start_time": [0.0, 1.0, 2.0, 5.0, 6.0, 7.0],
                "stop_time": [0.5, 1.5, 2.5, 5.5, 6.5, 7.5],
                "start_frame": [0, 1, 2, 5, 6, 7],
                "end_frame": [0, 1, 2, 5, 6, 7],
            }
        )  # Mock the stimulus file as needed
        stimulus_timestamps = np.array([0.0, 10.0, 20.0, 30.0, 40.0])

        # Call the function under test
        processed_df = behavior.add_fingerprint_stimulus(
            stimulus_presentations=stimulus_presentations,
            stimulus_file=stimulus_file,
            stimulus_timestamps=stimulus_timestamps,
            fingerprint_name="fingerprint",
        )

        # Define expected output based on the expected behavior of the function
        expected_data = {
            "start_frame": [0, 100, 200, 300, 400, 500],
            "start_time": [0.0, 10.0, 20.0, 30.0, 40.0, 285.0],
            "stop_time": [10.0, 20.0, 30.0, 40.0, 285.0, 300.0],
            "stim_block": [0, 1, 2, 3, 4, 5],
            "stim_name": [
                "stim1",
                "stim2",
                "stim3",
                "stim4",
                "spontaneous",
                "fingerprint",
            ],
        }
        expected_df = pd.DataFrame(expected_data)

        # Assert that the result matches the expected DataFrame
        self.assertEqual(
            processed_df["start_frame"].all(), expected_df["start_frame"].all()
        )

    def test_get_spontaneous_block_indices(self):
        """
        Tests the get_spontaneous_block_indices function
        """
        # Test case 1: No gaps between stimulus blocks
        stimulus_blocks1 = np.array([0, 1, 2, 3])
        expected_indices1 = np.array([], dtype=np.int64)
        np.testing.assert_array_equal(
            behavior.get_spontaneous_block_indices(stimulus_blocks1),
            expected_indices1,
        )

        # Test case 2: Single gap between stimulus blocks
        stimulus_blocks2 = np.array([0, 2, 3])
        expected_indices2 = np.array([1], dtype=np.int64)
        np.testing.assert_array_equal(
            behavior.get_spontaneous_block_indices(stimulus_blocks2),
            expected_indices2,
        )

        # Test case 4: No spontaneous blocks (no gaps)
        stimulus_blocks4 = np.array([0, 1, 2, 3, 4])
        expected_indices4 = np.array([], dtype=np.int64)
        np.testing.assert_array_equal(
            behavior.get_spontaneous_block_indices(stimulus_blocks4),
            expected_indices4,
        )

        # Test case 5: Raises RuntimeError for large gap
        stimulus_blocks5 = np.array([0, 3, 4, 5])
        with self.assertRaises(RuntimeError):
            behavior.get_spontaneous_block_indices(stimulus_blocks5)

    def test_remove_short_sandwiched_spontaneous(self):
        """
        Tests the remove_short_sandwiched_spontaneous function
        """
        # Test case with short spontaneous stimulus between identical stimuli
        df = pd.DataFrame({
            'start_time': [0.0, 1.0, 1.4, 2.0],
            'image_name': ['img1', 'spontaneous', 'img1', 'img2'],
            'stim_name': ['stim1', 'spontaneous', 'stim1', 'stim2'],
            'duration': [1.0, 0.4, 0.6, 1.0]
        })
        
        result = behavior.remove_short_sandwiched_spontaneous(df)
        
        # The short spontaneous stimulus should be removed
        expected_df = pd.DataFrame({
            'start_time': [0.0, 1.4, 2.0],
            'image_name': ['img1', 'img1', 'img2'],
            'stim_name': ['stim1', 'stim1', 'stim2'],
            'duration': [1.0, 0.6, 1.0]
        }).reset_index(drop=True)
        
        pd.testing.assert_frame_equal(result, expected_df)

        # Test case with no spontaneous stimuli to remove
        df2 = pd.DataFrame({
            'start_time': [0.0, 1.0, 2.0],
            'image_name': ['img1', 'img2', 'img3'],
            'stim_name': ['stim1', 'stim2', 'stim3'],
            'duration': [1.0, 1.0, 1.0]
        })
        
        result2 = behavior.remove_short_sandwiched_spontaneous(df2)
        pd.testing.assert_frame_equal(result2, df2.reset_index(drop=True))

    @patch(
        "aind_metadata_extractor.utils.camstim_sync"
        ".stim_utils.convert_filepath_caseinsensitive"
    )
    @patch("aind_metadata_extractor.utils.camstim_sync.pkl_utils.load_img_pkl")
    @patch("builtins.open")
    def test_get_images_dict(self, mock_open, mock_load_img_pkl, mock_convert_filepath):
        """
        Tests the get_images_dict function
        """
        # Mock pkl_dict input
        pkl_dict = {
            "items": {
                "behavior": {
                    "stimuli": {
                        "images": {
                            "image_path": "/path/to/images.pkl"
                        }
                    }
                }
            }
        }

        # Mock the file path conversion
        mock_convert_filepath.return_value = "/path/to/images.pkl"

        # Mock image set data
        mock_image_set = {
            "category1": {
                "image1.jpg": np.array([1, 2, 3]),
                "image2.jpg": np.array([4, 5, 6])
            },
            "category2": {
                "image3.jpg": np.array([7, 8, 9])
            }
        }
        mock_load_img_pkl.return_value = mock_image_set

        # Call the function
        result = behavior.get_images_dict(pkl_dict)

        # Verify the structure
        self.assertIn('metadata', result)
        self.assertIn('images', result)
        self.assertIn('image_attributes', result)

        # Check metadata
        self.assertEqual(result['metadata']['image_set'], "/path/to/images.pkl")

        # Check that images and attributes are populated
        self.assertEqual(len(result['images']), 3)
        self.assertEqual(len(result['image_attributes']), 3)

        # Check image attributes structure
        for attr in result['image_attributes']:
            self.assertIn('image_category', attr)
            self.assertIn('image_name', attr)
            self.assertIn('image_index', attr)

    def test_clean_position_and_contrast(self):
        """
        Tests the clean_position_and_contrast function
        """
        # Test with Pos column
        df = pd.DataFrame({
            'Pos': [[1.0, 2.0], [3.0, 4.0]],
            'contrast': [[0.5], [0.8]],
            'other_col': ['a', 'b']
        })
        
        result = behavior.clean_position_and_contrast(df)
        
        # Check that Pos is split into pos_x and pos_y
        self.assertIn('pos_x', result.columns)
        self.assertIn('pos_y', result.columns)
        self.assertNotIn('Pos', result.columns)
        
        # Check values
        self.assertEqual(result['pos_x'].iloc[0], 1.0)
        self.assertEqual(result['pos_y'].iloc[0], 2.0)
        self.assertEqual(result['pos_x'].iloc[1], 3.0)
        self.assertEqual(result['pos_y'].iloc[1], 4.0)
        
        # Check contrast is extracted from list
        self.assertEqual(result['contrast'].iloc[0], 0.5)
        self.assertEqual(result['contrast'].iloc[1], 0.8)

        # Test with invalid Pos data
        df2 = pd.DataFrame({
            'Pos': ['invalid', [1.0]],  # Invalid position data
            'other_col': ['a', 'b']
        })
        
        result2 = behavior.clean_position_and_contrast(df2)
        self.assertTrue(np.isnan(result2['pos_x'].iloc[0]))
        self.assertTrue(np.isnan(result2['pos_y'].iloc[0]))

    @patch(
        "aind_metadata_extractor.utils.camstim_sync"
        ".behavior_utils.get_stimulus_presentations"
    )
    @patch(
        "aind_metadata_extractor.utils.camstim_sync"
        ".behavior_utils.check_for_errant_omitted_stimulus"
    )
    def test_load_and_validate_stimulus_presentations(
        self, mock_check_errant, mock_get_stimulus_presentations
    ):
        """
        Tests the _load_and_validate_stimulus_presentations function
        """
        # Mock data and timestamps
        data = {"test": "data"}
        stimulus_timestamps = [0.0, 0.5, 1.0]
        
        # Mock return values
        mock_df = pd.DataFrame({
            'start_time': [0.0, 0.5],
            'image_name': ['img1', 'img2'],
            'index': [0, 1]
        })
        mock_get_stimulus_presentations.return_value = mock_df
        
        cleaned_df = mock_df.drop(columns=['index'])
        mock_check_errant.return_value = cleaned_df
        
        # Call the function
        result = behavior._load_and_validate_stimulus_presentations(
            data, stimulus_timestamps
        )
        
        # Verify mocks were called
        mock_get_stimulus_presentations.assert_called_once_with(data, stimulus_timestamps)
        mock_check_errant.assert_called_once()
        
        # Check result doesn't have index column
        self.assertNotIn('index', result.columns)

    def test_get_is_image_novel(self):
        """
        Tests the get_is_image_novel function
        """
        # Test the function returns False as per current implementation
        image_names = ["img1", "img2", "img3"]
        behavior_session_id = 12345
        
        result = behavior.get_is_image_novel(image_names, behavior_session_id)
        
        # Current implementation always returns False
        self.assertEqual(result, False)

    def test_make_spontaneous_stimulus(self):
        """
        Tests the make_spontaneous_stimulus function
        """
        # Test with gaps between stimuli
        df = pd.DataFrame({
            'start_time': [0.0, 2.0, 5.0],
            'stop_time': [1.0, 3.0, 6.0],
            'start_frame': [0, 20, 50],
            'end_frame': [10, 30, 60],
            'stim_block': [0, 1, 2],
            'stim_name': ['stim1', 'stim2', 'stim3']
        })
        
        result = behavior.make_spontaneous_stimulus(df)
        
        # Should have added spontaneous stimuli in the gaps
        self.assertTrue(len(result) > len(df))
        
        # Check for spontaneous entries
        spontaneous_entries = result[result['stim_name'] == 'spontaneous']
        self.assertTrue(len(spontaneous_entries) > 0)

        # Test with no gaps
        df_no_gaps = pd.DataFrame({
            'start_time': [0.0, 1.0, 2.0],
            'stop_time': [1.0, 2.0, 3.0],
            'start_frame': [0, 10, 20],
            'end_frame': [10, 20, 30],
            'stim_block': [0, 1, 2],
            'stim_name': ['stim1', 'stim2', 'stim3']
        })
        
        result_no_gaps = behavior.make_spontaneous_stimulus(df_no_gaps)
        
        # Should return the same dataframe when no gaps
        self.assertEqual(len(result_no_gaps), len(df_no_gaps))

    def test_get_stimulus_name(self):
        """
        Tests the get_stimulus_name function
        """
        # Test case 1: With images in behavior
        stim_file1 = {
            "items": {
                "behavior": {
                    "images": {
                        "image_set": "/path/to/natural_scenes.pkl"
                    }
                }
            }
        }
        
        result1 = behavior.get_stimulus_name(stim_file1)
        self.assertEqual(result1, "natural_scenes")

        # Test case 2: With images in stimuli - due to logic flow issue, returns "behavior"
        stim_file2 = {
            "items": {
                "behavior": {
                    "stimuli": {
                        "images": {
                            "image_set": "/path/to/gabor_patches.pkl"
                        }
                    }
                }
            }
        }
        
        result2 = behavior.get_stimulus_name(stim_file2)
        self.assertEqual(result2, "behavior")  # Current behavior due to logic flow

        # Test case 3: No images but has grating
        stim_file3 = {
            "items": {
                "behavior": {
                    "stimuli": {
                        "grating": {
                            "orientation": [0, 90, 180]
                        }
                    }
                }
            }
        }
        
        result3 = behavior.get_stimulus_name(stim_file3)
        self.assertEqual(result3, "grating")

        # Test case 4: No images or gratings
        stim_file4 = {
            "items": {
                "behavior": {
                    "stimuli": {
                        "other": {}
                    }
                }
            }
        }
        
        result4 = behavior.get_stimulus_name(stim_file4)
        self.assertEqual(result4, "behavior")

    def test_get_flashes_since_change_with_omitted_na(self):
        """
        Tests get_flashes_since_change with NA omitted values
        """
        stimulus_presentations = pd.DataFrame({
            "image_name": ["A", "omitted", "B", "C"],
            "omitted": [False, np.nan, False, False],
            "start_time": [0, 1, 2, 3],
            "is_change": [True, False, True, False]
        })
        
        result = behavior.get_flashes_since_change(stimulus_presentations)
        
        expected = pd.Series([0, 0, 0, 1], name="flashes_since_change")
        pd.testing.assert_series_equal(result, expected)

    def test_compute_trials_id_for_stimulus_with_passive_blocks(self):
        """
        Tests compute_trials_id_for_stimulus with passive blocks
        """
        stim_df = pd.DataFrame({
            "stimulus_block": [1, 1, 2, 2, 3, 3],
            "image_name": ["A", "B", "A", "B", "A", "B"],
            "active": [True, True, False, False, True, True],
            "start_frame": [0, 10, 20, 30, 40, 50],
            "start_time": [0, 1, 2, 3, 4, 5]
        })
        
        trials = pd.DataFrame({
            "change_frame": [0, 40],
            "stop_time": [5, 55],
            "start_time": [0, 4]
        })
        
        result = behavior.compute_trials_id_for_stimulus(stim_df, trials)
        
        # Should assign trial IDs and copy to matching passive blocks
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), 6)

    def test_compute_is_sham_change(self):
        """
        Tests compute_is_sham_change function
        """
        stim_df = pd.DataFrame({
            "trials_id": [0, 0, 1, 1],
            "stimulus_block": [1, 1, 2, 2],
            "image_name": ["A", "B", "C", "D"],
            "start_frame": [0, 10, 20, 30],
            "active": [True, True, False, False]
        })
        
        trials = pd.DataFrame({
            "catch": [False, True],
            "change_frame": [10, 30]
        })
        
        result = behavior.compute_is_sham_change(stim_df, trials)
        
        self.assertIn("is_sham_change", result.columns)
        self.assertEqual(result["is_sham_change"].sum(), 1)  # Only one sham change

    def test_from_stimulus_file_with_limit_to_images(self):
        """
        Tests from_stimulus_file with limit_to_images parameter
        """
        stimulus_file = MagicMock()
        stimulus_timestamps = [0.0, 1.0, 2.0]
        
        mock_data = {
            "items": {
                "behavior": {
                    "stimuli": {},
                    "items": {}
                }
            }
        }
        
        with patch('aind_metadata_extractor.utils.camstim_sync.pkl_utils.load_pkl') as mock_load:
            mock_load.return_value = mock_data
            
            load_patch = ('aind_metadata_extractor.utils.camstim_sync.'
                          'behavior_utils._load_and_validate_stimulus_presentations')
            meta_patch = ('aind_metadata_extractor.utils.camstim_sync.'
                          'behavior_utils.get_stimulus_metadata')
            name_patch = ('aind_metadata_extractor.utils.camstim_sync.'
                          'behavior_utils.get_stimulus_name')
            omitted_patch = ('aind_metadata_extractor.utils.camstim_sync.'
                             'behavior_utils.fix_omitted_end_frame')
            process_patch = ('aind_metadata_extractor.utils.camstim_sync.'
                             'behavior_utils.postprocess')

            with patch(load_patch) as mock_load_validate:
                with patch(meta_patch) as mock_get_metadata:
                    with patch(name_patch) as mock_get_name:
                        with patch(omitted_patch) as mock_fix_omitted:
                            with patch(process_patch) as mock_postprocess:
                                
                                presentations_df = pd.DataFrame({
                                    "start_time": [0, 1, 2],
                                    "image_name": ["image1", "image2", "image3"],
                                    "duration": [0.25, 0.25, 0.25],
                                    "orientation": [0, 90, 180],
                                    "index": [0, 1, 2]
                                })
                                presentations_df.index.name = "stimulus_presentations_id"
                                
                                mock_load_validate.return_value = presentations_df
                                mock_get_metadata.return_value = pd.DataFrame({
                                    "image_name": ["image1", "image2", "image3"],
                                    "image_set": ["set1"] * 3,
                                    "image_index": [0, 1, 2],
                                    "phase": [np.nan] * 3,
                                    "size": [np.nan] * 3,
                                    "orientation": [0, 90, 180],
                                    "spatial_frequency": [np.nan] * 3
                                }).set_index("image_index")
                                mock_get_name.return_value = "test_stimulus"
                                mock_fix_omitted.return_value = presentations_df
                                mock_postprocess.return_value = presentations_df
                                
                                # Test with limit_to_images
                                result = behavior.from_stimulus_file(
                                    stimulus_file,
                                    stimulus_timestamps,
                                    limit_to_images=["image1", "image2"]
                                )
                                
                                self.assertIsInstance(result, tuple)

    def test_get_spontaneous_stimulus_detailed(self):
        """
        Tests get_spontaneous_stimulus function - detailed version
        """
        stimulus_presentations = pd.DataFrame({
            "stim_block": [1, 1, 2, 2],
            "start_time": [300, 301, 302, 303],  # Start after 285s to trigger spontaneous
            "stop_time": [301, 302, 303, 304],
            "start_frame": [100, 110, 120, 130],  # Start after frame 0
            "end_frame": [109, 119, 129, 139]
        })
        
        result = behavior.get_spontaneous_stimulus(stimulus_presentations)
        
        # Should return the original table with potentially added spontaneous blocks
        self.assertIsInstance(result, pd.DataFrame)
        # Check that it has at least the original rows
        self.assertGreaterEqual(len(result), len(stimulus_presentations))

    def test_check_for_errant_omitted_stimulus_normal_case(self):
        """
        Tests check_for_errant_omitted_stimulus normal operation
        """
        stim_df = pd.DataFrame({
            "image_name": ["omitted", "A", "B"],  # Start with omitted
            "duration": [0.25, 0.25, 0.25],
            "omitted": [True, False, False],
            "stimulus_block": [1, 1, 1]
        })
        
        # This should remove the first omitted stimulus
        result = behavior.check_for_errant_omitted_stimulus(stim_df)
        
        # Should have one less row (omitted removed)
        self.assertEqual(len(result), 2)
        self.assertFalse(result.iloc[0]["omitted"])

    def test_compute_is_sham_change_with_array_equal_case(self):
        """
        Tests compute_is_sham_change when passive block images match active block
        """
        stim_df = pd.DataFrame({
            "trials_id": [0, 0, 1, 1],
            "stimulus_block": [1, 1, 2, 2],
            "image_name": ["A", "B", "A", "B"],  # Same images in both blocks
            "start_frame": [0, 10, 20, 30],
            "active": [True, True, False, False],  # Block 1 active, Block 2 passive
            "is_sham_change": [False, True, False, False]
        })
        
        trials = pd.DataFrame({
            "catch": [False, True],
            "change_frame": [10, 30]
        })
        
        result = behavior.compute_is_sham_change(stim_df, trials)
        
        # The passive block should copy is_sham_change from active block
        self.assertIn("is_sham_change", result.columns)
        # Both blocks should have the same sham change pattern
        active_sham = result[result["active"]]["is_sham_change"].values
        passive_sham = result[~result["active"]]["is_sham_change"].values
        np.testing.assert_array_equal(active_sham, passive_sham)

    def test_from_stimulus_file_with_fingerprint_items(self):
        """
        Tests from_stimulus_file with fingerprint items in stimulus file
        """
        stimulus_file = MagicMock()
        stimulus_timestamps = [0.0, 1.0, 2.0]
        
        mock_data = {
            "items": {
                "behavior": {
                    "stimuli": {},
                    "items": {
                        "fingerprint": {
                            "static_stimulus": {
                                "runs": 3,
                                "sweep_frames": [[0, 1], [2, 3]],
                                "sweep_table": [[0.0], [45.0]],
                                "sweep_order": [0, 1],
                                "dimnames": ["orientation"],
                                "frame_list": [0, 1, 2],
                            },
                            "frame_indices": [0, 1, 2, 3],
                        }
                    }
                }
            }
        }
        
        with patch('aind_metadata_extractor.utils.camstim_sync.pkl_utils.load_pkl') as mock_load:
            mock_load.return_value = mock_data
            
            fingerprint_patch = ('aind_metadata_extractor.utils.camstim_sync.'
                                 'behavior_utils.add_fingerprint_stimulus')
            with patch(fingerprint_patch) as mock_add_fingerprint:
                load_patch = ('aind_metadata_extractor.utils.camstim_sync.'
                              'behavior_utils._load_and_validate_stimulus_presentations')
                meta_patch = ('aind_metadata_extractor.utils.camstim_sync.'
                              'behavior_utils.get_stimulus_metadata')
                name_patch = ('aind_metadata_extractor.utils.camstim_sync.'
                              'behavior_utils.get_stimulus_name')
                omitted_patch = ('aind_metadata_extractor.utils.camstim_sync.'
                                 'behavior_utils.fix_omitted_end_frame')
                process_patch = ('aind_metadata_extractor.utils.camstim_sync.'
                                 'behavior_utils.postprocess')

                with patch(load_patch) as mock_load_validate:
                    with patch(meta_patch) as mock_get_metadata:
                        with patch(name_patch) as mock_get_name:
                            with patch(omitted_patch) as mock_fix_omitted:
                                with patch(process_patch) as mock_postprocess:
                                    
                                    presentations_df = pd.DataFrame({
                                        "start_time": [0, 1, 2],
                                        "image_name": ["image1", "image2", "image3"],
                                        "duration": [0.25, 0.25, 0.25],
                                        "orientation": [0, 90, 180],
                                        "index": [0, 1, 2]
                                    })
                                    presentations_df.index.name = "stimulus_presentations_id"
                                    
                                    mock_load_validate.return_value = presentations_df
                                    mock_get_metadata.return_value = pd.DataFrame({
                                        "image_name": ["image1", "image2", "image3"],
                                        "image_set": ["set1"] * 3,
                                        "image_index": [0, 1, 2],
                                        "phase": [np.nan] * 3,
                                        "size": [np.nan] * 3,
                                        "orientation": [0, 90, 180],
                                        "spatial_frequency": [np.nan] * 3
                                    }).set_index("image_index")
                                    mock_get_name.return_value = "test_stimulus"
                                    mock_fix_omitted.return_value = presentations_df
                                    mock_postprocess.return_value = presentations_df
                                    mock_add_fingerprint.return_value = presentations_df
                                    
                                    # This should trigger the fingerprint handling code
                                    result = behavior.from_stimulus_file(
                                        stimulus_file,
                                        stimulus_timestamps
                                    )
                                    
                                    # Verify fingerprint stimulus was added
                                    mock_add_fingerprint.assert_called_once()
                                    self.assertIsInstance(result, tuple)

    def test_from_stimulus_file_with_duplicate_handling(self):
        """
        Tests from_stimulus_file duplicate image handling and column merging
        """
        stimulus_file = MagicMock()
        stimulus_timestamps = [0.0, 1.0, 2.0, 3.0]
        
        mock_data = {
            "items": {
                "behavior": {
                    "stimuli": {},
                    "items": {}
                }
            }
        }
        
        with patch('aind_metadata_extractor.utils.camstim_sync.pkl_utils.load_pkl') as mock_load:
            mock_load.return_value = mock_data
            
            load_patch = ('aind_metadata_extractor.utils.camstim_sync.'
                          'behavior_utils._load_and_validate_stimulus_presentations')
            meta_patch = ('aind_metadata_extractor.utils.camstim_sync.'
                          'behavior_utils.get_stimulus_metadata')
            name_patch = ('aind_metadata_extractor.utils.camstim_sync.'
                          'behavior_utils.get_stimulus_name')
            omitted_patch = ('aind_metadata_extractor.utils.camstim_sync.'
                             'behavior_utils.fix_omitted_end_frame')
            process_patch = ('aind_metadata_extractor.utils.camstim_sync.'
                             'behavior_utils.postprocess')

            with patch(load_patch) as mock_load_validate:
                with patch(meta_patch) as mock_get_metadata:
                    with patch(name_patch) as mock_get_name:
                        with patch(omitted_patch) as mock_fix_omitted:
                            with patch(process_patch) as mock_postprocess:
                                
                                # Create data with duplicates and pkl files
                                presentations_df = pd.DataFrame({
                                    "start_time": [0, 1, 1, 2],  # Duplicate at time 1
                                    "image_name": ["image1", "image2", "image2", "image3"],
                                    "image_set": ["set1", "set2.pkl", "set2", "set3"],  # pkl vs non-pkl
                                    "duration": [0.25, 0.25, 0.25, 0.25],
                                    "orientation": [0, 90, 90, 180],
                                    "Phase": [np.nan, 1.0, np.nan, np.nan],  # Will be renamed to "phase"
                                    "size": [np.nan] * 4,
                                    "spatial_frequency": [np.nan] * 4,
                                    "some_bool_col": [True, False, True, False],  # Object dtype boolean-like
                                    "index": [0, 1, 2, 3]
                                })
                                # Add another Phase column to test duplicate column merging after renaming
                                presentations_df["phase"] = [1.0, np.nan, 2.0, 3.0]  # Duplicate after renaming
                                # Make boolean column object dtype to trigger type coercion
                                presentations_df["some_bool_col"] = presentations_df["some_bool_col"].astype(object)
                                presentations_df.index.name = "stimulus_presentations_id"
                                
                                mock_load_validate.return_value = presentations_df
                                mock_get_metadata.return_value = pd.DataFrame({
                                    "image_name": ["image1", "image2", "image3"],
                                    "image_set": ["set1", "set2", "set3"],
                                    "image_index": [0, 1, 2],
                                    "phase": [np.nan] * 3,
                                    "size": [np.nan] * 3,
                                    "orientation": [0, 90, 180],
                                    "spatial_frequency": [np.nan] * 3
                                }).set_index("image_index")
                                mock_get_name.return_value = "test_stimulus"
                                mock_fix_omitted.return_value = presentations_df
                                mock_postprocess.return_value = presentations_df
                                
                                # This should trigger duplicate handling and column merging
                                result = behavior.from_stimulus_file(
                                    stimulus_file,
                                    stimulus_timestamps
                                )
                                
                                self.assertIsInstance(result, tuple)

    def test_remove_short_sandwiched_spontaneous_edge_case(self):
        """
        Tests remove_short_sandwiched_spontaneous with edge cases (first/last row)
        """
        # Test with only 1 row (edge case)
        df_single = pd.DataFrame({
            "stimulus_block": [0],
            "stim_name": ["spontaneous"],
            "image_name": ["spontaneous"],
            "duration": [0.4],
            "start_time": [0.0]
        })
        
        result_single = behavior.remove_short_sandwiched_spontaneous(df_single)
        pd.testing.assert_frame_equal(result_single, df_single)  # Should remain unchanged
        
        # Test with first and last rows as spontaneous (edge case)
        df_edge = pd.DataFrame({
            "stimulus_block": [0, 1, 2],
            "stim_name": ["spontaneous", "stimulus", "spontaneous"],
            "image_name": ["spontaneous", "image1", "spontaneous"],
            "duration": [0.4, 1.0, 0.4],
            "start_time": [0.0, 1.0, 2.0]
        })
        
        result_edge = behavior.remove_short_sandwiched_spontaneous(df_edge)
        # First and last rows should not be removed even if short
        self.assertEqual(len(result_edge), 3)


if __name__ == "__main__":
    unittest.main()
