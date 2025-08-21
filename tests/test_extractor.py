"""Test cases for SmartSPIM extractor."""

import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime

from aind_metadata_extractor.smartspim.extractor import SmartspimExtractor
from aind_metadata_extractor.smartspim.job_settings import JobSettings
from tests.resources.smartspim.example_metadata import (
    example_metadata_info,
    example_imaging_info_from_slims,
    example_session_end_time
)


class TestSmartspimExtractor(unittest.TestCase):
    """Test SmartspimExtractor functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.job_settings_dict = {
            "subject_id": "804714",
            "metadata_service_path": "https://api.test.com/smartspim",
            "input_source": "/data/SmartSPIM_2025-08-19_15-03-00",
        }
        
        self.expected_channels = [
            "Ex_488_Em_525",
            "Ex_561_Em_593",
            "Ex_639_Em_667"
        ]

    def test_extractor_initialization(self):
        """Test SmartspimExtractor initialization."""
        extractor = SmartspimExtractor(self.job_settings_dict)

        self.assertIsInstance(extractor.job_settings, JobSettings)
        self.assertEqual(extractor.job_settings.subject_id, "804714")
        self.assertEqual(
            extractor.job_settings.metadata_service_path,
            "https://api.test.com/smartspim"
        )

    @patch('aind_metadata_extractor.smartspim.extractor.os.listdir')
    @patch('aind_metadata_extractor.smartspim.extractor.os.path.isdir')
    @patch('aind_metadata_extractor.smartspim.extractor.read_json_as_dict')
    @patch('aind_metadata_extractor.smartspim.extractor.get_session_end')
    @patch('pathlib.Path.exists')
    def test_extract_metadata_from_microscope_files(
            self, mock_exists, mock_get_session_end, mock_read_json,
            mock_isdir, mock_listdir):
        """Test extracting metadata from microscope files."""
        # Setup mocks
        mock_exists.return_value = True
        mock_listdir.return_value = self.expected_channels
        mock_isdir.return_value = True
        mock_read_json.return_value = example_metadata_info
        mock_get_session_end.return_value = example_session_end_time

        extractor = SmartspimExtractor(self.job_settings_dict)
        result = extractor._extract_metadata_from_microscope_files()

        # Check that all expected keys are present
        expected_keys = [
            'session_config', 'wavelength_config', 'tile_config',
            'session_start_time', 'session_end_time', 'filter_mapping'
        ]
        for key in expected_keys:
            self.assertIn(key, result)

        # Check specific values
        self.assertEqual(
            result['session_config'],
            example_metadata_info['session_config']
        )
        self.assertEqual(
            result['wavelength_config'],
            example_metadata_info['wavelength_config']
        )
        self.assertEqual(result['session_end_time'], example_session_end_time)

        # Check filter mapping
        expected_filter_mapping = {"488": 525, "561": 593, "639": 667}
        self.assertEqual(result['filter_mapping'], expected_filter_mapping)

    @patch('aind_metadata_extractor.smartspim.extractor.read_json_as_dict')
    @patch('aind_metadata_extractor.smartspim.extractor.os.listdir')
    @patch('aind_metadata_extractor.smartspim.extractor.os.path.isdir')
    @patch('pathlib.Path.exists')
    def test_extract_metadata_missing_asi_file(self, mock_exists, 
                                               mock_isdir, mock_listdir,
                                               mock_read_json):
        """Test extraction fails when ASI file is missing."""
        # Setup directory mocks
        mock_listdir.return_value = self.expected_channels
        mock_isdir.return_value = True
        mock_read_json.return_value = {}  # Won't be called but good to have
        
        def side_effect():
            """Mock side effect for Path.exists()"""
            # We can't access self here, but we can control the return value
            print("Path exists check called")  # Debug print
            print("Returning False for ASI file")  # Debug print
            return False
        
        mock_exists.side_effect = side_effect
        
        extractor = SmartspimExtractor(self.job_settings_dict)
        
        with self.assertRaises(FileNotFoundError) as context:
            extractor._extract_metadata_from_microscope_files()
        
        self.assertIn("ASI_logging.txt", str(context.exception))

    @patch('aind_metadata_extractor.smartspim.extractor.read_json_as_dict')
    @patch('aind_metadata_extractor.smartspim.extractor.os.listdir')
    @patch('aind_metadata_extractor.smartspim.extractor.os.path.isdir')
    @patch('pathlib.Path.exists')
    def test_extract_metadata_missing_metadata_file(self, mock_exists,
                                                    mock_isdir, mock_listdir,
                                                    mock_read_json):
        """Test extraction fails when metadata file is missing."""
        # Setup directory mocks
        mock_listdir.return_value = self.expected_channels
        mock_isdir.return_value = True
        mock_read_json.return_value = {}  # Won't be called but good to have
        
        def side_effect():
            """Mock side effect for Path.exists()"""
            # Return True for ASI file, False for metadata file
            print("Path exists check called")  # Debug print
            print("Returning True for metadata file check")  # Debug print
            return True
        
        # First call should succeed (ASI file), second call should fail (metadata file)
        mock_exists.side_effect = [True, False]
        
        extractor = SmartspimExtractor(self.job_settings_dict)
        
        with self.assertRaises(FileNotFoundError) as context:
            extractor._extract_metadata_from_microscope_files()
        
        self.assertIn("metadata.json", str(context.exception))

    @patch('aind_metadata_extractor.smartspim.extractor.os.listdir')
    @patch('aind_metadata_extractor.smartspim.extractor.os.path.isdir')
    @patch('aind_metadata_extractor.smartspim.extractor.read_json_as_dict')
    @patch('pathlib.Path.exists')
    def test_extract_metadata_empty_json(self, mock_exists, mock_read_json, 
                                          mock_isdir, mock_listdir):
        """Test extraction fails when metadata JSON is empty."""
        mock_exists.return_value = True
        mock_listdir.return_value = self.expected_channels
        mock_isdir.return_value = True
        
        # Return metadata with None values
        empty_metadata = {
            "session_config": None,
            "wavelength_config": None,
            "tile_config": None
        }
        mock_read_json.return_value = empty_metadata
        
        extractor = SmartspimExtractor(self.job_settings_dict)
        
        with self.assertRaises(ValueError) as context:
            extractor._extract_metadata_from_microscope_files()
        
        self.assertIn("Metadata json is empty", str(context.exception))

    def test_extract_metadata_invalid_input_source(self):
        """Test extraction fails with invalid input source."""
        # Create a new dict instead of modifying original to avoid type issues
        invalid_settings = {
            "subject_id": "804714",
            "metadata_service_path": "https://api.test.com/smartspim",
            "input_source": None,  # This will cause the error
        }

        extractor = SmartspimExtractor(invalid_settings)

        with self.assertRaises(ValueError) as context:
            extractor._extract_metadata_from_microscope_files()

        self.assertIn(
            "input_source must be a valid path", 
            str(context.exception)
        )

    def test_extract_metadata_list_input_source(self):
        """Test extraction with list input source."""
        # Create a new dict to handle list type properly
        list_settings = {
            "subject_id": "804714",
            "metadata_service_path": "https://api.test.com/smartspim",
            "input_source": [
                "/data/SmartSPIM_2025-08-19_15-03-00",
                "/data/additional_path"
            ]
        }

        with patch('pathlib.Path.exists') as mock_exists, \
             patch('aind_metadata_extractor.smartspim.extractor.os.listdir') as mock_listdir, \
             patch('aind_metadata_extractor.smartspim.extractor.os.path.isdir') as mock_isdir, \
             patch('aind_metadata_extractor.smartspim.extractor.read_json_as_dict') as mock_read_json, \
             patch('aind_metadata_extractor.smartspim.extractor.get_session_end') as mock_get_session_end:
            
            mock_exists.return_value = True
            mock_listdir.return_value = self.expected_channels
            mock_isdir.return_value = True
            mock_read_json.return_value = example_metadata_info
            mock_get_session_end.return_value = example_session_end_time
            
            extractor = SmartspimExtractor(list_settings)
            result = extractor._extract_metadata_from_microscope_files()
            
            # Should work and use the first path in the list
            self.assertIn('session_config', result)

    @patch('aind_metadata_extractor.smartspim.extractor.requests.get')
    def test_extract_metadata_from_slims_success(self, mock_get):
        """Test successful SLIMS metadata extraction."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [example_imaging_info_from_slims]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        extractor = SmartspimExtractor(self.job_settings_dict)
        result = extractor._extract_metadata_from_slims()
        
        self.assertEqual(result, example_imaging_info_from_slims)
        
        # Verify the request was made with correct parameters
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        self.assertEqual(
            call_args[0][0], 
            "https://api.test.com/smartspim"
        )
        self.assertEqual(
            call_args[1]["params"]["subject_id"], 
            "804714"
        )

    @patch('aind_metadata_extractor.smartspim.extractor.requests.get')
    def test_extract_metadata_from_slims_multiple_results(self, mock_get):
        """Test SLIMS extraction fails with multiple results."""
        # Mock response with multiple results
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                example_imaging_info_from_slims,
                example_imaging_info_from_slims  # Duplicate
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        extractor = SmartspimExtractor(self.job_settings_dict)
        
        with self.assertRaises(ValueError) as context:
            extractor._extract_metadata_from_slims()
        
        self.assertIn("More than one imaging session found", str(context.exception))

    @patch('aind_metadata_extractor.smartspim.extractor.requests.get')
    def test_extract_metadata_from_slims_no_results(self, mock_get):
        """Test SLIMS extraction with no results."""
        # Mock response with no results
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": []}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        extractor = SmartspimExtractor(self.job_settings_dict)
        result = extractor._extract_metadata_from_slims()
        
        self.assertEqual(result, {})

    @patch('aind_metadata_extractor.smartspim.extractor.requests.get')
    def test_extract_metadata_from_slims_with_date_filters(self, mock_get):
        """Test SLIMS extraction with date filters."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [example_imaging_info_from_slims]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        extractor = SmartspimExtractor(self.job_settings_dict)
        result = extractor._extract_metadata_from_slims(
            start_date_gte="2025-08-19T00:00:00Z",
            end_date_lte="2025-08-19T23:59:59Z"
        )
        
        self.assertEqual(result, example_imaging_info_from_slims)
        
        # Verify date parameters were passed
        call_args = mock_get.call_args
        params = call_args[1]["params"]
        self.assertIn("start_date_gte", params)
        self.assertIn("end_date_lte", params)

    @patch.object(SmartspimExtractor, '_extract_metadata_from_slims')
    @patch.object(SmartspimExtractor, '_extract_metadata_from_microscope_files')
    def test_extract_full_workflow(self, mock_extract_files, mock_extract_slims):
        """Test the complete extraction workflow."""
        # Mock return values
        mock_file_metadata = {
            "session_config": example_metadata_info["session_config"],
            "wavelength_config": example_metadata_info["wavelength_config"],
            "tile_config": example_metadata_info["tile_config"],
            "session_start_time": datetime(2025, 8, 19, 15, 3, 0),
            "session_end_time": example_session_end_time,
            "filter_mapping": {"488": 525, "561": 593, "639": 667}
        }
        mock_extract_files.return_value = mock_file_metadata
        mock_extract_slims.return_value = example_imaging_info_from_slims
        
        extractor = SmartspimExtractor(self.job_settings_dict)
        result = extractor.extract()
        
        # Verify the result structure
        self.assertIn('file_metadata', result)
        self.assertIn('slims_metadata', result)
        
        # Verify file metadata
        file_metadata = result['file_metadata']
        self.assertEqual(
            file_metadata['session_config'], 
            example_metadata_info["session_config"]
        )
        
        # Verify SLIMS metadata
        slims_metadata = result['slims_metadata']
        self.assertEqual(
            slims_metadata['subject_id'], 
            example_imaging_info_from_slims['subject_id']
        )

    def test_regex_date_extraction(self):
        """Test date extraction from input path."""
        # This tests the regex pattern used in the extractor
        from aind_metadata_extractor.smartspim.extractor import REGEX_DATE
        import re

        test_path = "SmartSPIM_2025-08-19_15-03-00"
        match = re.search(REGEX_DATE, test_path)

        self.assertIsNotNone(match)
        if match:
            self.assertEqual(match.group(), "2025-08-19_15-03-00")

    def test_regex_mouse_id_extraction(self):
        """Test mouse ID extraction regex."""
        from aind_metadata_extractor.smartspim.extractor import REGEX_MOUSE_ID
        import re

        test_string = "mouse_804714_experiment"
        match = re.search(REGEX_MOUSE_ID, test_string)

        self.assertIsNotNone(match)
        if match:
            self.assertEqual(match.group(1), "804714")

    @patch('aind_metadata_extractor.smartspim.extractor.os.listdir')
    @patch('aind_metadata_extractor.smartspim.extractor.os.path.isdir')
    @patch('aind_metadata_extractor.smartspim.extractor.read_json_as_dict')
    @patch('aind_metadata_extractor.smartspim.extractor.get_session_end')
    @patch('pathlib.Path.exists')
    def test_extract_metadata_invalid_date_format(self, mock_exists, 
                                                   mock_get_session_end,
                                                   mock_read_json,
                                                   mock_isdir, mock_listdir):
        """Test extraction fails with invalid date format in path."""
        # Setup mocks
        mock_exists.return_value = True
        mock_listdir.return_value = self.expected_channels
        mock_isdir.return_value = True
        mock_read_json.return_value = example_metadata_info
        mock_get_session_end.return_value = example_session_end_time
        
        # Use invalid date format in input source
        invalid_settings = self.job_settings_dict.copy()
        invalid_settings["input_source"] = "/data/SmartSPIM_invalid_date_format"
        
        extractor = SmartspimExtractor(invalid_settings)
        
        with self.assertRaises(ValueError) as context:
            extractor._extract_metadata_from_microscope_files()
        
        self.assertIn("Error while extracting session date", str(context.exception))


if __name__ == "__main__":
    unittest.main()
