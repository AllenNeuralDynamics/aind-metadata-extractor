"""Test cases for SmartSPIM utility functions."""

import unittest
from datetime import datetime
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import patch, mock_open

from aind_metadata_extractor.smartspim.utils import (
    read_json_as_dict,
    get_anatomical_direction,
    digest_asi_line,
    get_session_end,
    get_excitation_emission_waves,
    parse_channel_name,
    ensure_list,
)


class TestUtils(unittest.TestCase):
    """Test utility functions for SmartSPIM extractor."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_channels = ["Ex_488_Em_525", "Ex_561_Em_593", "Ex_639_Em_667"]
        self.test_metadata = {
            "session_config": {"obj_magnification": "3.600000"},
            "wavelength_config": {"488": {"power_left": "75.00"}},
            "tile_config": {"t_0": {"Exposure": "2"}},
        }

    @patch("builtins.open", new_callable=mock_open, read_data="{}")
    @patch("os.path.exists")
    @patch("json.load")
    def test_read_json_as_dict_valid_file(self, mock_json_load, mock_exists, mock_file):
        """Test reading valid JSON file."""
        mock_exists.return_value = True
        mock_json_load.return_value = self.test_metadata

        result = read_json_as_dict("/fake/path.json")
        self.assertEqual(result, self.test_metadata)

    @patch("os.path.exists")
    def test_read_json_as_dict_nonexistent_file(self, mock_exists):
        """Test reading non-existent JSON file returns empty dict."""
        mock_exists.return_value = False
        result = read_json_as_dict("/nonexistent/file.json")
        self.assertEqual(result, {})

    @patch("json.load")
    @patch("json.loads")
    @patch("builtins.open")
    @patch("os.path.exists")
    def test_read_json_as_dict_invalid_json(self, mock_exists, mock_open_func, mock_json_loads, mock_json_load):
        """Test reading invalid JSON file."""
        mock_exists.return_value = True

        # Mock json.load to raise JSONDecodeError on first call
        mock_json_load.side_effect = json.JSONDecodeError("Expecting value", "doc", 0)

        # Mock open calls: first for text mode (fails), second for binary mode (succeeds)
        mock_text_file = mock_open()()
        mock_binary_file = mock_open(read_data=b"invalid json")()
        mock_open_func.side_effect = [mock_text_file, mock_binary_file]

        # Mock json.loads to also raise JSONDecodeError for the binary fallback
        mock_json_loads.side_effect = json.JSONDecodeError("Expecting value", "doc", 0)

        with self.assertRaises(json.JSONDecodeError):
            read_json_as_dict("/fake/path.json")

    @patch("json.load")
    @patch("json.loads")
    @patch("builtins.open")
    @patch("os.path.exists")
    def test_read_json_as_dict_unicode_error(self, mock_exists, mock_open_func, mock_json_loads, mock_json_load):
        """Test reading JSON with unicode decode errors."""
        mock_exists.return_value = True

        # Mock json.load to raise UnicodeDecodeError on first call
        mock_json_load.side_effect = UnicodeDecodeError("utf-8", b"", 0, 1, "invalid start byte")

        # Mock open calls: first for text mode (fails), second for binary mode (succeeds)
        mock_text_file = mock_open()()
        mock_binary_file = mock_open(read_data=b'{"test": "valid"}')()
        mock_open_func.side_effect = [mock_text_file, mock_binary_file]

        # Mock json.loads to return the expected result after decoding with errors='replace'
        mock_json_loads.return_value = {"test": "valid"}

        result = read_json_as_dict("/fake/path.json")
        # Should handle the error and return data from binary fallback
        self.assertEqual(result, {"test": "valid"})

    def test_get_anatomical_direction_basic(self):
        """Test basic anatomical direction conversion."""
        test_cases = [
            ("anterior posterior", "anterior_posterior"),
            ("Left Right", "left_right"),
            ("  Superior Inferior  ", "superior_inferior"),
            ("dorsal_ventral", "dorsal_ventral"),
        ]

        for input_dir, expected in test_cases:
            with self.subTest(input_dir=input_dir):
                result = get_anatomical_direction(input_dir)
                self.assertEqual(result, expected)

    def test_digest_asi_line_valid_timestamp(self):
        """Test extracting datetime from valid ASI line."""
        test_line = b"8/19/2025 7:03:25 PM Some additional content"
        result = digest_asi_line(test_line.decode())

        expected = datetime(2025, 8, 19, 19, 3, 25)
        self.assertEqual(result, expected)

    def test_digest_asi_line_am_timestamp(self):
        """Test extracting AM timestamp from ASI line."""
        test_line = b"8/19/2025 7:03:25 AM Some additional content"
        result = digest_asi_line(test_line.decode())

        expected = datetime(2025, 8, 19, 7, 3, 25)
        self.assertEqual(result, expected)

    def test_digest_asi_line_midnight_timestamp(self):
        """Test extracting midnight timestamp from ASI line."""
        test_line = b"8/19/2025 12:00:00 AM Some additional content"
        result = digest_asi_line(test_line.decode())

        expected = datetime(2025, 8, 19, 0, 0, 0)
        self.assertEqual(result, expected)

    def test_digest_asi_line_whitespace_only(self):
        """Test that whitespace-only line returns None."""
        test_line = "   \n\t  "
        result = digest_asi_line(test_line)
        self.assertIsNone(result)

    def test_digest_asi_line_malformed_input(self):
        """Test that malformed input returns None."""
        # Test with insufficient parts
        result = digest_asi_line("single_part")
        self.assertIsNone(result)

        # Test with invalid date format
        result = digest_asi_line("invalid/date/format 12:30:45 PM")
        self.assertIsNone(result)

        # Test with invalid time format
        result = digest_asi_line("8/19/2025 invalid:time PM")
        self.assertIsNone(result)

    def test_get_session_end(self):
        """Test getting session end time from ASI file."""
        asi_content = [
            b"8/19/2025 7:03:25 PM Start of session\n",
            b"8/19/2025 7:15:30 PM Middle of session\n",
            b"8/19/2025 7:46:00 PM End of session\n",
            b"Some non-timestamp line\n",
            b"   \n",  # whitespace line
        ]

        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            f.writelines(asi_content)
            temp_path = f.name

        try:
            result = get_session_end(Path(temp_path))
            expected = datetime(2025, 8, 19, 19, 46, 0)
            self.assertEqual(result, expected)
        finally:
            os.unlink(temp_path)

    def test_get_excitation_emission_waves(self):
        """Test extracting excitation and emission wavelengths."""
        result = get_excitation_emission_waves(self.test_channels)

        expected = {"488": 525, "561": 593, "639": 667}
        self.assertEqual(result, expected)

    def test_get_excitation_emission_waves_with_prefixes(self):
        """Test extracting wavelengths with Em_ and Ex_ prefixes."""
        channels_with_prefixes = ["Em_488_Ex_525", "Em_561_Ex_593", "Ex_639_Em_667"]

        result = get_excitation_emission_waves(channels_with_prefixes)
        expected = {"488": 525, "561": 593, "639": 667}
        self.assertEqual(result, expected)

    def test_parse_channel_name_semicolon_format(self):
        """Test parsing channel name with semicolon format."""
        channel_str = "Laser = 445; Emission Filter = 469/35"
        result = parse_channel_name(channel_str)
        self.assertEqual(result, "Ex_445_Em_469")

    def test_parse_channel_name_comma_format(self):
        """Test parsing channel name with comma format."""
        channel_str = "Laser = 488, Emission Filter = 525/50"
        result = parse_channel_name(channel_str)
        self.assertEqual(result, "Ex_488_Em_525")

    def test_parse_channel_name_bandwidth_removal(self):
        """Test that bandwidth info after slash is removed."""
        channel_str = "Laser = 561; Emission Filter = 593/40"
        result = parse_channel_name(channel_str)
        self.assertEqual(result, "Ex_561_Em_593")

    def test_ensure_list_with_list_input(self):
        """Test ensure_list with list input."""
        test_list = ["item1", "item2", "item3"]
        result = ensure_list(test_list)
        self.assertEqual(result, test_list)
        self.assertIs(result, test_list)  # Should return same object

    def test_ensure_list_with_string_input(self):
        """Test ensure_list with string input."""
        test_string = "single_item"
        result = ensure_list(test_string)
        self.assertEqual(result, ["single_item"])

    def test_ensure_list_with_empty_string(self):
        """Test ensure_list with empty string."""
        result = ensure_list("")
        self.assertEqual(result, [])

    def test_ensure_list_with_whitespace_string(self):
        """Test ensure_list with whitespace-only string."""
        result = ensure_list("   \n\t  ")
        self.assertEqual(result, [])

    def test_ensure_list_with_none(self):
        """Test ensure_list with None input."""
        result = ensure_list(None)
        self.assertEqual(result, [])

    def test_ensure_list_with_numeric_input(self):
        """Test ensure_list with numeric input."""
        result = ensure_list(42)
        self.assertEqual(result, [])

    def test_ensure_list_with_dict_input(self):
        """Test ensure_list with dictionary input."""
        test_dict = {"key": "value"}
        result = ensure_list(test_dict)
        self.assertEqual(result, [])

    def test_integration_channel_processing(self):
        """Test integration of channel processing functions."""
        # Test the workflow from SLIMS format to filter mapping
        slims_channels = [
            "Laser = 488; Emission Filter = 525/50",
            "Laser = 561, Emission Filter = 593/40",
            "Laser = 639; Emission Filter = 667/30",
        ]

        # Parse channel names
        parsed_channels = [parse_channel_name(ch) for ch in slims_channels]
        expected_parsed = ["Ex_488_Em_525", "Ex_561_Em_593", "Ex_639_Em_667"]
        self.assertEqual(parsed_channels, expected_parsed)

        # Get excitation/emission mapping
        filter_mapping = get_excitation_emission_waves(parsed_channels)
        expected_mapping = {"488": 525, "561": 593, "639": 667}
        self.assertEqual(filter_mapping, expected_mapping)


if __name__ == "__main__":
    unittest.main()
