"""Tests extracting metadata information from bergamo rig."""

import gzip
import json
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

from aind_metadata_extractor.bergamo.extractor import Extractor
from aind_metadata_extractor.bergamo.models import ExtractedInfo, ExtractedInfoItem, RawImageInfo, TifFileGroup
from aind_metadata_extractor.bergamo.settings import Settings

RESOURCES_DIR = Path(os.path.dirname(os.path.realpath(__file__))) / ".." / "resources" / "bergamo"
EXAMPLE_READER_RESPONSE_PATH = RESOURCES_DIR / "reader_response.json.gz"


class TestBergamoExtractor(unittest.TestCase):
    """Test methods in Bergamo Extractor class."""

    @classmethod
    def setUpClass(cls):
        """Load record object and user settings before running tests."""
        with gzip.open(EXAMPLE_READER_RESPONSE_PATH, "rb") as f:
            reader_response = json.load(f)
        cls.example_job_settings = Settings(
            input_source=RESOURCES_DIR,
            output_filepath=Path("extracted_info.json"),
            experimenter_full_name=["John Apple"],
            subject_id="12345",
            imaging_laser_wavelength=920,  # nm
            fov_imaging_depth=200,  # microns
            fov_targeted_structure="Primary Motor Cortex",
            notes="test upload",
        )
        cls.reader_response = reader_response
        cls.extracted_info_example = ExtractedInfo(
            info=[
                ExtractedInfoItem(
                    raw_info_first_file=RawImageInfo(
                        reader_metadata_header={"hPhotostim": {"status": "Running"}},
                        reader_metadata_json={},
                        reader_descriptions=[],
                        reader_shape=[1466, 256, 512],
                    ),
                    raw_info_last_file=RawImageInfo(
                        reader_metadata_header={"hPhotostim": {"status": "Running"}},
                        reader_metadata_json={},
                        reader_descriptions=[],
                        reader_shape=[1466, 256, 512],
                    ),
                    tif_file_group=TifFileGroup.PHOTOSTIM,
                    file_stem="neuron4",
                    files=[Path("neuron4_00001.tif "), Path("neuron4_00002.tif "), Path("neuron4_00003.tif ")],
                )
            ]
        )

    @patch("os.walk")
    def test_get_tif_file_locations(self, mock_os_walk: MagicMock):
        """Tests _get_tif_file_locations method"""
        mock_os_walk.return_value = [
            (
                "example_dir",
                [],
                [
                    "neuron4_00001.tif",
                    "spont_00001.tif",
                    "spontpost_00001.tif",
                    "spontpost_slm_00001.tif",
                    "spont_slm_00001.tif",
                    "stack_00001.tif",
                ],
            )
        ]
        locations = Extractor(settings=self.example_job_settings).get_tif_file_locations()
        expected_locations = {
            "neuron4": [Path("example_dir") / "neuron4_00001.tif"],
            "spont": [Path("example_dir") / "spont_00001.tif"],
            "spontpost": [Path("example_dir") / "spontpost_00001.tif"],
            "spontpost_slm": [Path("example_dir") / "spontpost_slm_00001.tif"],
            "spont_slm": [Path("example_dir") / "spont_slm_00001.tif"],
            "stack": [Path("example_dir") / "stack_00001.tif"],
        }
        self.assertEqual(expected_locations, locations)

    def test_flat_dict_to_nested(self):
        """Test util method to convert dictionaries from flat to nested."""
        original_input = {
            "SI.LINE_FORMAT_VERSION": 1,
            "SI.VERSION_UPDATE": 0,
            "SI.acqState": "loop",
            "SI.acqsPerLoop": "10000",
            "SI.errorMsg": "",
            "SI.extTrigEnable": "1",
            "SI.fieldCurvatureRxs": "[]",
            "SI.fieldCurvatureZs": "[]",
            "SI.hBeams.enablePowerBox": "false",
            "SI.hBeams.errorMsg": "",
            "SI.hBeams.lengthConstants": "[200 Inf]",
            "SI.hBeams.name": "SI Beams",
        }

        expected_output = {
            "SI": {
                "LINE_FORMAT_VERSION": 1,
                "VERSION_UPDATE": 0,
                "acqState": "loop",
                "acqsPerLoop": "10000",
                "errorMsg": "",
                "extTrigEnable": "1",
                "fieldCurvatureRxs": "[]",
                "fieldCurvatureZs": "[]",
                "hBeams": {
                    "enablePowerBox": "false",
                    "errorMsg": "",
                    "lengthConstants": "[200 Inf]",
                    "name": "SI Beams",
                },
            }
        }

        actual_output = Extractor.flat_dict_to_nested(original_input)
        self.assertEqual(expected_output, actual_output)

    @patch("aind_metadata_extractor.bergamo.extractor.ScanImageTiffReader")
    def test_extract_raw_info_from_file(self, mock_scan_tiff_reader: MagicMock):
        """Tests extract_raw_info_from_file."""
        mock_open_reader = MagicMock()
        mock_open_reader.metadata.return_value = self.reader_response["metadata"]
        mock_open_reader.shape.return_value = self.reader_response["shape"]
        mock_open_reader.description.side_effect = self.reader_response["descriptions"]
        mock_open_reader.__len__.return_value = 1466
        mock_scan_tiff_reader.return_value.__enter__.return_value = mock_open_reader
        raw_info = Extractor(settings=self.example_job_settings).extract_raw_info_from_file(Path("neuron4_00001.tif"))
        mock_scan_tiff_reader.return_value.__exit__.assert_called_once()
        self.assertEqual(
            ["imagingRoiGroup", "photostimRoiGroups", "integrationRoiGroup"],
            list(raw_info.reader_metadata_json["RoiGroups"].keys()),
        )
        self.assertEqual(1466, len(raw_info.reader_descriptions))
        self.assertEqual([1466, 256, 512], raw_info.reader_shape)
        self.assertEqual(
            [
                "LINE_FORMAT_VERSION",
                "PREMIUM",
                "TIFF_FORMAT_VERSION",
                "VERSION_COMMIT",
                "VERSION_MAJOR",
                "VERSION_MINOR",
                "VERSION_UPDATE",
                "acqState",
                "acqsPerLoop",
                "errorMsg",
                "extTrigEnable",
                "fieldCurvatureRxs",
                "fieldCurvatureRys",
                "fieldCurvatureTilt",
                "fieldCurvatureTip",
                "fieldCurvatureZs",
                "hBeams",
                "hCameraManager",
                "hChannels",
                "hConfigurationSaver",
                "hCoordinateSystems",
                "hCycleManager",
                "hDisplay",
                "hFastZ",
                "hIntegrationRoiManager",
                "hMotionManager",
                "hMotors",
                "hPhotostim",
                "hPmts",
                "hRoiManager",
                "hScan2D",
                "hShutters",
                "hSlmScan",
                "hStackManager",
                "hTileManager",
                "hUserFunctions",
                "hWSConnector",
                "hWaveformManager",
                "imagingSystem",
                "loopAcqInterval",
                "name",
                "objectiveResolution",
                "reserverInfo",
                "shutDownScript",
                "startUpScript",
                "userInfo",
                "warnMsg",
            ],
            list(raw_info.reader_metadata_header.keys()),
        )

    def test_map_raw_image_info_to_tif_file_group(self):
        """Tests map_raw_image_info_to_tif_file_group."""
        raw_image_info_base = RawImageInfo(
            reader_shape=[1466, 256, 512],
            reader_descriptions=[],
            reader_metadata_json=dict(),
            reader_metadata_header=dict(),
        )
        group0 = Extractor.map_raw_image_info_to_tif_file_group(
            raw_image_info_base.model_copy(
                update={"reader_metadata_header": {"hPhotostim": {"status": "Running"}}}, deep=True
            )
        )
        group1 = Extractor.map_raw_image_info_to_tif_file_group(
            raw_image_info_base.model_copy(
                update={
                    "reader_metadata_header": {
                        "hIntegrationRoiManager": {"enable": "true", "outputChannelsEnabled": "true"},
                        "extTrigEnable": "1",
                    }
                },
                deep=True,
            )
        )
        group2 = Extractor.map_raw_image_info_to_tif_file_group(
            raw_image_info_base.model_copy(
                update={"reader_metadata_header": {"hStackManager": {"enable": "true"}}}, deep=True
            )
        )
        group3 = Extractor.map_raw_image_info_to_tif_file_group(raw_image_info_base)

        self.assertEqual(TifFileGroup.PHOTOSTIM, group0)
        self.assertEqual(TifFileGroup.BEHAVIOR, group1)
        self.assertEqual(TifFileGroup.STACK, group2)
        self.assertEqual(TifFileGroup.SPONTANEOUS, group3)

    @patch("aind_metadata_extractor.bergamo.extractor.Extractor.extract_raw_info_from_file")
    def test_extract_parsed_metadata_info_from_files(self, mock_extract_info: MagicMock):
        """Tests extract_parsed_metadata_info_from_files."""

        mock_extract_info.side_effect = [
            Exception("Error parsing file."),
            RawImageInfo(
                reader_shape=[1466, 256, 512],
                reader_descriptions=[],
                reader_metadata_json=dict(),
                reader_metadata_header={"hPhotostim": {"status": "Running"}},
            ),
            RawImageInfo(
                reader_shape=[1466, 256, 512],
                reader_descriptions=[],
                reader_metadata_json=dict(),
                reader_metadata_header={"hPhotostim": {"status": "Running"}},
            ),
        ]

        extractor = Extractor(settings=self.example_job_settings)
        with self.assertLogs(level="WARNING") as captured:
            extracted_info = extractor.extract_parsed_metadata_info_from_files(
                tif_file_locations={
                    "neuron4": [Path("neuron4_00001.tif "), Path("neuron4_00002.tif "), Path("neuron4_00003.tif ")]
                }
            )
        expected_info = self.extracted_info_example
        self.assertEqual(expected_info, extracted_info)
        self.assertEqual(["WARNING:root:Error parsing file."], captured.output)

    @patch("aind_metadata_extractor.bergamo.extractor.Extractor.get_tif_file_locations")
    @patch("aind_metadata_extractor.bergamo.extractor.Extractor.extract_parsed_metadata_info_from_files")
    @patch("builtins.open", new_callable=mock_open)
    def test_run_job(self, mock_file_open: MagicMock, mock_extract_info: MagicMock, mock_tif_file_locations: MagicMock):
        """Tests run_job."""
        extracted_info = self.extracted_info_example
        mock_extract_info.return_value = extracted_info
        Extractor(settings=self.example_job_settings).run_job()
        mock_tif_file_locations.assert_called_once()
        mock_file_open.assert_called_once_with(Path("extracted_info.json"), "w")


if __name__ == "__main__":
    unittest.main()
