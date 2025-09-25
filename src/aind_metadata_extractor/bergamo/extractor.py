"""Module to extract metadata from tif files."""

import bisect
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List

from ScanImageTiffReader import ScanImageTiffReader

from aind_metadata_extractor.bergamo.settings import Settings
from aind_metadata_extractor.core import BaseExtractor
from aind_metadata_extractor.models.bergamo import ExtractedInfo, ExtractedInfoItem, RawImageInfo, TifFileGroup


class Extractor(BaseExtractor):
    """Class to manage extracting metadata from files."""

    def __init__(self, settings: Settings):
        """Class constructor."""

        self.settings = settings

    def get_tif_file_locations(self) -> Dict[str, List[Path]]:
        """Scans the input source directory and returns a dictionary of file
        groups in an ordered list. For example, if the directory had
        [neuron2_00001.tif, neuron2_00002.tif, stackPost_00001.tif,
        stackPost_00002.tif, stackPost_00003.tif], then it will return
        { "neuron2": [neuron2_00001.tif, neuron2_00002.tif],
         "stackPost":
           [stackPost_00001.tif, stackPost_00002.tif, stackPost_00003.tif]
        }
        """
        compiled_regex = re.compile(r"^(.*)_.*?(\d+).tif+$")
        tif_file_map = {}
        for root, dirs, files in os.walk(self.settings.input_source):
            for name in files:
                matched = re.match(compiled_regex, name)
                if matched:
                    groups = matched.groups()
                    file_stem = groups[0]
                    # tif_number = groups[1]
                    tif_filepath = Path(os.path.join(root, name))
                    if tif_file_map.get(file_stem) is None:
                        tif_file_map[file_stem] = [tif_filepath]
                    else:
                        bisect.insort(tif_file_map[file_stem], tif_filepath)

            # Only scan the top level files
            break
        return tif_file_map

    @staticmethod
    def flat_dict_to_nested(flat: dict, key_delim: str = ".") -> dict:
        """
        Utility method to convert a flat dictionary into a nested dictionary.
        Modified from https://stackoverflow.com/a/50607551
        Parameters
        ----------
        flat : dict
          Example {"a.b.c": 1, "a.b.d": 2, "e.f": 3}
        key_delim : str
          Delimiter on dictionary keys. Default is '.'.

        Returns
        -------
        dict
          A nested dictionary like {"a": {"b": {"c":1, "d":2}, "e": {"f":3}}
        """

        def __nest_dict_rec(k, v, out) -> None:
            """Simple recursive method being called."""
            k, *rest = k.split(key_delim, 1)
            if rest:
                __nest_dict_rec(rest[0], v, out.setdefault(k, {}))
            else:
                out[k] = v

        result = {}
        for flat_key, flat_val in flat.items():
            __nest_dict_rec(flat_key, flat_val, result)
        return result

    def extract_raw_info_from_file(self, file_path: Path) -> RawImageInfo:
        """
        Use ScanImageTiffReader to read metadata from a single file and parse
        it into a RawImageInfo object
        Parameters
        ----------
        file_path : Path

        Returns
        -------
        RawImageInfo

        """
        with ScanImageTiffReader(str(file_path)) as reader:
            reader_metadata = reader.metadata()
            reader_shape = reader.shape()
            reader_descriptions = [
                dict([(s.split(" = ", 1)[0], s.split(" = ", 1)[1]) for s in reader.description(i).strip().split("\n")])
                for i in range(0, len(reader))
            ]

        metadata_first_part = reader_metadata.split("\n\n")[0]
        flat_metadata_header_dict = dict(
            [(s.split(" = ", 1)[0], s.split(" = ", 1)[1]) for s in metadata_first_part.split("\n")]
        )
        metadata_dict = self.flat_dict_to_nested(flat_metadata_header_dict)
        reader_metadata_json = json.loads(reader_metadata.split("\n\n")[1])
        # Move SI dictionary up one level
        if "SI" in metadata_dict.keys():
            si_contents = metadata_dict.pop("SI")
            metadata_dict.update(si_contents)
        return RawImageInfo(
            reader_shape=reader_shape,
            reader_metadata_header=metadata_dict,
            reader_metadata_json=reader_metadata_json,
            reader_descriptions=reader_descriptions,
        )

    @staticmethod
    def map_raw_image_info_to_tif_file_group(
        raw_image_info: RawImageInfo,
    ) -> TifFileGroup:
        """
        Map raw image info to a tiff file group type
        Parameters
        ----------
        raw_image_info : RawImageInfo

        Returns
        -------
        TifFileGroup

        """
        header = raw_image_info.reader_metadata_header
        if header.get("hPhotostim", dict()).get("status") in [
            "'Running'",
            "Running",
        ]:
            return TifFileGroup.PHOTOSTIM
        elif (
            header.get("hIntegrationRoiManager", dict()).get("enable") == "true"
            and header.get("hIntegrationRoiManager", dict()).get("outputChannelsEnabled") == "true"
            and header.get("extTrigEnable", dict()) == "1"
        ):
            return TifFileGroup.BEHAVIOR
        elif header.get("hStackManager", dict()).get("enable") == "true":
            return TifFileGroup.STACK
        else:
            return TifFileGroup.SPONTANEOUS

    def _extract(self, tif_file_locations: Dict[str, List[Path]]) -> ExtractedInfo:
        """
        Loop through list of files and use ScanImageTiffReader to read metadata
        Parameters
        ----------
        tif_file_locations : Dict[str, List[Path]]


        """
        extracted_info = []
        for file_stem, files in tif_file_locations.items():
            last_idx = -1
            metadata_extracted = False
            raw_info = None
            while not metadata_extracted:
                try:
                    last_file = files[last_idx]
                    raw_info = self.extract_raw_info_from_file(last_file)
                    metadata_extracted = True
                except Exception as e:
                    logging.warning(e)
                    last_idx -= 1
            raw_info_first_file = self.extract_raw_info_from_file(files[0])
            tif_file_group = self.map_raw_image_info_to_tif_file_group(raw_image_info=raw_info)
            extracted_item = ExtractedInfoItem(
                raw_info_last_file=raw_info,
                raw_info_first_file=raw_info_first_file,
                file_stem=file_stem,
                tif_file_group=tif_file_group,
                files=files,
            )
            extracted_info.append(extracted_item)

        return ExtractedInfo(info=extracted_info)

    def run_job(self):
        """Main entrypoint to extract info and save to file"""
        tif_file_locations = self.get_tif_file_locations()
        self.metadata = self._extract(tif_file_locations)
        with open(self.settings.output_filepath, "w") as f:
            json.dump(self.metadata.model_dump(mode="json"), f, indent=3)


if __name__ == "__main__":
    sys_args = sys.argv[1:]
    if len(sys_args) == 2 and sys_args[0] == "--settings":
        main_settings = Settings.model_validate_json(sys_args[1])
    else:
        # Pull from env vars and command line args
        # noinspection PyArgumentList
        main_settings = Settings()
    main_job = Extractor(settings=main_settings)
    main_job.run_job()
