"""Data models of extracted information from tiff files."""

from enum import Enum
from pathlib import Path
from typing import List

from pydantic import BaseModel


class TifFileGroup(str, Enum):
    """Type of stimulation a group of files belongs to"""

    BEHAVIOR = "behavior"
    PHOTOSTIM = "photostim"
    SPONTANEOUS = "spontaneous"
    STACK = "stack"


class RawImageInfo(BaseModel):
    """Raw metadata from a tif file"""

    reader_metadata_header: dict
    reader_metadata_json: dict
    # The reader descriptions for the last tif file
    reader_descriptions: List[dict]
    # Looks like [620, 800, 800]
    # [num_of_frames, pixel_width, pixel_height]?
    reader_shape: List[int]


class ExtractedInfoItem(BaseModel):
    """ExtractedInfo that can be used to build Session metadata."""

    raw_info_first_file: RawImageInfo
    raw_info_last_file: RawImageInfo
    tif_file_group: TifFileGroup
    file_stem: str
    files: List[Path]


class ExtractedInfo(BaseModel):
    """Main model to be used downstream."""

    info: List[ExtractedInfoItem]
