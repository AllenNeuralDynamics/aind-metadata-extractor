"""Utility functions for SmartSPIM metadata extraction."""

from datetime import datetime
import os
import json
import re
from typing import Any, List, Optional


def read_json_as_dict(filepath: str) -> dict:
    """
    Reads a json as dictionary with robust encoding handling.
    Parameters
    ------------------------
    filepath: PathLike
        Path where the json is located.
    Returns
    ------------------------
    dict:
        Dictionary with the data the json has.
    """

    dictionary = {}

    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as json_file:
                dictionary = json.load(json_file)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            print(f"Error reading json with utf-8: {e}")
            print("Falling back to binary read with character replacement.")

            # Fallback: read as binary and replace problematic characters
            with open(filepath, "rb") as json_file:
                data = json_file.read()
                # Decode with replacement, then clean up common problematic characters
                data_str = data.decode("utf-8", errors="replace")
                # Replace common non-UTF-8 micrometer symbols with proper UTF-8 µ
                data_str = data_str.replace("�m", "um")  # Replace replacement char + m with µm
                data_str = data_str.replace("μm", "um")  # Replace Greek mu with micro sign
                dictionary = json.loads(data_str)

    return dictionary


def get_anatomical_direction(anatomical_direction: str) -> str:
    """
    This function returns the correct anatomical
    direction defined in the aind_data_schema.

    Parameters
    ----------
    anatomical_direction: str
        String defining the anatomical direction
        of the data

    Returns
    -------
    AnatomicalDirection: class::Enum
        Corresponding enum defined in the anatomical
        direction class
    """
    anatomical_direction = anatomical_direction.strip().lower().replace(" ", "_")

    return anatomical_direction


def digest_asi_line(line: str) -> Optional[datetime]:
    """
    Scrape a datetime from a non-empty line, otherwise return None

    Parameters
    -----------
    line: str
        Line from the ASI file

    Returns
    -----------
    datetime
        A date that could be parsed from a string
    """

    if line.isspace():
        return None

    try:
        parts = line.split()
        if len(parts) < 3:
            return None

        mdy, hms, ampm = parts[0:3]

        mdy_parts = [int(i) for i in mdy.split("/")]
        ymd = [mdy_parts[i] for i in [2, 0, 1]]

        hms_parts = [int(i) for i in hms.split(":")]

        # Handle AM/PM conversion
        if ampm == "PM" and hms_parts[0] != 12:
            hms_parts[0] += 12
        elif ampm == "AM" and hms_parts[0] == 12:
            hms_parts[0] = 0

        ymdhms = ymd + hms_parts

        dtime = datetime(*ymdhms)
        return dtime
    except (ValueError, IndexError):
        # Return None for lines that can't be parsed as timestamps
        return None


def get_session_end(asi_file: os.PathLike) -> datetime:
    """
    Work backward from the last line until there is a timestamp

    Parameters
    ------------
    asi_file: PathLike
        Path where the ASI metadata file is
        located

    Returns
    ------------
    Date when the session ended
    """

    with open(asi_file, "rb") as file:
        asi_mdata = file.readlines()

    idx = -1
    result = None
    while result is None:
        result = digest_asi_line(asi_mdata[idx].decode())
        idx -= 1

    return result


def get_excitation_emission_waves(channels: List) -> dict:
    """
    Gets the excitation and emission waves for
    the existing channels within a dataset

    Parameters
    ------------
    channels: List[str]
        List with the channels.
        They must contain the emmision
        wavelenght in the name

    Returns
    ------------
    dict
        Dictionary with the excitation
        and emission waves
    """
    excitation_emission_channels = {}

    for channel in channels:
        channel = channel.replace("Em_", "").replace("Ex_", "")
        splitted = channel.split("_")
        excitation_emission_channels[splitted[0]] = int(splitted[1])

    return excitation_emission_channels


def parse_channel_name(channel_str: str) -> str:
    """
    Parses the channel string from SLIMS to a standard format.

    Parameters
    ----------
    channel_str: str
        The channel name to be parsed.
          ex: "Laser = 445; Emission Filter = 469/35"
          ex: "Laser = 488, Emission Filter = 525/50"
    Returns
    -------
    str
        The parsed channel name (ex: "Ex_445_Em_469").
    """
    s = channel_str.replace("Laser", "Ex").replace("Emission Filter", "Em")
    parts = [p.strip() for p in re.split(r"[;,]", s) if p.strip()]
    segments = []
    for part in parts:
        key, val = [t.strip() for t in part.split("=", 1)]
        # discard any bandwidth info after slash
        core = val.split("/", 1)[0]
        segments.append(f"{key}_{core}")

    return "_".join(segments)


def ensure_list(raw: Any) -> List[Any]:
    """
    Turn a value that might be a list, a single string, or None
    into a proper list of strings (or an empty list).
    """
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str) and raw.strip():
        return [raw]
    return []
