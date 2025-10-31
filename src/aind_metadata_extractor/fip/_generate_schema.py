import argparse
import logging
import os

import aind_physiology_fip
from aind_behavior_services.utils import export_schema
from aind_physiology_fip.data_mappers import ProtoAcquisitionDataSchema

__VERSION__ = aind_physiology_fip.__version__

logger = logging.getLogger(__name__)


def write_schema_to_file(file_path: str) -> None:
    logger.info(f"Writing schema to {file_path}. Using aind-physiology-fip version {__VERSION__}")
    schema = export_schema(ProtoAcquisitionDataSchema, remove_root=False)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write((schema))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export schema to JSON file")
    parser.add_argument("--filepath", default=None, help="Path to output JSON file")
    args = parser.parse_args()

    if args.filepath is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, "..", "models", "fip.json")
    else:
        filepath = args.filepath

    write_schema_to_file(filepath)
