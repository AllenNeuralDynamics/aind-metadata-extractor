"""Generate JSON schema for FIP data model."""

import argparse
import logging
import os
from typing import Annotated, Type

import aind_physiology_fip
from aind_behavior_services.base import SchemaVersionedModel
from aind_behavior_services.utils import export_schema
from aind_physiology_fip.data_mappers import ProtoAcquisitionDataSchema
from pydantic import BaseModel, create_model

__VERSION__ = aind_physiology_fip.__version__

logger = logging.getLogger(__name__)


def write_schema_to_file(file_path: str) -> None:
    """Write the JSON schema to a file."""
    logger.info(f"Writing schema to {file_path}. Using aind-physiology-fip version {__VERSION__}")
    schema = export_schema(_patch_model(ProtoAcquisitionDataSchema), remove_root=False)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write((schema))


def _patch_model(m: Type[BaseModel]) -> Type[BaseModel]:
    """Recursively patch a pydantic model to make version fields unsafe."""
    for field_name, field in m.model_fields.items():
        if isinstance(field.annotation, type) and issubclass(field.annotation, SchemaVersionedModel):
            updates = {}
            if "version" in field.annotation.model_fields:
                updates["version"] = Annotated[str, field.annotation.model_fields["version"]]
            if "aind_behavior_services_pkg_version" in field.annotation.model_fields:
                updates["aind_behavior_services_pkg_version"] = Annotated[
                    str, field.annotation.model_fields["aind_behavior_services_pkg_version"]
                ]
            new_model = create_model(
                f"{field.annotation.__name__}UnsafeVersion",
                __base__=field.annotation,
                __doc__=field.annotation.__doc__,
                **updates,
            )
            m.model_fields[field_name].annotation = new_model
            _patch_model(new_model)
        elif isinstance(field.annotation, type) and issubclass(field.annotation, BaseModel):
            _patch_model(field.annotation)
    return m


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
