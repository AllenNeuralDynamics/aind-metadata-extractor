"""Simple example demonstrating FiberPhotometryExtractor usage.

To run this example:
    conda create -n fip-extractor python=3.11 -y
    conda activate fip-extractor
    pip install -e .[fip]
    python examples/fip_example.py
"""

import json
from pathlib import Path
from datetime import datetime
from aind_metadata_extractor.fip.job_settings import JobSettings
from aind_metadata_extractor.fip.extractor import FiberPhotometryExtractor

DATA_DIR = Path("/allen/aind/scratch/bruno.cruz/fip_tests/781896_2025-07-18T192910Z/fib/fip_2025-07-18T192959Z")
OUTPUT_FILE = Path(__file__).parent / "fip_extracted_metadata.json"

job_settings = JobSettings(
    data_directory=DATA_DIR,
    subject_id="781896",
    rig_id="323_FIP_OPTO_2",
    mouse_platform_name="Standard",
    active_mouse_platform=False,
    iacuc_protocol="2115",
    notes="Example extraction",
    anaesthesia="none",
    animal_weight_prior=25.0,
    animal_weight_post=25.0,
    session_start_time=datetime(2025, 7, 18, 19, 29, 10),
    session_end_time=datetime(2025, 7, 18, 19, 46, 24),
    rig_config={"rig_id": "323_FIP_OPTO_2"},
    session_config={"session_type": "FIP"},
    output_directory=DATA_DIR,
)
extractor = FiberPhotometryExtractor(job_settings=job_settings)
response = extractor.extract()

with open(OUTPUT_FILE, "w") as f:
    json.dump(response, f, indent=2, default=str)

print(f"Extraction complete! Saved metadata to {OUTPUT_FILE}")

