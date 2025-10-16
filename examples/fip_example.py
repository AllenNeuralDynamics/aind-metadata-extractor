"""Simple example demonstrating FiberPhotometryExtractor usage.

To run this example:
    conda create -n fip-extractor python=3.11 -y
    conda activate fip-extractor
    pip install -e .[fip]
    python examples/fip_example.py
"""

import json
from pathlib import Path
from aind_metadata_extractor.fip.job_settings import JobSettings
from aind_metadata_extractor.fip.extractor import FiberPhotometryExtractor

DATA_DIR = Path("/allen/aind/scratch/bruno.cruz/fip_tests/781896_2025-07-18T192910Z/fib/fip_2025-07-18T192959Z")
OUTPUT_FILE = Path(__file__).parent / "fip_extracted_metadata.json"

job_settings = JobSettings(
    data_directory=DATA_DIR,
    mouse_platform_name="wheel",
    local_timezone="America/Los_Angeles",
    output_directory=DATA_DIR,
)
extractor = FiberPhotometryExtractor(job_settings=job_settings)
response = extractor.extract()

with open(OUTPUT_FILE, "w") as f:
    json.dump(response, f, indent=2, default=str)

print(f"Extraction complete! Saved metadata to {OUTPUT_FILE}")

