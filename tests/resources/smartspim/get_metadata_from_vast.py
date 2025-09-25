# Skip this file when running unittest
if "unittest" in sys.modules:
    import unittest
    raise unittest.SkipTest("This is an example script, not a test")

from aind_metadata_extractor.smartspim.job_settings import JobSettings
from aind_metadata_extractor.smartspim.extractor import SmartspimExtractor

vast_dir = "/Volumes/aind/stage/SmartSPIM/"

job_settings = JobSettings(
    subject_id="762444",
    metadata_service_path="http://aind-metadata-service/slims/smartspim_imaging",
    output_directory=".",
    acquisition_type="SmartSPIM",
    input_source=vast_dir + "SmartSPIM_762444_2025-07-16_20-47-57",
    slims_datetime="2025-0422T18:30:08.915000Z",
)
extractor = SmartspimExtractor(job_settings=job_settings)
response = extractor.extract()

# Write to a file in the current directory
with open("smartspim.json", "w") as f:
    f.write(response.model_dump_json(indent=4))
