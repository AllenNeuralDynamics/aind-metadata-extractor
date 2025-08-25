# aind-metadata-extractor

## Install

You should only install the dependencies for the specific extractor you plan to run. You can see the list of available extractors in the `pyproject.toml` file or in the folders in `src/aind_metadata/extractor`

During installation pass the extractor as an optional dependency:

```
pip install 'aind-metadata-extractor[<your-extractor>]'
```

## Run

Each extractor uses a `JobSettings` object to collect necessary information about data and metadata files to create an `Extractor` which is run by calling `.extract()`. For example, for *smartspim*:

```{python}
from pathlib import Path

from aind_metadata_extractor.smartspim.job_settings import JobSettings
from aind_metadata_extractor.smartspim.extractor import SmartspimExtractor

DATA_DIR = Path("<path-to-your-data>)

job_settings=JobSettings(
    subject_id="786846",
    metadata_service_path="http://aind-metadata-service/slims/smartspim_imaging",
    input_source=DATA_DIR+"SmartSPIM_786846_2025-04-22_16-44-50",
    slims_datetime="2025-0422T18:30:08.915000Z"
)
extractor = SmartspimExtractor(job_settings=job_settings)
response = extractor.extract()
```
