# aind-metadata-extractor

## Install

You should only install the dependencies for the specific extractor you plan to run. You can see the list of available extractors in the `pyproject.toml` file or in the folders in `src/aind_metadata/extractor`

During installation pass the extractor as an optional dependency:

```
pip install 'aind-metadata-extractor[<your-extractor>]'
```

## Develop

To build a new extractor, define a new output model in the models/ folder. Then create a new extractor folder and inherit from `BaseExtractor`. Implement the functions:

- `.run_job()` should store the metadata output object (matching the model) in self.metadata and return a dictionary with the `model_dump()` contents
- `._extract()` should perform the actual data loading, metadata-service calls, etc, necessary to build the metadata model and return it

Your extractor comes with an inherited function `.write()` which writes the metadata to the file <extractor>.json.

### Testing

When testing locally you only need to run your own tests (i.e. `coverage run -m unittest discover -s tests/<new-extractor>`). Do not modify the tests for other extractors in your PRs.

Before opening a PR, modify the file `test_and_lint.yml` and add a new test-group:

```
test-group: ['core', 'smartspim', 'mesoscope', 'utils', '<new-extractor>']
```

Then add the test-group settings below that:

```
    - test-group: '<new-extractor>'
    dependencies: '[dev,<new-extractor>]'
    test-path: 'tests/<new-extractor>'
    test-pattern: 'test_*.py'
```

When running on GitHub, all of the test groups will be run independently with their separate dependencies and then their coverage results are gathered together in a final step.

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
    output_directory=".",
    slims_datetime="2025-0422T18:30:08.915000Z"
)
extractor = SmartspimExtractor(job_settings=job_settings)
extractor.run_job()
extractor.write()
```

The results will be saved in `smartspin.json`
