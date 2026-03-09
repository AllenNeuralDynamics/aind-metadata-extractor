# aind-metadata-extractor

**Extractors** handle pulling metadata from acquisition data files. The output of an extractor is a data model (stored in the `models/` subfolder) which is a contract with the corresponding **mapper** in [aind-metadata-mapper](https://github.com/AllenNeuralDynamics/aind-metadata-mapper/).

Extractors need to be run on the rig immediately following acquisition.

Mappers are run automatically by the `GatherMetadataJob` on the data-transfer-service.

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
    output_directory=".",
    slims_datetime="2025-0422T18:30:08.915000Z"
)
extractor = SmartspimExtractor(job_settings=job_settings)
extractor.run_job()
extractor.write()
```

The results will be saved in `smartspim.json`

## Why

Every data acquisition is required to capture [Acquisition](https://aind-data-schema.readthedocs.io/en/latest/acquisition.html) metadata. In many situations this requires accessing the raw data files, which can mean installing custom rig-specific libraries. To maintain a clean separation of logic we are putting all rig-specific code into the **extractors** in this repository and keeping any code related to transforming to [aind-data-schema](https://github.com/allenNeuralDynamics/aind-data-schema) in the **mapper**. In between the extractor and the mapper there is a **contract**, a pydantic model that contains all of the necessary information to run the mapper.

This pattern also allows us to keep any code that access metadata services (e.g. [aind-metadata-service](http://aind-metadata-service)) off of the rigs.

Finally, this separation means that your mappers can be run automatically! You can find more details about mappers in the [aind-metadata-mapper](https://github.com/AllenNeuralDynamics/aind-metadata-mapper/) repository.

## Develop

The only requirement for extractors is that you output a file `<your-extractor-name>.json` which validates against the corresponding model in the `models/` subfolder.

### Define a model

Define a new contract model in the `models/` folder. Your model class should inherit from `pydantic.BaseModel`. You can nest sub-models if you find it helpful for organizing your metadata, see `models/smartspim.py` as an example.

### Define extractor code

You do not need to keep your extractor code in this repository, but if you do put it here it will make it easier for us to coordinate updates with you in the future as metadata requirements evolve.

### Option 1: Extractor code maintained elsewhere

Have your extractor code (in your acquisition code) output a file named `<your-extractor-name>.json` that is validated against your model. The intermediate model file should be stored alongside any other metadata files you are providing (usually the instrument.json, at a minimum).

### Option 2: Extractor code in aind-metadata-extractor

Create a new extractor folder with a matching name and inherit from `BaseExtractor`. Implement the functions:

- `.run_job()` accepts a `JobSettings` object as a parameter and should store the metadata output object (matching the model) in `self.metadata`. Return a *dictionary* with the `model_dump()` contents.
- `._extract()` should perform the actual data loading, metadata-service calls, etc, necessary to build the metadata model and return it. This function should return the actual model, validated against what is in the `models/` folder.

Extractor classes inherit the `.write()` function, which writes the metadata to the file <your-extractor-name>.json. Users will then be able to run your extractor according to the instructions in the [run](#run) block, above.

### Testing

To run all test suites locally (mirroring GitHub Actions, with isolated virtual environments per suite):

```bash
python run_tests.py
```

To run only the tests for the extractor you are working on:

```bash
coverage run --source=aind_metadata_extractor.<your-extractor> -m unittest discover -s tests/<your-extractor> -p 'test_*.py'
coverage report
```

For example, for *mesoscope*:

```bash
coverage run --source=aind_metadata_extractor.mesoscope -m unittest discover -s tests/mesoscope -p 'test_*.py'
coverage report
```

Before opening a PR, add a new GitHub Actions workflow file for your extractor. Copy an existing workflow (e.g. `.github/workflows/test_smartspim.yml`) and update the following fields:

- `name:` — change to `Test - <your-extractor>`
- `filters:` paths — point to your extractor's source and test folders
- `pip install` — add your extractor as an optional dependency: `.[dev,<your-extractor>]`
- `coverage run --source` — set to `aind_metadata_extractor.<your-extractor>`
- `unittest discover -s` — set to `tests/<your-extractor>`
