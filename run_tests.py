#!/usr/bin/env python3
"""
Run tests locally in the same way they run in GitHub Actions.

Each test suite installs its own set of extras into an isolated virtual
environment so dependency conflicts between suites are avoided.

Usage:
    python run_tests.py                     # run all suites
    python run_tests.py core bergamo        # run specific suites
    python run_tests.py --list              # show available suites
    python run_tests.py --no-recreate       # reuse existing venvs
    python run_tests.py --no-coverage       # skip coverage report

Available suites: core, bergamo, mesoscope, smartspim, utils
"""

import argparse
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
VENVS_DIR = ROOT / ".test_venvs"


# Mirror of each GitHub Actions workflow
def _omit(*rel_paths: str) -> list[str]:
    """Return absolute omit glob patterns for coverage."""
    return [str(ROOT / p) for p in rel_paths]


SUITES = {
    "core": {
        "extras": "dev",
        # Use module name so coverage resolves via the editable install,
        # consistent with pyproject.toml [tool.coverage.run] source setting.
        "source": "aind_metadata_extractor",
        # Omit subdirectory modules and utils; keep only core.py / __init__.py
        "omit": _omit(
            "src/aind_metadata_extractor/*/",
            "src/aind_metadata_extractor/utils/*",
        ),
        "test_dir": "tests/core",
    },
    "bergamo": {
        "extras": "dev,bergamo",
        "source": "aind_metadata_extractor.bergamo",
        "omit": None,
        "test_dir": "tests/bergamo",
    },
    "mesoscope": {
        "extras": "dev,mesoscope",
        "source": "aind_metadata_extractor.mesoscope",
        "omit": None,
        "test_dir": "tests/mesoscope",
    },
    "smartspim": {
        "extras": "dev,smartspim",
        "source": "aind_metadata_extractor.smartspim",
        "omit": None,
        "test_dir": "tests/smartspim",
    },
    "utils": {
        "extras": "dev,utils",
        "source": "aind_metadata_extractor.utils",
        "omit": None,
        "test_dir": "tests/utils",
    },
}


def python_in_venv(venv_path: Path) -> Path:
    """Return path to the python executable in the given venv."""
    return venv_path / "bin" / "python"


def ensure_venv(suite_name: str, recreate: bool) -> Path:
    """Ensure a virtual environment exists for the given suite."""
    venv_path = VENVS_DIR / suite_name
    if venv_path.exists() and not recreate:
        print(f"  [venv] Reusing existing venv at {venv_path.relative_to(ROOT)}")
        return venv_path
    if venv_path.exists():
        import shutil

        print("  [venv] Removing old venv...")
        shutil.rmtree(venv_path)
    print(f"  [venv] Creating venv at {venv_path.relative_to(ROOT)}")
    venv.create(str(venv_path), with_pip=True, clear=True)
    return venv_path


def install_deps(suite_name: str, cfg: dict, venv_path: Path) -> bool:
    """Install dependencies for the given suite into its venv."""
    extras = cfg["extras"]
    print(f"  [pip]  Installing .[{extras}]")
    result = subprocess.run(
        [str(python_in_venv(venv_path)), "-m", "pip", "install", "-e", f".[{extras}]"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  [pip]  FAILED\n{result.stdout}\n{result.stderr}")
        return False
    return True


def run_tests(suite_name: str, cfg: dict, venv_path: Path, with_coverage: bool) -> bool:
    """Run tests for the given suite in its venv."""
    python = str(python_in_venv(venv_path))
    test_dir = cfg["test_dir"]
    source = cfg["source"]
    omit = cfg.get("omit")

    if with_coverage:
        cmd = [python, "-m", "coverage", "run", f"--source={source}"]
        if omit:
            # Pass each pattern as a separate --omit flag for reliable expansion
            for pattern in omit:
                cmd.append(f"--omit={pattern}")
        cmd += ["-m", "unittest", "discover", "-s", test_dir, "-p", "test_*.py"]
    else:
        cmd = [python, "-m", "unittest", "discover", "-s", test_dir, "-p", "test_*.py"]

    print(f"  [test] {' '.join(cmd[2:])}")
    result = subprocess.run(cmd, cwd=ROOT, text=True)

    if result.returncode != 0:
        return False

    if with_coverage:
        cov_result = subprocess.run(
            [python, "-m", "coverage", "report"],
            cwd=ROOT,
            text=True,
        )
        if cov_result.returncode != 0:
            return False

    return True


def run_suite(suite_name: str, recreate: bool, with_coverage: bool) -> bool:
    """Run the given test suite."""
    cfg = SUITES[suite_name]
    print(f"\n{'='*60}")
    print(f"Suite: {suite_name}  (extras: [{cfg['extras']}])")
    print(f"{'='*60}")

    venv_path = ensure_venv(suite_name, recreate)

    if not install_deps(suite_name, cfg, venv_path):
        print(f"  [FAIL] Dependency installation failed for '{suite_name}'")
        return False

    success = run_tests(suite_name, cfg, venv_path, with_coverage)
    status = "PASS" if success else "FAIL"
    print(f"\n  [{status}] {suite_name}")
    return success


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run tests locally, mirroring GitHub Actions workflows.")
    parser.add_argument(
        "suites",
        nargs="*",
        choices=[*SUITES.keys(), []],
        metavar="SUITE",
        help=f"Suites to run (default: all). Choices: {', '.join(SUITES)}",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available test suites and exit.",
    )
    parser.add_argument(
        "--no-recreate",
        dest="recreate",
        action="store_false",
        default=True,
        help="Reuse existing virtual environments instead of recreating them.",
    )
    parser.add_argument(
        "--no-coverage",
        dest="coverage",
        action="store_false",
        default=True,
        help="Skip coverage instrumentation and report.",
    )
    args = parser.parse_args()

    if args.list:
        print("Available test suites:")
        for name, cfg in SUITES.items():
            print(f"  {name:<12} extras=[{cfg['extras']}]  tests={cfg['test_dir']}")
        return

    selected = args.suites if args.suites else list(SUITES.keys())

    # Validate suite names
    invalid = [s for s in selected if s not in SUITES]
    if invalid:
        print(f"Unknown suite(s): {', '.join(invalid)}")
        print(f"Available: {', '.join(SUITES)}")
        sys.exit(1)

    VENVS_DIR.mkdir(exist_ok=True)

    results: dict[str, bool] = {}
    for suite_name in selected:
        results[suite_name] = run_suite(suite_name, args.recreate, args.coverage)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    all_passed = True
    for suite_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {suite_name}")
        if not passed:
            all_passed = False

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    """Run the main function."""
    main()
