"""Test cases for Fiber Photometry legacy job settings."""

import unittest
from pathlib import Path
from aind_metadata_extractor.fip_legacy.job_settings import JobSettings


class TestJobSettings(unittest.TestCase):
    """Test JobSettings model validation and functionality for legacy."""

    def test_minimal_required_fields(self):
        """Test creation with only required fields."""
        settings = JobSettings(
            subject_id="SUBJ001", rig_id="RIG01", iacuc_protocol="IACUC123", notes="Legacy session notes"
        )
        self.assertEqual(settings.subject_id, "SUBJ001")
        self.assertEqual(settings.rig_id, "RIG01")
        self.assertEqual(settings.iacuc_protocol, "IACUC123")
        self.assertEqual(settings.notes, "Legacy session notes")
        self.assertEqual(settings.job_settings_name, "FiberPhotometry")
        self.assertEqual(settings.session_type, "FIB")
        self.assertEqual(settings.experimenter_full_name, [])
        self.assertIsNone(settings.mouse_platform_name)
        self.assertFalse(settings.active_mouse_platform)
        self.assertIsNone(settings.anaesthesia)
        self.assertIsNone(settings.animal_weight_post)
        self.assertIsNone(settings.animal_weight_prior)
        self.assertEqual(settings.data_streams, [])
        self.assertEqual(settings.protocol_id, [])
        self.assertIsNone(settings.data_directory)
        self.assertIsNone(settings.output_directory)
        self.assertEqual(settings.output_filename, "session_fip.json")
        self.assertEqual(settings.local_timezone, "America/Los_Angeles")

    def test_optional_fields(self):
        """Test creation with optional fields."""
        settings = JobSettings(
            subject_id="SUBJ002",
            rig_id="RIG02",
            iacuc_protocol="IACUC456",
            notes="Optional notes",
            experimenter_full_name=["Alice", "Bob"],
            session_type="CUSTOM",
            mouse_platform_name="PlatformX",
            active_mouse_platform=True,
            anaesthesia="Isoflurane",
            animal_weight_post=25.5,
            animal_weight_prior=24.8,
            data_streams=[{"name": "stream1"}, {"name": "stream2"}],
            protocol_id=["P1", "P2"],
            data_directory=Path("/tmp/data"),
            output_directory=Path("/tmp/output"),
            output_filename="custom_output.json",
            local_timezone="UTC",
        )
        self.assertEqual(settings.experimenter_full_name, ["Alice", "Bob"])
        self.assertEqual(settings.session_type, "CUSTOM")
        self.assertEqual(settings.mouse_platform_name, "PlatformX")
        self.assertTrue(settings.active_mouse_platform)
        self.assertEqual(settings.anaesthesia, "Isoflurane")
        self.assertEqual(settings.animal_weight_post, 25.5)
        self.assertEqual(settings.animal_weight_prior, 24.8)
        self.assertEqual(settings.data_streams, [{"name": "stream1"}, {"name": "stream2"}])
        self.assertEqual(settings.protocol_id, ["P1", "P2"])
        self.assertEqual(settings.data_directory, Path("/tmp/data"))
        self.assertEqual(settings.output_directory, Path("/tmp/output"))
        self.assertEqual(settings.output_filename, "custom_output.json")
        self.assertEqual(settings.local_timezone, "UTC")

    def test_defaults(self):
        """Test default values for optional fields."""
        settings = JobSettings(subject_id="SUBJ003", rig_id="RIG03", iacuc_protocol="IACUC789", notes="Default test")
        self.assertEqual(settings.output_filename, "session_fip.json")
        self.assertEqual(settings.local_timezone, "America/Los_Angeles")
        self.assertIsNone(settings.output_directory)
        self.assertIsNone(settings.data_directory)
        self.assertEqual(settings.experimenter_full_name, [])
        self.assertEqual(settings.data_streams, [])
        self.assertEqual(settings.protocol_id, [])


if __name__ == "__main__":
    unittest.main()