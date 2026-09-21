"""Standard-library wrappers for the pytest-style numerical tests."""

import unittest

from tests import test_reference_records, test_transport


class TransportTests(unittest.TestCase):
    def test_polar_transport(self) -> None:
        test_transport.test_polar_transport_is_unitary_and_reports_fidelity()

    def test_root_assignment_false_negative(self) -> None:
        test_transport.test_local_assignment_can_miss_continuous_route_difference()

    def test_gauge_invariance(self) -> None:
        test_transport.test_plaquette_scalars_and_back_transport_are_gauge_invariant()

    def test_nonclosure_bound(self) -> None:
        test_transport.test_normalized_nonclosure_is_bounded()


class ReferenceRecordTests(unittest.TestCase):
    def test_lih(self) -> None:
        test_reference_records.test_lih_reference_record()

    def test_beh2(self) -> None:
        test_reference_records.test_beh2_reference_record()

    def test_cyclopentadienyl_cation(self) -> None:
        test_reference_records.test_cyclopentadienyl_reference_record()


if __name__ == "__main__":
    unittest.main()
