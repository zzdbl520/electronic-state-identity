from scripts.verify_reference_results import (
    verify_beh2,
    verify_cyclopentadienyl_cation,
    verify_lih,
)


def test_lih_reference_record() -> None:
    verify_lih()


def test_beh2_reference_record() -> None:
    verify_beh2()


def test_cyclopentadienyl_reference_record() -> None:
    verify_cyclopentadienyl_cation()
