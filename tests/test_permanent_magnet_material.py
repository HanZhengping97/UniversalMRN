from math import isclose
import pytest

from src.material import MU_0, LinearMagneticMaterial, LinearPermanentMagnetMaterial, MagnetizationDirection


def test_pm_material_coercive_field_calculated():
    pm = LinearPermanentMagnetMaterial(1, "ferrite", 1.05, 0.45)
    assert isclose(pm.coercive_field_strength, 0.45 / (MU_0 * 1.05))
    assert pm.is_permanent_magnet
    assert not LinearMagneticMaterial(0, "air", 1.0).is_permanent_magnet


def test_invalid_pm_material_values_rejected():
    with pytest.raises(ValueError):
        LinearPermanentMagnetMaterial(1, "bad", 1.0, 0.0)
    with pytest.raises(ValueError):
        LinearPermanentMagnetMaterial(1, "bad", 0.0, 0.1)


def test_magnetization_vector_normalization_and_zero_rejected():
    d = MagnetizationDirection(3.0, 4.0)
    assert d.unit_vector == (0.6, 0.8)
    with pytest.raises(ValueError):
        MagnetizationDirection(0.0, 0.0)


def test_ferrite_expected_coercive_field():
    pm = LinearPermanentMagnetMaterial(2, "ferrite", 1.05, 0.45)
    assert isclose(pm.coercive_field_strength, 341046.306, rel_tol=1e-6)
