import math, pytest
from src.machine.dssr_afpm_phi_z import default_dssr_afpm_phi_z_config

def test_default_configuration_validates(): assert default_dssr_afpm_phi_z_config().pole_count == 44
def test_outer_diameter_exceeds_inner():
    with pytest.raises(ValueError): default_dssr_afpm_phi_z_config(outer_diameter=.1, inner_diameter=.2)
def test_even_pole_count():
    with pytest.raises(ValueError): default_dssr_afpm_phi_z_config(pole_count=45)
def test_invalid_slot_opening_ratio():
    with pytest.raises(ValueError): default_dssr_afpm_phi_z_config(slot_opening_ratio=1)
def test_magnet_width_larger_than_pole_pitch_rejected():
    c=default_dssr_afpm_phi_z_config();
    with pytest.raises(ValueError): default_dssr_afpm_phi_z_config(magnet_circumferential_width=c.mean_radius*c.pole_pitch_angle*1.1)
