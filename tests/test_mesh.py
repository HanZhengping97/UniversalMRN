"""Tests for foundational mesh data structures."""

import pytest

from src.mesh import Cell, Mesh, MeshGenerator, Node


def test_mesh_add_and_get_entities() -> None:
    mesh = Mesh()
    node = Node(id=1, x=0.0, y=1.0)
    cell = Cell(id=2, center_x=0.5, center_y=0.5, dx=1.0, dy=1.0, material_id=3)

    mesh.add_node(node)
    mesh.add_cell(cell)

    assert mesh.get_node(1) == node
    assert mesh.get_cell(2) == cell


def test_mesh_rejects_duplicate_ids() -> None:
    mesh = Mesh()
    mesh.add_node(Node(id=1, x=0.0, y=0.0))
    mesh.add_cell(Cell(id=1, center_x=0.5, center_y=0.5, dx=1.0, dy=1.0, material_id=0))

    with pytest.raises(ValueError):
        mesh.add_node(Node(id=1, x=1.0, y=1.0))

    with pytest.raises(ValueError):
        mesh.add_cell(Cell(id=1, center_x=1.5, center_y=1.5, dx=1.0, dy=1.0, material_id=0))


def test_generate_rectangular_mesh_counts_and_geometry() -> None:
    mesh = MeshGenerator.generate_rectangular_mesh(nx=2, ny=3, width=4.0, height=6.0)

    assert len(mesh.nodes) == 12
    assert len(mesh.cells) == 6
    assert mesh.get_node(0) == Node(id=0, x=0.0, y=0.0)
    assert mesh.get_node(11) == Node(id=11, x=4.0, y=6.0)
    assert mesh.get_cell(0) == Cell(id=0, center_x=1.0, center_y=1.0, dx=2.0, dy=2.0, material_id=0)
    assert mesh.get_cell(5) == Cell(id=5, center_x=3.0, center_y=5.0, dx=2.0, dy=2.0, material_id=0)


@pytest.mark.parametrize(
    ("nx", "ny", "width", "height"),
    [(0, 1, 1.0, 1.0), (1, 0, 1.0, 1.0), (1, 1, 0.0, 1.0), (1, 1, 1.0, 0.0)],
)
def test_generate_rectangular_mesh_rejects_invalid_dimensions(
    nx: int, ny: int, width: float, height: float
) -> None:
    with pytest.raises(ValueError):
        MeshGenerator.generate_rectangular_mesh(nx=nx, ny=ny, width=width, height=height)
