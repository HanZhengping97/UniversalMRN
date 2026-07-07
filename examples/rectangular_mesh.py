"""Example: create a simple rectangular mesh."""

from src.mesh import MeshGenerator


if __name__ == "__main__":
    mesh = MeshGenerator.generate_rectangular_mesh(nx=4, ny=2, width=1.0, height=0.5)
    print(f"Generated {len(mesh.nodes)} nodes and {len(mesh.cells)} cells.")
