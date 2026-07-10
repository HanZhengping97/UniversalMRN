"""Connectivity checks for mesh branch graphs."""

from __future__ import annotations

from collections import defaultdict, deque

from src.mesh import Mesh


def find_connected_components(mesh: Mesh) -> tuple[tuple[int, ...], ...]:
    """Return deterministic connected components of the undirected branch graph."""

    adjacency: dict[int, set[int]] = defaultdict(set)
    for node_id in mesh.nodes:
        adjacency[node_id]
    for branch in mesh.branches.values():
        if branch.start_node_id not in mesh.nodes:
            raise ValueError(f"branch {branch.id} references missing start node {branch.start_node_id}.")
        if branch.end_node_id not in mesh.nodes:
            raise ValueError(f"branch {branch.id} references missing end node {branch.end_node_id}.")
        adjacency[branch.start_node_id].add(branch.end_node_id)
        adjacency[branch.end_node_id].add(branch.start_node_id)

    seen: set[int] = set()
    components: list[tuple[int, ...]] = []
    for node_id in sorted(mesh.nodes):
        if node_id in seen:
            continue
        queue: deque[int] = deque([node_id])
        seen.add(node_id)
        component: list[int] = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        components.append(tuple(sorted(component)))
    return tuple(components)


def validate_connected_mesh(mesh: Mesh) -> None:
    """Raise ``ValueError`` unless ``mesh`` has exactly one connected component."""

    components = find_connected_components(mesh)
    if len(components) != 1:
        sizes = tuple(len(component) for component in components)
        raise ValueError(f"mesh is disconnected: found {len(components)} connected components with sizes {sizes}.")
