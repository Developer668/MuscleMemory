"""Deterministic A* teacher used only while generating training worlds."""

import heapq
from collections.abc import Iterable

from muscle_memory.worlds.generation._geometry import Aabb, object_aabb
from muscle_memory.worlds.models import TrainingWorld, Vec2
from muscle_memory.worlds.rules import WorldRules

GridCell = tuple[int, int]


def _cells(width_m: float, depth_m: float, resolution_m: float) -> tuple[int, int]:
    return round(width_m / resolution_m), round(depth_m / resolution_m)


def _point_for_cell(cell: GridCell, resolution_m: float) -> Vec2:
    return Vec2(
        x=round((cell[0] + 0.5) * resolution_m, 6),
        y=round((cell[1] + 0.5) * resolution_m, 6),
    )


def _cell_for_point(point: Vec2, resolution_m: float) -> GridCell:
    return int(point.x / resolution_m), int(point.y / resolution_m)


def _blocked_bounds(world: TrainingWorld, rules: WorldRules) -> tuple[Aabb, ...]:
    inflation = rules.robot_radius_m + rules.minimum_clearance_m
    return tuple(object_aabb(obstacle).inflated(inflation) for obstacle in world.objects)


def _traversable(
    cell: GridCell,
    *,
    columns: int,
    rows: int,
    resolution_m: float,
    wall_margin_m: float,
    blocked: tuple[Aabb, ...],
    width_m: float,
    depth_m: float,
) -> bool:
    if not (0 <= cell[0] < columns and 0 <= cell[1] < rows):
        return False
    point = _point_for_cell(cell, resolution_m)
    if not (
        wall_margin_m <= point.x <= width_m - wall_margin_m
        and wall_margin_m <= point.y <= depth_m - wall_margin_m
    ):
        return False
    return not any(bounds.contains(point) for bounds in blocked)


def _neighbours(cell: GridCell) -> Iterable[GridCell]:
    x, y = cell
    yield x + 1, y
    yield x, y + 1
    yield x - 1, y
    yield x, y - 1


def _manhattan(left: GridCell, right: GridCell) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def find_baseline_path(world: TrainingWorld, rules: WorldRules) -> tuple[Vec2, ...] | None:
    """Find a clearance-aware deterministic A* path, or return ``None``."""
    resolution = rules.grid_resolution_m
    columns, rows = _cells(world.template.width_m, world.template.depth_m, resolution)
    start = _cell_for_point(world.start, resolution)
    destination = _cell_for_point(world.destination, resolution)
    blocked = _blocked_bounds(world, rules)
    wall_margin = rules.robot_radius_m + rules.minimum_clearance_m

    def traversable(cell: GridCell) -> bool:
        return _traversable(
            cell,
            columns=columns,
            rows=rows,
            resolution_m=resolution,
            wall_margin_m=wall_margin,
            blocked=blocked,
            width_m=world.template.width_m,
            depth_m=world.template.depth_m,
        )

    if not traversable(start) or not traversable(destination):
        return None

    frontier: list[tuple[int, int, int, int, GridCell]] = []
    heapq.heappush(frontier, (_manhattan(start, destination), 0, start[0], start[1], start))
    came_from: dict[GridCell, GridCell | None] = {start: None}
    best_cost: dict[GridCell, int] = {start: 0}

    while frontier:
        _, cost, _, _, current = heapq.heappop(frontier)
        if current == destination:
            break
        if cost != best_cost[current]:
            continue
        for neighbour in _neighbours(current):
            if not traversable(neighbour):
                continue
            next_cost = cost + 1
            if next_cost >= best_cost.get(neighbour, next_cost + 1):
                continue
            best_cost[neighbour] = next_cost
            came_from[neighbour] = current
            priority = next_cost + _manhattan(neighbour, destination)
            heapq.heappush(
                frontier,
                (priority, next_cost, neighbour[0], neighbour[1], neighbour),
            )

    if destination not in came_from:
        return None

    cells: list[GridCell] = []
    cursor: GridCell | None = destination
    while cursor is not None:
        cells.append(cursor)
        cursor = came_from[cursor]
    cells.reverse()

    grid_points = tuple(_point_for_cell(cell, resolution) for cell in cells)
    return (world.start, *grid_points, world.destination)
