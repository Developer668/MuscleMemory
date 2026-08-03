"""Clearance-aware path teacher available only while producing demonstrations."""

from __future__ import annotations

import heapq
import math
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise

from muscle_memory.robot.command import TaskCommand
from muscle_memory.worlds.generation._geometry import Aabb, object_aabb
from muscle_memory.worlds.models import TrainingWorld, Vec2
from muscle_memory.worlds.rules import WorldRules, load_world_rules

EXPERT_OBSTACLE_CLEARANCE_M = 0.85
EXPERT_MAXIMUM_FORWARD_SPEED_MPS = 0.3
EXPERT_MAXIMUM_TURN_RATE_RAD_S = 0.5
EXPERT_WAYPOINT_RADIUS_M = 0.35
EXPERT_STOP_REQUEST_DISTANCE_M = 0.43
EXPERT_TURN_IN_PLACE_THRESHOLD_RAD = 0.22

GridCell = tuple[int, int]


@dataclass(frozen=True, slots=True)
class ExpertPath:
    """A training-only path and the stricter clearance used to produce it."""

    waypoints: tuple[Vec2, ...]
    obstacle_clearance_m: float
    length_m: float


def _point_for_cell(cell: GridCell, resolution_m: float) -> Vec2:
    return Vec2(
        x=round((cell[0] + 0.5) * resolution_m, 6),
        y=round((cell[1] + 0.5) * resolution_m, 6),
    )


def _cell_for_point(point: Vec2, resolution_m: float) -> GridCell:
    return int(point.x / resolution_m), int(point.y / resolution_m)


def _neighbours(cell: GridCell) -> Iterable[tuple[GridCell, float]]:
    x, y = cell
    for delta_x, delta_y in (
        (1, 0),
        (0, 1),
        (-1, 0),
        (0, -1),
        (1, 1),
        (-1, 1),
        (-1, -1),
        (1, -1),
    ):
        step_cost = math.sqrt(2.0) if delta_x and delta_y else 1.0
        yield (x + delta_x, y + delta_y), step_cost


def _inside_any(point: Vec2, bounds: tuple[Aabb, ...]) -> bool:
    return any(item.contains(point) for item in bounds)


def _segment_is_clear(
    start: Vec2,
    end: Vec2,
    *,
    blocked: tuple[Aabb, ...],
    wall_margin_m: float,
    world: TrainingWorld,
    sample_spacing_m: float,
) -> bool:
    distance = math.dist((start.x, start.y), (end.x, end.y))
    sample_count = max(2, math.ceil(distance / sample_spacing_m))
    for index in range(sample_count + 1):
        amount = index / sample_count
        point = Vec2(
            x=start.x + amount * (end.x - start.x),
            y=start.y + amount * (end.y - start.y),
        )
        if not (
            wall_margin_m <= point.x <= world.template.width_m - wall_margin_m
            and wall_margin_m <= point.y <= world.template.depth_m - wall_margin_m
        ):
            return False
        if _inside_any(point, blocked):
            return False
    return True


def _simplify_path(
    points: tuple[Vec2, ...],
    *,
    blocked: tuple[Aabb, ...],
    wall_margin_m: float,
    world: TrainingWorld,
    sample_spacing_m: float,
) -> tuple[Vec2, ...]:
    simplified = [points[0]]
    current = 0
    while current < len(points) - 1:
        target = len(points) - 1
        while target > current + 1 and not _segment_is_clear(
            points[current],
            points[target],
            blocked=blocked,
            wall_margin_m=wall_margin_m,
            world=world,
            sample_spacing_m=sample_spacing_m,
        ):
            target -= 1
        simplified.append(points[target])
        current = target
    return tuple(simplified)


def plan_expert_path(
    world: TrainingWorld,
    rules: WorldRules | None = None,
    *,
    obstacle_clearance_m: float = EXPERT_OBSTACLE_CLEARANCE_M,
) -> ExpertPath | None:
    """Create a deterministic eight-connected training path with extra clearance."""
    active_rules = rules or load_world_rules()
    if obstacle_clearance_m < active_rules.minimum_clearance_m:
        raise ValueError("expert clearance cannot be lower than the success threshold")
    resolution = float(active_rules.grid_resolution_m)
    columns = round(float(world.template.width_m) / resolution)
    rows = round(float(world.template.depth_m) / resolution)
    wall_margin = float(active_rules.robot_radius_m + active_rules.minimum_clearance_m)
    obstacle_inflation = float(active_rules.robot_radius_m + obstacle_clearance_m)
    blocked = tuple(
        object_aabb(obstacle).inflated(obstacle_inflation)
        for obstacle in world.objects
    )

    def traversable(cell: GridCell) -> bool:
        if not (0 <= cell[0] < columns and 0 <= cell[1] < rows):
            return False
        point = _point_for_cell(cell, resolution)
        return (
            wall_margin <= point.x <= float(world.template.width_m) - wall_margin
            and wall_margin <= point.y <= float(world.template.depth_m) - wall_margin
            and not _inside_any(point, blocked)
        )

    start = _cell_for_point(world.start, resolution)
    destination = _cell_for_point(world.destination, resolution)
    if not traversable(start) or not traversable(destination):
        return None
    frontier: list[tuple[float, float, int, int, GridCell]] = []
    heapq.heappush(frontier, (0.0, 0.0, start[0], start[1], start))
    came_from: dict[GridCell, GridCell | None] = {start: None}
    best_cost: dict[GridCell, float] = {start: 0.0}
    while frontier:
        _, cost, _, _, current = heapq.heappop(frontier)
        if current == destination:
            break
        if not math.isclose(cost, best_cost[current], abs_tol=1e-12):
            continue
        for neighbour, step_cost in _neighbours(current):
            if not traversable(neighbour):
                continue
            delta_x = neighbour[0] - current[0]
            delta_y = neighbour[1] - current[1]
            if delta_x and delta_y and (
                not traversable((current[0] + delta_x, current[1]))
                or not traversable((current[0], current[1] + delta_y))
            ):
                continue
            next_cost = cost + step_cost
            if next_cost >= best_cost.get(neighbour, math.inf):
                continue
            best_cost[neighbour] = next_cost
            came_from[neighbour] = current
            heuristic = math.hypot(
                destination[0] - neighbour[0],
                destination[1] - neighbour[1],
            )
            heapq.heappush(
                frontier,
                (next_cost + heuristic, next_cost, neighbour[0], neighbour[1], neighbour),
            )
    if destination not in came_from:
        return None

    cells: list[GridCell] = []
    cursor: GridCell | None = destination
    while cursor is not None:
        cells.append(cursor)
        cursor = came_from[cursor]
    cells.reverse()
    dense = (world.start, *(_point_for_cell(cell, resolution) for cell in cells), world.destination)
    waypoints = _simplify_path(
        dense,
        blocked=blocked,
        wall_margin_m=wall_margin,
        world=world,
        sample_spacing_m=resolution / 4.0,
    )
    length = sum(
        math.dist((left.x, left.y), (right.x, right.y))
        for left, right in pairwise(waypoints)
    )
    return ExpertPath(
        waypoints=waypoints,
        obstacle_clearance_m=obstacle_clearance_m,
        length_m=length,
    )


def _wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


class ExpertNavigator:
    """Convert a teacher path into the same three outputs used by learned policies."""

    def __init__(self, path: ExpertPath) -> None:
        if len(path.waypoints) < 2:
            raise ValueError("expert path needs at least two waypoints")
        self.path = path
        self.waypoint_index = 1
        self._stop_requested = False

    def command(self, position: Vec2, yaw_radians: float) -> TaskCommand:
        destination = self.path.waypoints[-1]
        while self.waypoint_index < len(self.path.waypoints) - 1:
            waypoint = self.path.waypoints[self.waypoint_index]
            if math.dist((position.x, position.y), (waypoint.x, waypoint.y)) >= (
                EXPERT_WAYPOINT_RADIUS_M
            ):
                break
            self.waypoint_index += 1

        distance_to_destination = math.dist(
            (position.x, position.y),
            (destination.x, destination.y),
        )
        self._stop_requested = (
            self._stop_requested
            or distance_to_destination <= EXPERT_STOP_REQUEST_DISTANCE_M
        )
        if self._stop_requested:
            return TaskCommand(0.0, 0.0, 1.0)
        target = self.path.waypoints[self.waypoint_index]
        desired_yaw = math.atan2(target.y - position.y, target.x - position.x)
        heading_error = _wrap_angle(desired_yaw - yaw_radians)
        turning_rate = max(
            -EXPERT_MAXIMUM_TURN_RATE_RAD_S,
            min(EXPERT_MAXIMUM_TURN_RATE_RAD_S, 1.4 * heading_error),
        )
        if abs(heading_error) > EXPERT_TURN_IN_PLACE_THRESHOLD_RAD:
            return TaskCommand(0.0, turning_rate, 0.0)
        forward_speed = EXPERT_MAXIMUM_FORWARD_SPEED_MPS
        if distance_to_destination < 0.8:
            forward_speed = min(forward_speed, 0.18)
        return TaskCommand(forward_speed, turning_rate, 0.0)
