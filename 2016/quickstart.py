import math
import random
import sys

from model.ActionType import ActionType
from model.Faction import Faction
from model.Game import Game
from model.Move import Move
from model.Wizard import Wizard
from model.World import World

WAYPOINT_RADIUS = 100.00
LOW_HP_FACTOR = 0.25

class Point2D:
    """
    Вспомогательный класс для хранения позиций на карте.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_distance_to(self, x, y):
        return math.hypot(self.x - x, self.y - y)

    def get_distance_to_point(self, point):
        return self.get_distance_to(point.x, point.y)

    def get_distance_to_unit(self, unit):
        return self.get_distance_to(unit.x, unit.y)


def get_nearest_target(me: Wizard, world: World):
    """
    Находим ближайшую цель для атаки, независимо от её типа и других характеристик.
    """
    targets = []
    targets.extend(world.buildings)
    targets.extend(world.wizards)
    targets.extend(world.minions)

    nearest_target = None
    nearest_target_distance = sys.float_info.max

    for target in targets:
        # Нейтралов атакуем тоже если их хп меньше максимального - они стригеренны
        if (target.faction == me.faction or
                        target.faction == Faction.NEUTRAL and target.life < target.max_life):
            continue

        distance = me.get_distance_to_unit(target)
        if distance < nearest_target_distance:
            nearest_target = target
            nearest_target_distance = distance
    return nearest_target


def apply_go_to_move(point: Point2D, me: Wizard, game: Game, move: Move):
    """
    Простейший способ перемещения волшебника.
    """
    angle = me.get_angle_to(point.x, point.y)
    move.turn = angle

    if math.fabs(angle) < game.staff_sector / 4.0:
        move.speed = game.wizard_forward_speed


def get_next_waypoint(waypoints, me: Wizard):
    """
    Данный метод предполагает, что все ключевые точки на линии упорядочены по уменьшению дистанции до последней
    ключевой точки. Перебирая их по порядку, находим первую попавшуюся точку, которая находится ближе к последней
    точке на линии, чем волшебник. Это и будет следующей ключевой точкой.

    Дополнительно проверяем, не находится ли волшебник достаточно близко к какой-либо из ключевых точек. Если это
    так, то мы сразу возвращаем следующую ключевую точку.
    """
    last_waypoint = waypoints[-1]
    for i, waypoint in enumerate(waypoints[:-1]):
        if waypoint.get_distance_to_unit(me) <= WAYPOINT_RADIUS:
            return waypoints[i + 1]

        if last_waypoint.get_distance_to_point(waypoint) < last_waypoint.get_distance_to_unit(me):
            return waypoint

    return last_waypoint


def get_waypoints_by_id(id, game: Game):
    map_size = game.map_size
    if id in [1, 2, 6, 7]:
        # точки верхней линии
        return [
            Point2D(100.0, map_size - 100.0),
            Point2D(100.0, map_size - 400.0),
            Point2D(200.0, map_size - 800.0),
            Point2D(200.0, map_size * 0.75),
            Point2D(200.0, map_size * 0.5),
            Point2D(200.0, map_size * 0.25),
            Point2D(200.0, 200.0),
            Point2D(map_size * 0.25, 200.0),
            Point2D(map_size * 0.5, 200.0),
            Point2D(map_size * 0.75, 200.0),
            Point2D(map_size - 200.0, 200.0)
        ]
    elif id in [3, 8]:
        # точки средней линии
        return [
            Point2D(100.0, map_size - 100.0),
            random.choice([Point2D(600.0, map_size - 200.0), Point2D(200.0, map_size - 600.0)]),
            Point2D(800.0, map_size - 800.0),
            Point2D(map_size - 600.0, 600.0)
        ]
    else:
        # точки нижней линии
        return [
            Point2D(100.0, map_size - 100.0),
            Point2D(400.0, map_size - 100.0),
            Point2D(800.0, map_size - 200.0),
            Point2D(map_size * 0.25, map_size - 200.0),
            Point2D(map_size * 0.5, map_size - 200.0),
            Point2D(map_size * 0.75, map_size - 200.0),
            Point2D(map_size - 200.0, map_size - 200.0),
            Point2D(map_size - 200.0, map_size * 0.75),
            Point2D(map_size - 200.0, map_size * 0.5),
            Point2D(map_size - 200.0, map_size * 0.25),
            Point2D(map_size - 200.0, 200.0)
        ]


class MyStrategy:
    def __init__(self):
        super().__init__()
        self.initialized = False
        self.waypoints = None

    def initialize(self, me: Wizard, game: Game):
        random.seed(game.random_seed)
        self.waypoints = get_waypoints_by_id(me.id, game)
        self.initialized = True

    def move(self, me: Wizard, world: World, game: Game, move: Move):
        if not self.initialized:
            self.initialize(me, game)

        move.strafe_speed = random.choice([game.wizard_strafe_speed, -game.wizard_strafe_speed])

        if me.life < me.max_life * LOW_HP_FACTOR:
            apply_go_to_move(get_next_waypoint(self.waypoints[::-1], me))
            return

        nearest_target = get_nearest_target(me, world)
        if nearest_target is not None:
            distance = me.get_distance_to_unit(nearest_target)

            if distance <= me.cast_range:
                angle = me.get_angle_to_unit(nearest_target)
                move.turn = angle

                if math.fabs(angle) < game.staff_sector / 2.0:
                    move.action = ActionType.MAGIC_MISSILE
                    move.cast_angle = angle
                    move.min_cast_distance = distance - nearest_target.radius + game.magic_missile_radius
                    return

        apply_go_to_move(get_next_waypoint(self.waypoints, me), me, game, move)