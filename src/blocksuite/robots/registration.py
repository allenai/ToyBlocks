from __future__ import annotations
from dataclasses import dataclass
from typing import Type, TypeVar, TYPE_CHECKING
import logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from blocksuite.robots.base_robot import BaseRobot, BaseRobotCfg
    T = TypeVar("T", bound=BaseRobot)

@dataclass
class RobotSpec:
    robot_class: Type[BaseRobot]
    """the class for the robot"""
    cfg_dataclass: Type[BaseRobotCfg]
    """the configuration dataclass for the robot"""

REGISTERED_ROBOTS: dict[str, RobotSpec] = dict()

def register_robot(override=False):
    """A decorator to register robots so they can be used easily by string uid.

    Args:
        uid (str): unique id of the agent.
        override (bool): whether to override the agent if it is already registered.
    """

    def _register_agent(robot_cls: Type[T]) -> Type[T]:
        if robot_cls.id in REGISTERED_ROBOTS:
            if override:
                logger.warning(f"Overriding registered robot {robot_cls.id}")
                REGISTERED_ROBOTS.pop(robot_cls.id)
            else:
                logger.warning(
                    f"Robot {robot_cls.id} is already registered. Skip registration."
                )
                return robot_cls
        REGISTERED_ROBOTS[robot_cls.id] = RobotSpec(
            robot_class=robot_cls, cfg_dataclass=robot_cls.__annotations__["cfg"]
        )
        return robot_cls

    return _register_agent