"""Explicit task registry, shared by training and dependency-free replay audits."""
from . import task, task_ordered


def get_task(name='pick_place'):
    if name == 'robot_lift':
        from . import task_robot
        return task_robot
    if name == 'mujoco_docking':
        from . import task_mujoco
        return task_mujoco
    return {'pick_place': task, 'ordered_two': task_ordered}[name]
