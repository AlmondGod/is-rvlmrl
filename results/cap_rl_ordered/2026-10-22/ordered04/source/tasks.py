"""Explicit task registry, shared by training and dependency-free replay audits."""
from . import task, task_ordered


def get_task(name='pick_place'):
    return {'pick_place': task, 'ordered_two': task_ordered}[name]
