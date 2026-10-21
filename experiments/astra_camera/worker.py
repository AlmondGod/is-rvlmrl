"""Controller process receives only configuration and the observation allowlist."""
import traceback

OBSERVATION_KEYS = frozenset({
    'time','image_time','rgb','cameras','wrist_position','wrist_quaternion',
    'finger_q','finger_qvel','wrist_target','wrist_target_quaternion',
    'finger_targets','joint_limits',
})


def serve(connection, config):
    # No simulator instance or evaluator result is passed to this process.
    try:
        from .controller import Controller
        controller = Controller(config)
        controller.reset()
        connection.send({'ready': True})
        while True:
            message = connection.recv()
            if message is None:
                break
            if set(message) != OBSERVATION_KEYS:
                raise ValueError('Observation must match the camera/proprio allowlist exactly')
            action = controller.command(message)
            connection.send({'action': action, 'status': getattr(controller, 'status', {})})
    except BaseException:
        connection.send({'error': traceback.format_exc()})
    finally:
        connection.close()
