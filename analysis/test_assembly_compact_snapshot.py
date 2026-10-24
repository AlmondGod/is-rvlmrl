import io
import pickle
import unittest

import mujoco
import numpy as np
from analysis import assembly_compact_snapshot as codec


class CompactSnapshotTests(unittest.TestCase):
    def test_native_bytes_aliases_and_continued_dynamics_match(self):
        model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint type="free"/><geom type="sphere" size=".1"/></body></worldbody></mujoco>')
        data=mujoco.MjData(model)
        for _ in range(10):mujoco.mj_step(model,data)
        stream=io.BytesIO();codec.dump([model,data,data],stream)
        self.assertLess(len(stream.getvalue()),len(pickle.dumps([model,data,data],protocol=5)))
        stream.seek(0);restored_model,restored_data,alias=codec.load(stream)
        self.assertIs(restored_data,alias)
        self.assertEqual(restored_model.__getstate__(),model.__getstate__())
        self.assertEqual(restored_data.__getstate__(),data.__getstate__())
        for _ in range(50):
            mujoco.mj_step(model,data);mujoco.mj_step(restored_model,restored_data)
            # Native serialization also contains wall-clock solver timers,
            # which legitimately differ after independently executing a step.
            self.assertEqual(restored_data.time,data.time)
            for name in ('qpos','qvel','ctrl','qacc','qacc_warmstart','xpos','xmat',
                         'qfrc_applied','xfrc_applied','actuator_force','sensordata'):
                np.testing.assert_array_equal(getattr(restored_data,name),getattr(data,name))
    def test_distinct_model_payloads_are_not_conflated(self):
        models=[mujoco.MjModel.from_xml_string(f'<mujoco><option gravity="0 0 {g}"/><worldbody/></mujoco>') for g in (-9.81,-2.)]
        stream=io.BytesIO();codec.dump(models,stream);stream.seek(0)
        restored=codec.load(stream)
        for before,after in zip(models,restored):self.assertEqual(before.__getstate__(),after.__getstate__())
        self.assertNotEqual(restored[0].opt.gravity[2],restored[1].opt.gravity[2])


if __name__=='__main__':unittest.main()
