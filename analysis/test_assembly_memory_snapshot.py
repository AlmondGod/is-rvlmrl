import unittest

import mujoco
import numpy as np

from analysis.assembly_memory_snapshot import clone


class MemorySnapshotTests(unittest.TestCase):
    def test_full_native_state_aliases_and_mutable_model_isolation(self):
        xml = '<mujoco><size nuserdata="3"/><worldbody><body pos="0 0 1"><freejoint/><geom name="g" type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>'
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model);other = mujoco.MjData(model)
        data.userdata[:] = [1, 2, 3];data.qfrc_applied[0] = .7
        for _ in range(20):mujoco.mj_step(model, data)
        value = dict(model=model, data=data, alias=data, other=other, nested=[model])
        restored, receipt = clone(value)
        self.assertEqual(receipt['models'], 1);self.assertEqual(receipt['data'], 2)
        self.assertIs(restored['data'], restored['alias'])
        self.assertIs(restored['data'].model, restored['model'])
        self.assertIs(restored['other'].model, restored['model'])
        self.assertIs(restored['nested'][0], restored['model'])
        self.assertIsNot(restored['model'], model)
        self.assertIsNot(restored['data'], data)
        self.assertEqual(restored['model'].__getstate__(), model.__getstate__())
        self.assertEqual(restored['data'].__getstate__(), data.__getstate__())
        for _ in range(25):
            mujoco.mj_step(model, data)
            mujoco.mj_step(restored['model'], restored['data'])
            for field in ('qpos','qvel','qacc','qacc_warmstart','qfrc_applied','userdata'):
                np.testing.assert_array_equal(getattr(data,field),getattr(restored['data'],field))
        model.geom_size[0,0] = .2
        self.assertEqual(restored['model'].geom_size[0,0], .1)
        restored['data'].userdata[0] = 99
        self.assertEqual(data.userdata[0], 1)

    def test_distinct_models_are_not_merged(self):
        first=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=".1"/></worldbody></mujoco>')
        second=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=".2"/></worldbody></mujoco>')
        result,receipt=clone([mujoco.MjData(first),mujoco.MjData(second),first,second])
        self.assertEqual(receipt['models'],2)
        self.assertIs(result[0].model,result[2]);self.assertIs(result[1].model,result[3])
        self.assertIsNot(result[2],result[3])
        np.testing.assert_array_equal(result[2].geom_size,first.geom_size)
        np.testing.assert_array_equal(result[3].geom_size,second.geom_size)


if __name__ == '__main__':unittest.main()
