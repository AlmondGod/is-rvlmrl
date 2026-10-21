"""Offline constitutive checks on prescribed inputs; no robot/rigid-body rollout."""
from dataclasses import replace
import math
import unittest
import numpy as np
from .design import Config,parts
from .fastening import Parameters,Frame,Connector,JointProcess,box_joint,cure_fraction,driving_increment


def setup(condition='combined',p=None):
    c=Config();s=box_joint(c);spec=parts(c)
    a=Frame(np.array(spec[s.a]['target']),np.eye(3));b=Frame(np.array(spec[s.b]['target']),np.eye(3))
    return JointProcess(s,p,condition),a,b


def tick(j,a,b,dt=.001,**kw):
    args={'nailer_position':a.world(j.s.entry_a),'nailer_axis':a.rotation@j.s.normal_a,'nose_force_N':4.}
    args.update(kw)
    return j.step(dt,a,b,**args)


def drive(j,a,b,dt=.001):
    loads=[]
    for _ in range(round(.5/dt)):loads=tick(j,a,b,dt,trigger=True)
    return loads


def coat(j,a,b):
    # Explicit prescribed nozzle poses; this is a constitutive fixture, NOT robot success.
    for pa in j.s.patches_a:
        tick(j,a,b,.11,dispense=True,glue_position=a.world(pa)+np.array([0,.001,0]),glue_axis=np.array([0,-1,0]))


class FasteningTests(unittest.TestCase):
    def test_units_and_invalid_parameters(self):
        for kw in ({'nail_length_m':float('nan')},{'drive_force_limit_N':-1},{'glue_handling_fraction':2}):
            with self.assertRaises(ValueError):Parameters(**kw)

    def test_no_automatic_fastening(self):
        j,a,b=setup();tick(j,a,b,1000)
        self.assertIsNone(j.nail);self.assertTrue(all(x.connector is None for x in j.glue))
        self.assertFalse(j.evaluator_metrics()['model_handling_ready'])

    def test_trigger_contact_alignment_and_receiver_gates(self):
        for overrides in ({'nose_force_N':0.},{'nailer_axis':np.array([1,0,0])},{'nailer_position':np.array([1.,1.,1.])}):
            j,a,b=setup();tick(j,a,b,trigger=True,**overrides);self.assertEqual(j.depth,0.)
        j,a,b=setup();b.position+=np.array([0,.01,0]);tick(j,a,b,trigger=True);self.assertEqual(j.depth,0.)

    def test_drive_law_independent_closed_form_and_work(self):
        p=Parameters();force=100.;dt=.01
        dx,active,work,penetration=driving_increment(0.,force,dt,p,10.)
        expected=(force-p.drive_intercept_N)/p.drive_slope_N_m*(1-math.exp(-p.drive_slope_N_m*dt/p.drive_drag_N_s_m))
        self.assertAlmostEqual(dx,expected,places=14);self.assertAlmostEqual(active,dt)
        self.assertGreaterEqual(work,penetration);self.assertAlmostEqual(work,force*dx)

    def test_energy_cap_and_stall(self):
        p=Parameters();dx,_,work,_=driving_increment(0.,110.,1.,p,.1)
        self.assertLessEqual(work,.1+1e-12);self.assertAlmostEqual(dx,.1/110.)
        self.assertEqual(driving_increment(0.,1.,1.,p,4.),(0.,0.,0.,0.))

    def test_drive_reaction_force_and_moment_balance(self):
        j,a,b=setup('nails_only');loads=tick(j,a,b,trigger=True)
        np.testing.assert_allclose(sum((x[1] for x in loads),np.zeros(3)),0,atol=1e-12)
        np.testing.assert_allclose(sum((np.cross(x[2],x[1]) for x in loads),np.zeros(3)),0,atol=1e-12)
        self.assertTrue(any(name=='nailer' and np.linalg.norm(f)>0 for name,f,_ in loads))

    def test_no_connection_before_receiving_member(self):
        j,a,b=setup('nails_only');tick(j,a,b,trigger=True)
        self.assertLess(j.depth,j.s.entry_thickness_m);self.assertIsNone(j.nail)

    def test_drive_seating_and_work_ledger(self):
        j,a,b=setup('nails_only');drive(j,a,b)
        self.assertAlmostEqual(j.depth,j.p.nail_length_m)
        self.assertTrue(j.evaluator_metrics()['nail_seated'])
        self.assertAlmostEqual(j.drive_work,j.penetration_work+j.drive_viscous_work,places=12)
        self.assertFalse(j.evaluator_metrics()['physical_acceptance'])

    def test_lost_contact_requires_new_trigger(self):
        j,a,b=setup('nails_only');tick(j,a,b,trigger=True);depth=j.depth
        tick(j,a,b,trigger=True,nose_force_N=0);tick(j,a,b,trigger=True)
        self.assertEqual(j.depth,depth)
        tick(j,a,b,trigger=False);tick(j,a,b,trigger=True);self.assertGreater(j.depth,depth)

    def test_failure_is_irreversible_and_no_retrigger_repair(self):
        j,a,b=setup('nails_only');drive(j,a,b);original=b.position.copy()
        b.position+=np.array([0,.02,0]);tick(j,a,b)
        self.assertTrue(j.nail_failed)
        b.position=original;tick(j,a,b,trigger=True)
        self.assertTrue(j.nail_failed);self.assertFalse(j.evaluator_metrics()['nail_seated'])
        self.assertEqual(j.last_gate,'failed_nail_requires_replacement')

    def test_nail_lateral_failure(self):
        j,a,b=setup('nails_only');drive(j,a,b);b.position+=np.array([.02,0,0]);tick(j,a,b)
        self.assertTrue(j.nail_failed)

    def test_force_energy_and_objectivity(self):
        a=Frame(np.zeros(3),np.eye(3));b=Frame(np.zeros(3),np.eye(3));c=Connector.at(a,b,np.zeros(3))
        b.position=np.array([.001,0,0]);b.linear=np.array([.01,.02,0])
        force,point,U,loss=c.evaluate(a,b,np.array([1.,0,0]),1000.,2.,100.,100.,.1,.001)
        np.testing.assert_allclose(force,[1.02,.04,0]);self.assertAlmostEqual(U,.0005)
        self.assertGreaterEqual(loss,0.)
        # Same common point for opposite forces => zero total force and moment.
        np.testing.assert_allclose(np.cross(point,force)+np.cross(point,-force),0.)
        # Rigidly rotate and translate both frames: no additional strain is created.
        R=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
        a.position=R@a.position+3;a.rotation=R;b.position=R@b.position+3;b.rotation=R;b.linear=R@b.linear
        rotated,_,U2,_=c.evaluate(a,b,R@np.array([1.,0,0]),1000.,2.,100.,100.,.1,.001)
        np.testing.assert_allclose(rotated,R@force,atol=1e-11);self.assertAlmostEqual(U,U2)

    def test_glue_requires_application_coverage_alignment_and_clamping(self):
        j,a,b=setup('glue_only');coat(j,a,b)
        self.assertTrue(all(x.volume_m3>0 for x in j.glue))
        tick(j,a,b,61,patch_clamp_N=np.zeros(4));self.assertTrue(all(x.connector is None for x in j.glue))
        b.position+=np.array([.02,0,0]);tick(j,a,b,61,patch_clamp_N=np.ones(4))
        self.assertTrue(all(x.connector is None for x in j.glue))
        b.position-=np.array([.02,0,0]);tick(j,a,b,61,patch_clamp_N=np.ones(4))
        self.assertTrue(all(x.connector is not None for x in j.glue))
        self.assertFalse(j.evaluator_metrics()['model_handling_ready'])

    def test_clamp_interruption_resets_hold(self):
        j,a,b=setup('glue_only');coat(j,a,b)
        tick(j,a,b,30,patch_clamp_N=np.ones(4));tick(j,a,b,.01)
        tick(j,a,b,31,patch_clamp_N=np.ones(4));self.assertTrue(all(x.connector is None for x in j.glue))

    def test_glue_cure_failure_and_no_healing(self):
        j,a,b=setup('glue_only');coat(j,a,b);tick(j,a,b,61,patch_clamp_N=np.ones(4));tick(j,a,b,86400)
        self.assertTrue(j.evaluator_metrics()['model_handling_ready'])
        original=b.position.copy();b.position+=np.array([0,.01,0]);tick(j,a,b)
        self.assertTrue(all(x.connector.failed for x in j.glue));self.assertFalse(j.evaluator_metrics()['model_handling_ready'])
        b.position=original;tick(j,a,b,86400,patch_clamp_N=np.ones(4))
        self.assertTrue(all(x.connector.failed for x in j.glue))

    def test_expired_glue_cannot_be_refreshed_by_dispensing(self):
        j,a,b=setup('glue_only');coat(j,a,b);tick(j,a,b,301)
        coat(j,a,b);tick(j,a,b,61,patch_clamp_N=np.ones(4))
        self.assertTrue(all(x.expired and x.connector is None for x in j.glue))

    def test_missing_application_and_wrong_nozzle_do_not_bond(self):
        j,a,b=setup('glue_only');tick(j,a,b,61,patch_clamp_N=np.ones(4));self.assertTrue(all(x.connector is None for x in j.glue))
        tick(j,a,b,1,dispense=True,glue_position=a.world(j.s.patches_a[0])+[0,.001,0],glue_axis=np.array([0,1.,0]))
        self.assertTrue(all(x.volume_m3==0 for x in j.glue));self.assertGreater(j.wasted_volume,0)

    def test_cure_clock_explicit_and_clamp_clock_not_accelerated(self):
        j,a,b=setup('glue_only',replace(Parameters(),chemistry_time_scale=10.));tick(j,a,b,2)
        self.assertEqual(j.clock_s,2.);self.assertEqual(j.chemical_clock_s,20.)
        self.assertAlmostEqual(cure_fraction(86400,86400),.95)

    def test_ablation_conditions(self):
        for condition in ('unfastened','nails_only','glue_only','combined'):
            j,a,b=setup(condition);drive(j,a,b);coat(j,a,b);tick(j,a,b,61,patch_clamp_N=np.ones(4))
            self.assertEqual(j.nail is not None,condition in ('nails_only','combined'))
            self.assertEqual(any(x.connector is not None for x in j.glue),condition in ('glue_only','combined'))

    def test_sensor_boundary(self):
        j,a,b=setup();drive(j,a,b)
        self.assertEqual(set(j.sensor_packet()),{'nose_contact_switch','driver_stroke_m','driver_force_N','dispensed_volume_m3'})
        self.assertFalse(set(j.sensor_packet()) & {'nail_failed','model_handling_ready','glue','cure_fraction','last_drive_gate'})

    def test_timestep_convergence_prescribed_drive(self):
        rows=[]
        for dt in (.001,.0005,.00025):
            j,a,b=setup('nails_only');drive(j,a,b,dt);rows.append((j.depth,j.drive_work,j.penetration_work))
        np.testing.assert_allclose(rows,[rows[-1]]*3,rtol=1e-10,atol=1e-12)


if __name__=='__main__':unittest.main()
