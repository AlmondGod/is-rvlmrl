"""Pure geometry/configuration tests. No physics imports or simulation."""
import itertools
import unittest
from .design import Config,parts,beveled_vertices,manifest


class DesignTests(unittest.TestCase):
    def test_nominal_parts_do_not_overlap(self):
        p=parts(Config())
        for a,b in itertools.combinations(p.values(),2):
            overlaps=[min(a['target'][i]+a['size'][i]/2,b['target'][i]+b['size'][i]/2)-
                max(a['target'][i]-a['size'][i]/2,b['target'][i]-b['size'][i]/2) for i in range(3)]
            self.assertFalse(all(x>1e-10 for x in overlaps),overlaps)

    def test_panel_bottom_faces_meet_base_top(self):
        c=Config();p=parts(c)
        for name,item in p.items():
            if name!='bottom':self.assertAlmostEqual(item['target'][2]-item['size'][2]/2,c.thickness)

    def test_real_clearance_is_in_geometry(self):
        c=Config();p=parts(c)
        front_inner=p['front']['target'][1]+p['front']['size'][1]/2
        side_end=-p['left']['size'][1]/2
        self.assertAlmostEqual(side_end-front_inner,c.corner_gap)

    def test_bevel_vertices_stay_inside_measured_envelope(self):
        c=Config()
        for item in parts(c).values():
            v=beveled_vertices(item['size'],c.edge_bevel)
            self.assertEqual(len(v),24)
            for p in v:
                for i in range(3):self.assertLessEqual(abs(p[i]),item['size'][i]/2+1e-12)

    def test_no_fictional_measurements_or_finished_assembly(self):
        m=manifest(Config());self.assertEqual(m['measured_parameters'],[])
        self.assertFalse(m['full_assembly_success_available'])

    def test_invalid_geometry_is_rejected(self):
        for kw in [{'thickness':.12},{'edge_bevel':.005},{'corner_gap':-.001},{'timestep':.01}]:
            with self.assertRaises(ValueError):Config(**kw)


if __name__=='__main__':unittest.main()
