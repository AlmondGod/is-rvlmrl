import unittest
from analysis.retire_assembly_online import decision


class RetirementDecisionTests(unittest.TestCase):
    def setUp(self):
        self.keys=[f'v{i}' for i in range(8)]
        self.metadata=dict(optimizer_algorithm='grpo_visual_balanced_v1',steps=8,
                           splits=dict(validation=self.keys))
    def report(self,count):
        return dict(metrics=dict(n=8,successes=count,success_rate=count/8),
                    rows=[dict(layout_id=k,success=i<count) for i,k in enumerate(self.keys)])
    def test_only_retire_after_two_nonimproving_validations(self):
        self.assertTrue(decision(self.metadata,self.report(4),{2:self.report(3),4:self.report(4)}))
        self.assertFalse(decision(self.metadata,self.report(4),{2:self.report(3),4:self.report(5)}))
        self.assertFalse(decision(self.metadata,self.report(4),{2:self.report(6),4:self.report(4)}))
    def test_missing_validation_and_wrong_aggregates_are_errors(self):
        with self.assertRaises(ValueError):decision(self.metadata,self.report(4),{4:self.report(4)})
        wrong=self.report(5);wrong['metrics']=self.report(4)['metrics']
        with self.assertRaises(ValueError):decision(self.metadata,self.report(4),{2:self.report(3),4:wrong})
    def test_repeated_layout_and_wrong_baseline_are_errors(self):
        wrong=self.report(3);wrong['rows'][0]['layout_id']=self.keys[1]
        with self.assertRaises(ValueError):decision(self.metadata,self.report(4),{2:wrong,4:self.report(4)})
        with self.assertRaises(ValueError):decision(self.metadata,self.report(3),{2:self.report(3),4:self.report(4)})


if __name__=='__main__':unittest.main()
