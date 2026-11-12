import unittest
import numpy as np
from study_core import registry, quote_mask, frozen_json, source_identity


class StudyTests(unittest.TestCase):
    def test_full_matrix_and_paired_initializations(self):
        jobs = registry()
        self.assertEqual(len(jobs), 432)
        self.assertEqual(len({j['key'] for j in jobs}), 432)
        self.assertEqual(len({j['initial_key'] for j in jobs}), 12)

    def test_duplicate_effective_strikes_share_holdout(self):
        mask = quote_mask(['2020-01-02']*3, ['2020-02-26']*3, [2.5, 2.5, 2.6])
        self.assertEqual(bool(mask[0]), bool(mask[1]))
        np.testing.assert_array_equal(mask, quote_mask(['2020-01-02']*3,
            ['2020-02-26']*3, [2.5, 2.5, 2.6]))

    def test_frozen_configuration_cannot_silently_change(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'config.json'
            frozen_json(p, {'epochs':100})
            frozen_json(p, {'epochs':100})
            with self.assertRaises(ValueError):
                frozen_json(p, {'epochs':1})

if __name__ == '__main__':
    unittest.main()
