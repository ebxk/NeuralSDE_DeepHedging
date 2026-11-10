import bootstrap
import unittest
from bottleneck_train import inputs_for

class FoldInputsTests(unittest.TestCase):
    def test_each_rolling_fold_uses_its_own_episode_split(self):
        for end in (2021,2022,2023):
            inputs=inputs_for(bootstrap.PROJECT,dict(end=end,kind='plain'),
                             bootstrap.PROJECT/'工作区/数据/处理后/瓶颈验证实验')
            self.assertGreater(len(inputs.paths[inputs.folds[0]]),0)
            self.assertGreater(len(inputs.paths[inputs.folds[1]]),0)
            self.assertGreater(len(inputs.paths[inputs.folds[2]]),0)
            self.assertEqual(inputs.episode_file.parent.name,str(end))

if __name__=='__main__':unittest.main()
