import unittest
import torch
import torchvision.models as models

class Tester(unittest.TestCase):
    def test_forward_shapes(self):
        model = models.DiffusionDepthFlowEstimator(num_steps=2)
        img = torch.randn(1, 3, 32, 32)
        depth, flow = model(img)
        self.assertEqual(depth.shape, (1, 1, 32, 32))
        self.assertEqual(flow.shape, (1, 2, 32, 32))

if __name__ == '__main__':
    unittest.main()
