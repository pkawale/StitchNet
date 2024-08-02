from pathlib import Path
import torch
import unittest
from functools import wraps


LAM_KEY = f"after_training_model2_part2_stitching_{1.0:.3f}"


def foreach_file(test_fn):
    @wraps(test_fn)
    def wrapper(self):
        for filename in self.files:
            with self.subTest(filename):
                results = torch.load(filename)
                test_fn(self, results)

    return wrapper


class TestSanityCheck(unittest.TestCase):
    WHERE = Path("/data/projects/StitchNet/logs/analysis/")
    data = None

    @classmethod
    def setUpClass(cls):
        cls.files = list(cls.WHERE.glob("results_resnet18_resnet34_*_lr1.00e-03.pth"))

    @foreach_file
    def test_init_loss_sensible(self, results):
        self.assertLess(
            results["init"]["losses"]["model1_loss"]["test_cross_entropy"], 1.0
        )
        self.assertLess(
            results["init"]["losses"]["model2_loss"]["test_cross_entropy"], 1.0
        )
        self.assertGreater(
            results["init"]["losses"]["stitching_model_loss"]["test_cross_entropy"], 1.0
        )

    @foreach_file
    def test_regression_loss_sensible(self, results):
        self.assertLess(
            results["after_regression"]["losses"]["stitching_model_loss"][
                "test_cross_entropy"
            ],
            results["init"]["losses"]["stitching_model_loss"]["test_cross_entropy"],
        )

    @foreach_file
    def test_learning_loss_sensible(self, results):
        self.assertLess(
            results["after_training"]["losses"]["stitching_model_loss"][
                "test_cross_entropy"
            ],
            results["after_regression"]["losses"]["stitching_model_loss"][
                "test_cross_entropy"
            ],
        )

    @foreach_file
    def test_m2_loss_sensible(self, results):
        self.assertLess(
            results[LAM_KEY]["losses"]["stitching_model_loss"]["test_cross_entropy"],
            results["after_training"]["losses"]["stitching_model_loss"][
                "test_cross_entropy"
            ]
            * 1.1,
        )
        self.assertGreater(
            results[LAM_KEY]["losses"]["stitching_model_loss"]["test_delta_weights"],
            0.0,
        )

    @foreach_file
    def test_regression_does_not_affect_non_stitching_params(self, results):
        state_dict_before = results["init"]["state_dict"]
        state_dict_after = results["after_regression"]["state_dict"]
        for key in state_dict_before.keys():
            if "stitching_layer" in key:
                continue
            if not isinstance(state_dict_before[key], torch.Tensor):
                continue
            self.assertTrue(
                torch.allclose(state_dict_before[key], state_dict_after[key]),
                msg=f"Key: {key}",
            )

    @foreach_file
    def test_regression_does_affect_stitching_params(self, results):
        state_dict_before = results["init"]["state_dict"]
        state_dict_after = results["after_regression"]["state_dict"]
        for key in state_dict_before.keys():
            if "stitching_layer" not in key:
                continue
            if not isinstance(state_dict_before[key], torch.Tensor):
                continue
            self.assertFalse(
                torch.allclose(state_dict_before[key], state_dict_after[key]),
                msg=f"Key: {key}",
            )

    @foreach_file
    def test_stitching_layer_learning_does_not_affect_non_stitching_params(
        self, results
    ):
        state_dict_before = results["after_regression"]["state_dict"]
        state_dict_after = results["after_training"]["state_dict"]
        for key in state_dict_before.keys():
            if "stitching_layer" in key:
                continue
            if not isinstance(state_dict_before[key], torch.Tensor):
                continue
            self.assertTrue(
                torch.allclose(state_dict_before[key], state_dict_after[key]),
                msg=f"Key: {key}",
            )

    @foreach_file
    def test_stitching_layer_learning_does_affect_stitching_params(self, results):
        state_dict_before = results["after_regression"]["state_dict"]
        state_dict_after = results["after_training"]["state_dict"]
        for key in state_dict_before.keys():
            if "stitching_layer" not in key:
                continue
            if not isinstance(state_dict_before[key], torch.Tensor):
                continue
            self.assertFalse(
                torch.allclose(state_dict_before[key], state_dict_after[key]),
                msg=f"Key: {key}",
            )

    @foreach_file
    def test_m2_learning_does_not_affect_wrong_stuff(self, results):
        state_dict_before = results["after_training"]["state_dict"]
        state_dict_after = results[LAM_KEY]["state_dict"]
        for key in state_dict_before.keys():
            if "stitching_layer" in key or "part2_model2" in key:
                continue
            if not isinstance(state_dict_before[key], torch.Tensor):
                continue
            self.assertTrue(
                torch.allclose(state_dict_before[key], state_dict_after[key]),
                msg=f"Key: {key}",
            )

    @foreach_file
    def test_m2_learning_does_affect_stitching_params(self, results):
        state_dict_before = results["after_regression"]["state_dict"]
        state_dict_after = results["after_training"]["state_dict"]
        for key in state_dict_before.keys():
            if "stitching_layer" not in key or "part2_model2" not in key:
                continue
            if not isinstance(state_dict_before[key], torch.Tensor):
                continue
            self.assertFalse(
                torch.allclose(state_dict_before[key], state_dict_after[key]),
                msg=f"Key: {key}",
            )
