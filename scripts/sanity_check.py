from pathlib import Path
import torch
import unittest


LAM_KEY = f"after_training_model2_part2_stitching_{1.0:.3f}"


class TestSanityCheck(unittest.TestCase):
    WHERE = Path("/data/projects/StitchNet/logs/analysis/")
    data = None

    @classmethod
    def setUpClass(cls):
        files = list(cls.WHERE.glob("results_resnet18_resnet34_*_*.pth"))
        cls.data = {file.name: torch.load(file) for file in files}

    def test_init_loss_sensible(self):
        for key, value in self.data.items():
            self.assertLess(value["init"]["losses"]["model1_loss"], 1.0)
            self.assertLess(value["init"]["losses"]["model2_loss"], 1.0)
            self.assertGreater(value["init"]["losses"]["stitching_model_loss"], 1.0)

    def test_regression_loss_sensible(self):
        for key, value in self.data.items():
            self.assertLess(
                value["after_regression"]["losses"]["stitching_model_loss"],
                value["init"]["losses"]["stitching_model_loss"],
            )

    def test_learning_loss_sensible(self):
        for key, value in self.data.items():
            self.assertLess(
                value["after_training"]["losses"]["stitching_model_loss"],
                value["after_regression"]["losses"]["stitching_model_loss"],
            )

    def test_m2_loss_sensible(self):
        for key, value in self.data.items():
            self.assertLess(
                value[LAM_KEY]["losses"]["stitching_model_loss"],
                value["after_training"]["losses"]["stitching_model_loss"],
            )

    def test_regression_does_not_affect_non_stitching_params(self):
        for key, value in self.data.items():
            state_dict_before = value["init"]["state_dict"]
            state_dict_after = value["after_regression"]["state_dict"]
            for key in state_dict_before.keys():
                if "stitching_layer" in key:
                    continue
                if not isinstance(state_dict_before[key], torch.Tensor):
                    continue
                self.assertTrue(
                    torch.allclose(state_dict_before[key], state_dict_after[key]),
                    msg=f"Key: {key}",
                )

    def test_regression_does_affect_stitching_params(self):
        for key, value in self.data.items():
            state_dict_before = value["init"]["state_dict"]
            state_dict_after = value["after_regression"]["state_dict"]
            for key in state_dict_before.keys():
                if "stitching_layer" not in key:
                    continue
                if not isinstance(state_dict_before[key], torch.Tensor):
                    continue
                self.assertFalse(
                    torch.allclose(state_dict_before[key], state_dict_after[key]),
                    msg=f"Key: {key}",
                )

    def test_stitching_layer_learning_does_not_affect_non_stitching_params(self):
        for key, value in self.data.items():
            state_dict_before = value["after_regression"]["state_dict"]
            state_dict_after = value["after_training"]["state_dict"]
            for key in state_dict_before.keys():
                if "stitching_layer" in key:
                    continue
                if not isinstance(state_dict_before[key], torch.Tensor):
                    continue
                self.assertTrue(
                    torch.allclose(state_dict_before[key], state_dict_after[key]),
                    msg=f"Key: {key}",
                )

    def test_stitching_layer_learning_does_affect_stitching_params(self):
        for key, value in self.data.items():
            state_dict_before = value["after_regression"]["state_dict"]
            state_dict_after = value["after_training"]["state_dict"]
            for key in state_dict_before.keys():
                if "stitching_layer" not in key:
                    continue
                if not isinstance(state_dict_before[key], torch.Tensor):
                    continue
                self.assertFalse(
                    torch.allclose(state_dict_before[key], state_dict_after[key]),
                    msg=f"Key: {key}",
                )

    def test_m2_learning_does_not_affect_wrong_stuff(self):
        for key, value in self.data.items():
            state_dict_before = value["after_training"]["state_dict"]
            state_dict_after = value[LAM_KEY]["state_dict"]
            for key in state_dict_before.keys():
                if "stitching_layer" in key or "part2_model2" in key:
                    continue
                if not isinstance(state_dict_before[key], torch.Tensor):
                    continue
                self.assertTrue(
                    torch.allclose(state_dict_before[key], state_dict_after[key]),
                    msg=f"Key: {key}",
                )

    def test_m2_learning_does_affect_stitching_params(self):
        for key, value in self.data.items():
            state_dict_before = value["after_regression"]["state_dict"]
            state_dict_after = value["after_training"]["state_dict"]
            for key in state_dict_before.keys():
                if "stitching_layer" not in key or "part2_model2" not in key:
                    continue
                if not isinstance(state_dict_before[key], torch.Tensor):
                    continue
                self.assertFalse(
                    torch.allclose(state_dict_before[key], state_dict_after[key]),
                    msg=f"Key: {key}",
                )
