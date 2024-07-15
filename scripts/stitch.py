import argparse
from pathlib import Path
import pytorch_lightning as pl
import torch
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger

from helper_scripts.CIFAR10Module import CIFAR10Module
from helper_scripts.StitchingModelModule import StitchingModelModule
from stitching_layer import StitchingModel
from helper_scripts.CIFAR10Data import CIFAR10Data
from helper_scripts.utils import find_checkpoint_for_model


def save_model_states(results, stage, model1_name, model1, model2_name,  model2, stitching_model):
    # QUESTION: could we just use stitching_model.state_dict(), since the stitching_model contains
    # both model1 and model2?
    results[stage] = {
        f"{model1_name}_state_dict": model1.state_dict(),
        f"{model2_name}_state_dict": model2.state_dict(),
        "stitching_model_state_dict": stitching_model.stitching_layer.state_dict(),
    }


# SUGGESTED REWRITE:
def snapshot(stitching_model, data_module) -> dict:
    return {
        "state_dict": stitching_model.state_dict(),
        "loss": test_model(stitching_model, data_module, trainer),  # maybe test_model should instantiate its own trainer? it's awkward to pass one in here...
    }

def load_model(model_name, log_dir):
    checkpoint_path = find_checkpoint_for_model(log_dir, model_name)
    print(f"[INFO]: loading {model_name} from", checkpoint_path)
    model = CIFAR10Module.load_from_checkpoint(checkpoint_path)
    return model


def do_linear_regression(stitching_model, datamodule):
    train_loader = datamodule.train_dataloader()
    batch_im, _ = next(iter(train_loader))
    batch_im = batch_im.to(
        next(stitching_model.parameters()).device
    )  # Ensure batch_im is on the same device
    stitching_model.initialize_stitching_layer(batch_im)


def test_model(model, datamodule, trainer):
    test_module = StitchingModelModule(model, learning_rate=1e-3)
    test_results = trainer.test(test_module, datamodule=datamodule, verbose=False)
    print(f"Test results for {model.__class__.__name__}: {test_results}")
    # Assuming the default key for loss in the test results is 'test_loss'
    # Find the key containing the loss
    possible_keys = ["test_loss_epoch", "loss", "test_loss"]
    for key in possible_keys:
        if key in test_results[0]:
            return test_results[0][key]
    raise KeyError("No recognized loss key found in test results")


def main(model1_name, model2_name, split1, split2, log_dir, data_dir, num_epochs):
    # BEST-PRACTICES ISSUE: many lines of duplicate code here and in stitch_with_learning.py. Could
    # we refactor the duplicated parts?
    # ALSO, A SUGGESTION: just define results = {} here. Later, we can do results[stage] = ..., and
    # this way we can use the existence of a key in the dictionary to check if a stage has been run.
    results = {
        "init": {},
        "after_regression": {},
        "before_training": {},
        "after_training": {},
        "losses": {},  # TODO: See other comments... losses should be part of the snapshot at every stage.
    }

    model1 = load_model(model1_name, log_dir)
    model2 = load_model(model2_name, log_dir)
    stitching_model = StitchingModel(model1, model2, split1, split2)

    log_dir = Path(log_dir) / "checkpoints"
    # log_dir.mkdir(exist_ok=True, parents=True)

    cifar10_data = CIFAR10Data(
        data_dir=data_dir, batch_size=32, num_workers=4, pin_memory=True
    )
    cifar10_data.prepare_data()
    cifar10_data.setup(stage="fit")

    # SUGGESTION: do results["init"] = save_model_states(...). No need for the save_model_states
    # function to have a "side effect" of modifying the results dictionary.
    save_model_states(results, "init", model1_name, model1, model2_name, model2, stitching_model)

    # SUGGESTION 2: call torch.save() on the results after each stage. Then we can also torch.load()
    # whatever is already done each run and continue from there.

    do_linear_regression(stitching_model, cifar10_data)

    save_model_states(results, "after_regression", model1_name, model1, model2_name, model2, stitching_model)

    logger = TensorBoardLogger(save_dir=log_dir, name="stitching_model")

    # SUGGESTION: add callbacks for LR decay and early-stopping.
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        logger=logger,
        strategy=(
            DDPStrategy(find_unused_parameters=True)
            if torch.cuda.device_count() > 1
            else None
        ),
    )

    stitching_module = StitchingModelModule(stitching_model, learning_rate=1e-3)

    # QUESTION: is "before_training" identical to "after_regression"? Nothing has been done to the
    # model in between, right?
    save_model_states(results, "before_training", model1_name, model1, model2_name, model2, stitching_model)

    trainer.fit(stitching_module, datamodule=cifar10_data)

    save_model_states(results, "after_training", model1_name, model1, model2_name, model2, stitching_model)

    # Test models and capture losses
    # TODO: losses must be added to the results dictionary *at every stage*. I recommend redesigning
    #  the save_model_states function (which used to be 'snapshot()')... see above...
    results["losses"][f"{model1_name}_loss"] = test_model(model1, cifar10_data, trainer)
    results["losses"][f"{model2_name}_loss"] = test_model(model2, cifar10_data, trainer)
    results["losses"]["stitching_model_loss"] = test_model(
        stitching_model, cifar10_data, trainer
    )

    # Log model architecture
    batch = next(iter(cifar10_data.train_dataloader()))[0].to(
        next(stitching_model.parameters()).device
    )
    logger.experiment.add_graph(stitching_model, batch)

    # Reshape weights for embedding
    # COMMENT: nice idea to visualize the weights. I'm not sure what to make of them just yet, but
    #  in the spirit of "log everything now, parse it later", it's not a bad idea.
    conv_weights = stitching_model.stitching_layer.conv.weight.data
    conv_weights_reshaped = conv_weights.view(conv_weights.size(0), -1)
    logger.experiment.add_embedding(
        conv_weights_reshaped, metadata=None, label_img=None
    )

    # Save results
    torch.save(results, log_dir / "results.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stitching Model Creation Script")
    parser.add_argument(
        "--model1_name", type=str, required=True, help="Name of the first model"
    )
    parser.add_argument(
        "--model2_name", type=str, required=True, help="Name of the second model"
    )
    parser.add_argument(
        "--index1",
        type=int,
        required=True,
        help="Split Index of the layer in the first model",
    )
    parser.add_argument(
        "--index2",
        type=int,
        required=True,
        help="Split Index of the layer in the second model",
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        required=True,
        help="Directory to store logs and checkpoints",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Directory to store data",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of epochs to train the model"
    )
    args = parser.parse_args()

    main(
        args.model1_name,
        args.model2_name,
        args.index1,
        args.index2,
        args.log_dir,
        args.data_dir,
        args.num_epochs,
    )
