import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_results(results_file):
    try:
        results = torch.load(results_file, map_location="cpu")
        return results
    except FileNotFoundError:
        print(f"[ERROR]: Results file not found: {results_file}")
        return None


def plot_losses(results, output_dir):
    stages = [key for key in results.keys()]
    for stage in stages:
        losses = results.get(stage, {}).get("losses")
        if not losses:
            print(f"[ERROR]: No losses found in {stage}")
            continue

        # Extract losses
        resnet18_loss = losses["resnet18_loss"]
        resnet34_loss = losses["resnet34_loss"]
        stitching_model_loss = losses["stitching_model_loss"]

        # Plotting losses
        plt.figure()
        stages = ["ResNet-18", "ResNet-34", "Stitching Model"]
        loss_values = [resnet18_loss, resnet34_loss, stitching_model_loss]

        plt.bar(stages, loss_values, color=["blue", "green", "red"])
        plt.xlabel("Models")
        plt.ylabel("Loss")
        plt.title(f"Losses for {stage}")
        plt.grid(True)
        plt.savefig(output_dir / f"losses_{stage}.png")
        plt.close()
        print(f"[INFO]: Saved losses plot to {output_dir / f'losses_{stage}.png'}")


def plot_weight_diff_vs_loss(results, output_dir):
    stages = [key for key in results.keys() if key.startswith("after_training_model2_part2_stitching")]
    if not stages:
        print("[ERROR]: No stages found for weight difference and loss plotting")
        return

    weight_diffs = []
    losses = []
    lambda_values = []

    for stage in stages:
        losses_dict = results[stage].get("losses")
        if not losses_dict:
            continue

        stitching_model_loss = losses_dict["stitching_model_loss"]

        init_weights = results["init"]["state_dict"]
        final_weights = results[stage]["state_dict"]

        model2_weight_diff = 0
        for key in final_weights.keys():
            if key.startswith("part2_model2") and torch.is_floating_point(final_weights[key]):
                model2_weight_diff += torch.norm(final_weights[key].float() - init_weights[key].float()).item()

        lambda_value = float(stage.split("_")[-1])
        regularizer = lambda_value * model2_weight_diff
        total_loss = stitching_model_loss + regularizer

        weight_diffs.append(model2_weight_diff)
        losses.append(total_loss)
        lambda_values.append(lambda_value)

    plt.figure()
    plt.scatter(weight_diffs, losses, c=lambda_values, cmap='viridis', label="Loss vs Weight Difference")
    plt.colorbar(label='Lambda Value')
    plt.xlabel("Weight Difference")
    plt.ylabel("Total Loss (Loss + Regularizer)")
    plt.title("Weight Difference vs Loss with Regularizer")
    plt.grid(True)
    plt.savefig(output_dir / "weight_diff_vs_loss.png")
    plt.close()
    print(f"[INFO]: Saved weight difference vs loss plot to {output_dir / 'weight_diff_vs_loss.png'}")


def plot_weights(results, output_dir):
    init_state_dict = results.get("init", {}).get("state_dict")
    if not init_state_dict:
        print("[ERROR]: No initial state dict found in results")
        return

    conv_weights = init_state_dict.get("stitching_layer.conv.weight")
    if conv_weights is None:
        print("[ERROR]: No convolution weights found in initial state dict")
        return

    conv_weights_reshaped = conv_weights.view(conv_weights.size(0), -1).numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(conv_weights_reshaped, cmap="viridis")
    plt.xlabel("Weights")
    plt.ylabel("Filters")
    plt.title("Stitching Layer Weights")
    plt.savefig(output_dir / "stitching_layer_weights.png")
    plt.close()
    print(f"[INFO]: Saved stitching layer weights plot to {output_dir / 'stitching_layer_weights.png'}")


def main(results_file, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    results = load_results(results_file)
    if results is None:
        return

    plot_losses(results, output_dir)
    plot_weights(results, output_dir)
    plot_weight_diff_vs_loss(results, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualization Script for Stitching Model Results")
    parser.add_argument(
        "--results_file", type=Path, required=True, help="Path to the results file"
    )
    parser.add_argument(
        "--output_dir", type=Path, required=True, help="Directory to save visualizations"
    )
    args = parser.parse_args()
    main(args.results_file, args.output_dir)