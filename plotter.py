from matplotlib import pyplot as plt


def plot_stitching_penalty(penalties, model1_name, model2_name):
    fractions = [i / (len(penalties) - 1) for i in range(len(penalties))]

    plt.figure(figsize=(10, 6))
    plt.plot(fractions, penalties, "bo-", label=f"{model1_name}")
    plt.xlabel("Fraction of layers from bottom model")
    plt.ylabel("Stitching Penalty (Loss)")
    plt.title("Stitching Penalty across different fractions of layers")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"stitching_penalty_{model1_name}.png")
    plt.show()


def plot_losses(
    stitched_train_losses,
    stitched_test_losses,
    model1_train_losses,
    model1_test_losses,
    model1_name,
    model2_name,
    index1,
    index2,
):
    epochs = range(1, len(stitched_train_losses) + 1)

    plt.figure(figsize=(12, 8))

    # Plot training losses
    plt.subplot(2, 1, 1)
    plt.plot(epochs, stitched_train_losses, "bo-", label="Stitched Model Training Loss")
    plt.plot(epochs, model1_train_losses, "ro-", label="Model 1 Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.legend()
    plt.grid(True)

    # Plot testing losses
    plt.subplot(2, 1, 2)
    plt.plot(epochs, stitched_test_losses, "bo-", label="Stitched Model Testing Loss")
    plt.plot(epochs, model1_test_losses, "ro-", label="Model 1 Testing Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Testing Loss")
    plt.legend()
    plt.grid(True)

    plt.suptitle(
        f"Training and Testing Losses for Stitched Model and Model 1 ({model1_name}, {model2_name})"
    )
    plt.tight_layout()
    plt.savefig(
        f"training_testing_losses_{model1_name}_{model2_name}_{index1}_{index2}.png"
    )
    plt.close()
