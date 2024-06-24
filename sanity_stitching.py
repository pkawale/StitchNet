import argparse
import torch
from torch import nn, optim

from stitching_layer import StitchingModel
from utils import setup_logging, load_dataset
from main_controller import train, test
import matplotlib.pyplot as plt


def main(
    model1_name,
    model2_name,
    split1,
    split2,
    batch_size,
    num_workers,
    pin_memory,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_dataset(
        batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory
    )

    training_logger, testing_logger, comparison_logger = setup_logging(
        model1_name, model2_name
    )

    #####
    # REGRESSION INIT
    #####

    stitching_model_regression_init = StitchingModel(
        model1_name, model2_name, split1, split2
    ).to(device)

    with torch.no_grad():
        batch_im, _ = next(iter(train_loader))
        batch_im = batch_im.to(device)
        model1_out = stitching_model_regression_init.part1_model1(batch_im)
        model2_out = stitching_model_regression_init.part1_model2(batch_im)
        stitching_model_regression_init.stitching_layer.initialize_weights_with_regression(
            model1_out, model2_out
        )
        del batch_im

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(stitching_model_regression_init.parameters(), lr=0.001)

    ### TEMPORARY
    train(stitching_model_regression_init.model1, train_loader, criterion, optimizer, device, 10, training_logger)
    train(stitching_model_regression_init.model2, train_loader, criterion, optimizer, device, 10, training_logger)
    model1_loss, model1_acc = test(stitching_model_regression_init.model1, test_loader, criterion, device)
    model2_loss, model2_acc = test(stitching_model_regression_init.model2, test_loader, criterion, device)
    print("Model 1 (pretrained) loss:", model1_loss)
    print("Model 1 (pretrained) acc:", model1_acc)
    print("Model 2 (pretrained) loss:", model2_loss)
    print("Model 2 (pretrained) acc:", model2_acc)

    assert model1_acc > 50 and model2_acc > 50, "Models and data are mismatched; fix this before stitching!"

    regression_init_train_losses = []
    regression_init_test_losses = []

    for epoch in range(10):
        train_loss = train(
            stitching_model_regression_init,
            train_loader,
            criterion,
            optimizer,
            device,
            num_epochs=1,
            logger=training_logger,
        )
        test_loss = test(stitching_model_regression_init, test_loader, criterion, device)[0]

        regression_init_train_losses.append(train_loss)
        regression_init_test_losses.append(test_loss)

    #####
    # RANDOM INIT
    #####

    stitching_model_random_init = StitchingModel(
        model1_name, model2_name, split1, split2
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(stitching_model_random_init.parameters(), lr=0.001)

    random_init_test_losses = []
    random_init_train_losses = []

    for epoch in range(10):
        train_loss = train(
            stitching_model_random_init,
            train_loader,
            criterion,
            optimizer,
            device,
            num_epochs=1,
            logger=training_logger,
        )
        test_loss = test(stitching_model_random_init, test_loader, criterion, device)[0]

        random_init_train_losses.append(train_loss)
        random_init_test_losses.append(test_loss)

    plt.figure(figsize=(10,6))
    plt.plot(regression_init_train_losses, label="Regression Init Train Loss")
    plt.plot(regression_init_test_losses, label="Regression Init Test Loss")
    plt.plot(random_init_train_losses, label="Random Init Train Loss")
    plt.plot(random_init_test_losses, label="Random Init Test Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Test Losses')
    plt.savefig("graph.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stitching Model Training Script")
    parser.add_argument(
        "--model1_name", type=str, required=True, help="Name of the first model"
    )
    parser.add_argument(
        "--model2_name", type=str, required=True, help="Name of the second model"
    )
    parser.add_argument(
        "--index1", type=int, required=True, help="Split index for the first model"
    )
    parser.add_argument(
        "--index2", type=int, required=True, help="Split index for the second model"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training and testing"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,  # Reduced number of workers
        help="Number of worker threads for data loading",
    )
    parser.add_argument(
        "--pin_memory", action="store_true", help="Use pinned memory for data loading"
    )

    args = parser.parse_args()

    main(
        args.model1_name,
        args.model2_name,
        args.index1,
        args.index2,
        args.batch_size,
        args.num_workers,
        args.pin_memory,
    )
