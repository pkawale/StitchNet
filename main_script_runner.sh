#!/bin/bash
# Universal shell script to execute the stitching model training script with provided parameters

# Define the Python script to run
PYTHON_SCRIPT="main_controller.py"
REQUIREMENTS_FILE="requirements.txt"

# Specify the GPUs to use (comma-separated list of GPU indices)
# shellcheck disable=SC2054
CUDA_VISIBLE_DEVICES=(0,1,2)

# Function to check if the script is running in a supported environment
check_environment() {
    if command -v python3 &>/dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &>/dev/null; then
        PYTHON_CMD="python"
    else
        echo "Python is not installed. Please install Python and try again."
        exit 1
    fi
}

# Function to install requirements if not already installed
install_requirements() {
    if [ -f "$REQUIREMENTS_FILE" ]; then
        echo "Installing requirements from $REQUIREMENTS_FILE..."
        $PYTHON_CMD -m pip install --upgrade pip
        $PYTHON_CMD -m pip install -r $REQUIREMENTS_FILE
    else
        echo "No requirements.txt file found. Skipping installation of requirements."
    fi
}

# Array of parameters (model1_name, model2_name, index1, index2)
PARAM_SETS=(
    "resnet18 resnet34 5 5"
)

# Function to run the Python script with the parameters
run_python_script() {
    local model1_name=$1
    local model2_name=$2
    local index1=$3
    local index2=$4
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES $PYTHON_CMD $PYTHON_SCRIPT --model1_name $model1_name --model2_name $model2_name --index1 $index1 --index2 $index2 --num_epochs $NUM_EPOCHS --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS $PIN_MEMORY
}

# Check environment
check_environment

# Run additional commands
install_requirements

# Set common parameters
NUM_EPOCHS=10
BATCH_SIZE=64
NUM_WORKERS=4
PIN_MEMORY="--pin_memory"

# Iterate over each parameter set and run the Python script
for params in "${PARAM_SETS[@]}"; do
    set -- $params
    model1_name=$1
    model2_name=$2
    index1=$3
    index2=$4

    echo "Running script with parameters: Model1=$model1_name, Model2=$model2_name, Index1=$index1, Index2=$index2"
    run_python_script $model1_name $model2_name $index1 $index2
done
