#!/bin/bash
# Universal shell script to execute the stitching model training script with provided parameters

# Set the parameter values
MODEL1_NAME="resnet18"
MODEL2_NAME="resnet34"
INDEX1=5
INDEX2=5
NUM_EPOCHS=2
BATCH_SIZE=64
NUM_WORKERS=4
PIN_MEMORY="--pin_memory"

# Define the Python script to run
PYTHON_SCRIPT="main_controller.py"
REQUIREMENTS_FILE="requirements.txt"

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

# Function to run the Python script with the parameters
run_python_script() {
    $PYTHON_CMD $PYTHON_SCRIPT --model1_name $MODEL1_NAME --model2_name $MODEL2_NAME --index1 $INDEX1 --index2 $INDEX2 --num_epochs $NUM_EPOCHS --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS $PIN_MEMORY
}

# Check environment
check_environment

# Run additional commands
install_requirements

# Run the Python script
run_python_script
