#!/bin/bash
# Universal shell script to execute the stitching model training script with provided parameters

# Define the Python script to run
PYTHON_SCRIPT="main_controller.py"
REQUIREMENTS_FILE="requirements.txt"

# Specify the GPUs to use (comma-separated list of GPU indices)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4

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
    "resnet18 resnet18 5 5"
)

# Function to run the Python script with the parameters
run_python_script() {
    local model1_name=$1
    local model2_name=$2
    local index1=$3
    local index2=$4
    local num_epochs=$5
    local batch_size=$6
    local num_workers=$7
    local pin_memory=$8
    local data_dir=$9
    local pretrained=${10}
    local test_phase=${11}
    local dev=${12}
    local precision=${13}
    local learning_rate=${14}
    local weight_decay=${15}

    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES $PYTHON_CMD $PYTHON_SCRIPT \
        --model1_name $model1_name \
        --model2_name $model2_name \
        --index1 $index1 \
        --index2 $index2 \
        --num_epochs $num_epochs \
        --batch_size $batch_size \
        --num_workers $num_workers \
        $pin_memory \
        --data_dir $data_dir \
        --pretrained $pretrained \
        --test_phase $test_phase \
        --dev $dev \
        --precision $precision \
        --learning_rate $learning_rate \
        --weight_decay $weight_decay
}

# Check environment
check_environment

# Run additional commands
install_requirements

# Set common parameters
NUM_EPOCHS=150
BATCH_SIZE=64
NUM_WORKERS=4
PIN_MEMORY="--pin_memory"
DATA_DIR="./data"  # Set this to your actual data directory
PRETRAINED=0
TEST_PHASE=0
DEV=0
PRECISION=32
LEARNING_RATE=1e-3
WEIGHT_DECAY=1e-4

# Iterate over each parameter set and run the Python script
for params in "${PARAM_SETS[@]}"; do
    set -- $params
    model1_name=$1
    model2_name=$2
    index1=$3
    index2=$4

    echo "Running script with parameters: Model1=$model1_name, Model2=$model2_name, Index1=$index1, Index2=$index2"
    run_python_script $model1_name $model2_name $index1 $index2 $NUM_EPOCHS $BATCH_SIZE $NUM_WORKERS $PIN_MEMORY $DATA_DIR $PRETRAINED $TEST_PHASE $DEV $PRECISION $LEARNING_RATE $WEIGHT_DECAY
done