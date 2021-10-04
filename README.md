# RW-Eng-v3

This repository holds extended research of a thematic fit model that has been trained using a multi-task residual role-filler approach. 

## MODEL-SUBCLASS BRANCH
The `model-subclass` branch holds implementation of our models where we subclass the Tensorflow `Model` object in
order to create the models. Doing it this way allows us to access `.fit()` and `.evaluate()` directly on the Model.
However, when doing it this way using multiple inputs and multiple outputs like our model has, it led to training issues
where validation performance degraded each epoch. Only when I removed the sublcass and implemented a `build_model()`
function instead did training occur as normal. In about a week time frame, we asked a question regarding this on
StackOverflow. That question can be seen [here](https://stackoverflow.com/questions/69285578/validation-test-worse-metrics-with-multi-input-output-tensorflow-model-subclass).
No response happened, and so we moved this implementation into another branch. If time permits and we get a fix,
then maybe we will update the `main` branch with the subclass. For now, this stays here.

## Key Dependencies

- **Tested OS:** Ubuntu 18.04, Windows
- Python 3.7
- Tensorflow 2.4
- CUDA 11/10.1

Numerous problems were encountered using Python versions above 3.7 as well as Tensorflow 2.5. Instructions for setting up CUDA 11 and 10.1 on Ubuntu systems can be found [here](https://www.tensorflow.org/install/gpu#install_cuda_with_apt) and [here](http://web.archive.org/web/20201207152356/https://www.tensorflow.org/install/gpu) respectively.

## Setting Up

Once NVIDIA, CUDA, and cuDNN libraries are installed and verified with `nvidia-smi` and `nvcc --version`, run the following commands to set up and activate an Anaconda environment with our libraries.

```bash
cd environments
conda env create -f <ENV_FILE>.yml
conda activate bloomberg
```

Make sure to select the correct environment file pertaining to your OS. This will create the `bloomberg` environment.

## Usage

Everything is run from `main.py`. We have included a small dataset of 1000 samples in `processed_data`. This folder is currently set as the data directory. Any additional datasets should be placed here. Simply choose the model and dataset name, along with other optional parameters such as experiment name, epochs, etc. Experiments are saved to the `experiments` subfolder.

```bash
python main.py v4 v2 \
  --experiment_name test_exp \
  --epochs 15 \
  --batch_size 64 \
  --do_eval \
  --evaluation_tasks pado mcrae
```

This will train the MTRFv4Res model with 15 epochs and save under `test_exp`. It will also perform the Pado07 and McRae05 thematic fit evaluation tasks.

### Only Evaluation

If evaluating on an already trained model is desired, then simply provide the experiment name and the `--eval_only` flag:

```bash
python main.py v4 v2 \
  --experiment_name test_exp \
  --eval_only \
  --run_all_tasks
```

Due to the very last flag, this will run ALL the thematic fit evaluation tasks.

## Research Training Results

Input table here of training and evaluation results that are in the paper...

