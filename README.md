# RW-Eng-v3

This repository holds extended research of a thematic fit model that has been trained using a multi-task residual role-filler approach. 

## Data and License
The corpus itself is available at: http://yuvalmarton.com/rw-eng/

The corpus contains documents coming from the British National Corpus (BNC) and ukWaC. Therefore, the license for using it is the same license as the ukWac ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode): summary, which is not instead of the license: attribute our work, share, adapt, whatever, just not for commercial use, and don’t sue us for anything). The BNC no longer requires a license.

If you use this corpus or the code in this repository, please cite us (See details below).

## Please cite:
Yuval Marton, Asad Sayeed (2021). *Thematic fit bits: Annotation quality and quantity for event participant representation.* http://arxiv.org/abs/2105.06097

BibTex:
```
@misc{marton-sayeed-2021-RW-eng-v2,
      title={Thematic fit bits: Annotation quality and quantity for event participant representation},
      author={Yuval Marton and Asad Sayeed},
      year={2021},
      eprint={2105.06097},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Key Dependencies

- **Tested OS:** Ubuntu 18.04, Windows
- Python 3.7
- Tensorflow 2.4
- CUDA 11/10.1

Numerous problems were encountered using Python versions above 3.7 as well as Tensorflow 2.5. Instructions for setting up CUDA 11 and 10.1 on Ubuntu systems can be found [here](https://www.tensorflow.org/install/gpu#install_cuda_with_apt) and [here](http://web.archive.org/web/20201207152356/https://www.tensorflow.org/install/gpu) respectively. For Windows systems, please refer to official installers for NVIDIA drivers and CUDA versions. cudNN "installation" comprises of moving files to the CUDA installation directory and can be found from the [NVIDIA Developer page](https://developer.nvidia.com/rdp/cudnn-archive) (account required). Once again, please take care of compatible versions. **Install CUDA versions above 11.2 at your own risk, as Tensorflow has not been tested for those versions.**

## Setting Up

Once NVIDIA, CUDA, and cuDNN libraries are installed and verified with `nvidia-smi` and `nvcc --version`, run the following commands to set up and activate an Anaconda environment with our libraries.

```bash
git clone https://github.com/MughilM/RW-Eng-v3-src.git
cd RW-Eng-v3-src
conda env create -f environments/capstoneenv_linux.yml
conda activate rweng2-wheres-the-learning
```

Make sure to select the correct environment file pertaining to your OS. This will create the `rweng2-wheres-the-learning` environment. The quickest way to test GPU visibility is to run `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`. If all is working corrcetly, you should see a single entry with your GPU listed.

Once activated the conda env, you may need to add the following as well:
```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
# the following line may or may not be needed:
python3 -m pip install tensorflow
# Verify install:
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# if getting error with libcusolver.so.10 , per the advice in https://github.com/tensorflow/tensorflow/issues/43947#issuecomment-715295153 , try adding something like:
sudo ln -s YOUR_CONDA_ENV_PATH/lib/libcusolver.so.11  YOUR_CONDA_ENV_PATH/lib/libcusolver.so.10 
# You may also need this:
sudo apt install graphviz
```

## Usage

Everything is run from `main.py`. We have included a small dataset of 1000 samples in `processed_data` called `v2` (The second argument is the `--data_version`). The `processed_data` folder is currently set as the data directory, which means **any dataset you wish to use needs to be moved to this folder**. Simply choose the model and dataset name, along with other optional parameters such as experiment name, epochs, etc. Experiments are saved to the `experiments` subfolder.

```bash
cd event_rep
python main.py \
  v4 \
  v2 \
  --experiment_name test_exp \
  --epochs 15 \
  --batch_size 64 \
  --do_eval \
  --evaluation_tasks pado mcrae
```

This will train the MTRFv4Res model with 15 epochs and save under `test_exp`. It will also perform the Pado07 and McRae05 thematic fit evaluation tasks. Please see `models.py` to see a list of models currently implemented. The `PARAM_TO_MODEL` variable in `main.py` states which argument name maps to which model. Any extra models you wish to implement should extend `MTRFv4Res`, placed in `models.py`, and have `PARAM_TO_MODEL` updated.

### Only Evaluation

If evaluating on an already trained model is desired, then simply provide the experiment name and the `--eval_only` flag:

```bash
cd event_rep
python main.py \
  v4 \
  v2 \
  --experiment_name test_exp \
  --eval_only \
  --evaluation_tasks all
```

Due to the very last flag, this will run ALL the thematic fit evaluation tasks.

## Research Training Results

Input table here of training and evaluation results that are in the paper...

