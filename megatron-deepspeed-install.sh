!/bin/bash

#### Environment configuration ####

echo "Setting up CUDA environment"
export MAX_JOBS=10
yes | conda install -c nvidia/label/cuda-12.1.0 cuda-toolkit
yes | conda install cudnn -c nvidia/label/cuda-12.1.0

echo "Installing required Python packages"
yes | pip install tensorstore==0.1.45 zarr==2.18.2 six==1.16.0 regex==2024.5.15 psutil==6.0.0 pybind11==2.13.1 packaging==24.1 tensorboard==2.17.0

echo "Installing PyTorch"
yes | conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia

echo "Installing Apex"
git clone --branch 24.04.01 --single-branch https://github.com/NVIDIA/apex
yes | pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./apex/

echo "Installing Flash Attention"
yes | pip install flash-attn --no-build-isolation

echo "Installing DeepSpeed"
yes | pip install deepspeed==0.16.2 transformers==4.44.2

#### Validate your installation of deepspeed ####

ds_report