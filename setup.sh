#!/usr/bin/env bash
eval "$(conda shell.bash hook)"

# create env
conda create -n reatt python=3.7

# activate env
conda activate reatt

# install dependencies
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install -c conda-forge faiss-gpu cudatoolkit=11.1
conda install -c conda-forge six

pip install transformers==4.15.0
pip install https://data.pyg.org/whl/torch-1.8.0%2Bcu111/torch_scatter-2.0.7-cp37-cp37m-linux_x86_64.whl
pip install beir
pip install wandb==0.12.10
pip install bertviz
pip install jupyterlab==3.2.9
pip install ipywidgets==7.6.5
pip install rouge-score==0.0.4
pip install entmax==1.0
pip install datasets==1.18.3
pip install spacy==3.2.3
pip install ujson==5.2.0
pip install cupy-cuda111==10.3.1
pip install indexed==1.2.1
pip install protobuf==3.20.*
