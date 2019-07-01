#!/bin/bash

git clone https://github.com/SPFlow/SPFlow.git
git clone https://github.com/anicolson/DeepXi.git
virtualenv --system-site-packages -p python3 ~/venv/SPN-Spk-Rec
source ~/venv/SPN-Spk-Rec/bin/activate
pip install --upgrade tensorflow-gpu
pip install -r ./DeepXi/requirements.txt
