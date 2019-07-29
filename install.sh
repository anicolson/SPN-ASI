#!/bin/bash
virtualenv --system-site-packages -p python3 ~/venv/SPN-Spk-Rec
source ~/venv/SPN-Spk-Rec/bin/activate
pip install tensorflow
pip install -r .requirements.txt
