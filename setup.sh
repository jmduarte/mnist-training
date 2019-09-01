conda create -n mnist-training python=3.6
source activate mnist-training
pip uninstall -y tensorflow
pip uninstall -y tf-nightly
pip install -q -U tf-nightly-gpu
pip install -q tensorflow-model-optimization
pip install matplotlib
pip install guildai
pip uninstall -y enum34
