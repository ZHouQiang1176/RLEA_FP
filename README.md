## Requirements

* [PyTorch](http://pytorch.org/)
* [OpenAI baselines](https://github.com/openai/baselines)
* [DREAMPlace](https://github.com/limbo018/DREAMPlace)

In order to install requirements, follow:

```bash
# PyTorch
conda install pytorch torchvision -c soumith

# Baselines for Atari preprocessing
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .

# Other requirements
pip install -r requirements.txt

# DREAMplace installation
git clone --recursive https://github.com/limbo018/DREAMPlace.git
mkdir build 
cd build 
cmake .. -DCMAKE_INSTALL_PREFIX=../.. -DPYTHON_EXECUTABLE=$(which python)
make 
make install

#Get benchmarks
python benchmarks/ispd2005_2015.py

# DGL installation
conda install -c dglteam dgl-cuda10.2
```

## Training

### Macro Placement

```bash
python main.py 


