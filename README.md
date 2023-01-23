This is the PyTorch implementation of our paper "EAAINet: An Element-wise Attention Network with Global Affinity Information for Accurate Indoor Visual Localization".

# Installation
Firstly, we need to set up python3 environment from requirement.txt:

```bash
pip3 install -r requirement.txt 
```


Subsequently, we need to build the cython module to install the PnP solver:
```bash
cd ./pnpransac
rm -rf build
python setup.py build_ext --inplace
```

# Acknowledgements

