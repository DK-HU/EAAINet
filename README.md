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

# Datasets
We utilize two standard datasets (i.e, 7-Scenes and 12-Scenes) to evaluate our method.
- 7-Scenes: The 7-Scenes dataset can be downloaded from [7-Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)
- 12-Scenes: The 12-Scenes dataset can be downloaded from [12-Scenes](https://graphics.stanford.edu/projects/reloc/)

# Evaluation


# Acknowledgements
The PnP-RANSAC pose solver is referenced from [HSCNet](https://github.com/AaltoVision/hscnet/tree/master/pnpransac).


