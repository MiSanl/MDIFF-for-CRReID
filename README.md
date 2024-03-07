# Multi Deep Invariant Feature Learning for Cross-Resolution Person Re-Identification

This repository contains the testing code associated with our paper titled "Multi Deep Invariant Feature Learning for Cross-Resolution Person Re-Identification". Our method presents a novel approach to person re-identification across different resolutions, achieving significant improvements.

## Project Overview

Our approach focuses on learning invariant features across different resolutions by our framework MDIFF.

Despite promising results, we acknowledge the potential for further improvement in our method, and we look forward to exploring these possibilities in future research.

## Code Usage

The testing code provided here allows for independent evaluation of our method's performance.

### 0. Requirements
You need `torch`, `numpy` and `pillow` installed in your environment.

### 1. Download dataset and pretrained model
Download the VIPeR dataset from this url: https://cloud.foreup.top/s/v8RiZ, which has been processed the origin VIPeR to Market1501 formate.
Download the pretrained weight from this url: https://cloud.foreup.top/s/14eIv.

### 2. Clone project
```shell
git clone https://github.com/MiSanl/MDIFF-for-CRReID.git
```

### 2. Run test
```
python test.py --data /path/to/your/dataset_root_dir --model_path /path/to/your/pretrained/model --gpu GPU_ID
```

