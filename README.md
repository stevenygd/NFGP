# Geometry Processing with Neural Fields

Pytorch implementation for the **NeurIPS 2021** paper:

[Geometry Processing with Neural Fields](https://openreview.net/pdf?id=JG-SlCAx5_K)

[Guandao Yang](https://www.guandaoyang.com/), 
[Serge Belongie](https://blogs.cornell.edu/techfaculty/serge-belongie/),
[Bharath Hariharan](http://home.bharathh.info/),
[Vladlen Koltun](http://vladlen.info/)


## Introduction

Most existing geometry processing algorithms use meshes as the default shape representation.  
Manipulating meshes, however, requires one to maintain high quality in the surface discretization.  
For example, changing the topology of a mesh usually requires additional procedures such as remeshing. 
This paper instead proposes the use of neural fields for geometry processing. 
Neural fields can compactly store complicated shapes without spatial discretization. 
Moreover, neural fields are infinitely differentiable, which allows them to be optimized for objectives that involve higher-order derivatives. 
This raises the question: _can geometry processing be done entirely using neural fields?_ 
We introduce loss functions and architectures to show that some of the most challenging geometry processing tasks, such as deformation and filtering, can be done with neural fields. 
Experimental results show that our methods are on par with the well-established mesh-based methods without committing to a particular surface discretization.

## Installation 

This repository provides a [Anaconda](https://www.anaconda.com/) environment, and requires NVIDIA GPU to run the
 optimization routine. 
The environment can be set-up using the following commands:
```bash
conda env create -f environment.yml
conda activate NFGP
```

## Dataset

We offer pre-processed dataset for reproducing the experimental results in the paper. 
The instruciton is in section [_Download Dataset_](#download-dataset). 
If you want to run experiment for your own shape, you need to prepare data in the following two steps:
1. Create SDF samples for the shape (See section [_Create SDF Samples_](#create-sdf-samples))
2. Train a neural field to fit those samples (See section []()).
3. Depending on the tasks, create the user specified input. For deofmration task, please refer to section [_Create
 Deformation Handles_](#create-deformation-handles). For sharpening and smoothing, you can directly set it through
  the configuration file or the hyper-parameter.

#### Download Dataset


#### Create SDF Samples 

Following command creates SDF ground truth samples for obtaining the Neural Fields that approximates the SDF of the
 shape.
```bash
python scripts/prep_sdf_data.py <mesh_file_name> --out_path data/<mesh_file_name>

# Example
python scripts/prep_sdf_data.py armidillo.obj --out_path data/armidillo
```

The data preprocessing pipeline will sample 5M points uniformly withint `[-1, 1]^3` and another 5M near surface (i.e
. surface plus a Gaussian with standard deviation of `0.1`).
We then use the `mesh_to_sdf` package to compute the SDF of the points.
These points will be saved to following files:
- `<out_path>/mesh.obj` : a copy of the original mesh.
- `<out_path>/sdf.npy` : a numpy file that contains the sampled points, their SDF, and the original mesh.

Note that this pipeline works the best with watertight mesh.
This processing pipeline takes about 5 - 10 minutes to finish for a mesh with 30k faces.

#### Create Deformation Handles 

## 
**Detailed instruction, dataset, and pretrained models will be available before the conference.**
