# Geometry Processing with Neural Fields

Pytorch implementation for the **NeurIPS 2021** paper [(project page)](https://www.guandaoyang.com/NFGP):

[Geometry Processing with Neural Fields](https://openreview.net/pdf?id=JG-SlCAx5_K)

[Guandao Yang](https://www.guandaoyang.com/), 
[Serge Belongie](https://blogs.cornell.edu/techfaculty/serge-belongie/),
[Bharath Hariharan](http://home.bharathh.info/),
[Vladlen Koltun](http://vladlen.info/)

![Teaser](docs/assets/teaser.png)



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
The code is tested in [CUDA 10.2](), ubuntu 16.04.
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
2. Train a neural field to fit those samples (See section [Create Input Neural Fields](#create-input-neural-fields)).
3. Depending on the tasks, create the user specified input. For deofmration task, please refer to section [_Create
 Deformation Handles_](#create-deformation-handles). For sharpening and smoothing, you can directly set it through
  the configuration file or the hyper-parameter.

#### Download Dataset

You can find the dataset in the following Google drive folder: [Google Drive](https://drive.google.com/drive/folders/1Hbl566qaJrbfDokPo5kgCv0djOutJB0R?usp=sharing). 
Alternatively, you can also download data using the following command:
```bash
# Download data to train Neural Fields
wget https://geometry-processing-with-neural-fields.s3.us-east-2.amazonaws.com/nf_data.zip
unzip nf_data.zip

# Download the deformation Data
wget https://geometry-processing-with-neural-fields.s3.us-east-2.amazonaws.com/deform_data.zipA
unzip deform_data.zip
```

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

#### Create Input Neural Fields 

Once you've obtained the `sdf.npy` file in the previous subsection,
you can use those data to train a neural field:
```bash
python train.py configs/recon/create_neural_fields.yaml --hparams data.path=<your_sdf.npy>
```
You can also create your own config following the examples in folder `configs/recon`.


#### Create Deformation Handles 

Please see `notebooks/deformation-handles-*.ipynb` for examples.


## Smoothing and Sharpening 

To run our methods on the smoothing or sharpening task, you can use the following configurations:
```bash
# Armadillo
python train.py configs/filtering/filtering_Armadillo_beta0.yaml  # smoothing
python train.py configs/filtering/filtering_Armadillo_beta2.yaml  # sharpening
# Noisy sphere
python train.py configs/filtering/filtering_HalfNoisySphere_beta0.yaml # smoothing
python train.py configs/filtering/filtering_HalfNoisySphere_beta2.yaml # sharpening
# Noisy torus
python train.py configs/filtering/filtering_NoisyTorus_beta0.yaml # smoothing
python train.py configs/filtering/filtering_NoisyTorus_beta2.yaml # sharpening
```

You can change the value of `beta` by adding `--hparams trainer.beta=<yourbeta>`.
To run it on a different shape, you need to change the `models.decoder` to load the appropriate neural fields.

## Deformation

To run the deformation experiments, you can use the following configurations:
```bash
# Jolteon
python train.py configs/deformation/jolteon_jump_s1e-1_b1e-3.yaml
python train.py configs/deformation/jolteon_nosedown_s1e-1_b1e-3.yaml
```

To change the amount of bending or stretching resistent, 
you can change the value of `trainer.loss_stretch.weight` and `trainer.loss_bend,weught` by
adding `--hparams trainer.loss_stretch.weight=<new_weight>` or `--hparams trainer.loss_bend.weight=<new_weight>`

## Citation 

If you find our paper or code useful, please cite us:
```
@inproceedings{yang2021geometry,
  title={Geometry Processing with Neural Fields},
  author={Yang, Guandao and Belongie, Serge and Hariharan, Bharath and Koltun, Vladlen},
  booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
  year={2021}
}
```

## Acknowledgement
Guandao’s PhD was supported in part by a research gift from Magic Leap and
a donation from NVIDIA. We want to thank Wenqi Xian, Professor Steve Marschner, and members
of Intel Labs for providing insightful feedback for this project.