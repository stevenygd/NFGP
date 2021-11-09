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


**Detailed instruction, dataset, and pretrained models will be available before the conference.**