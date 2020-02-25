# BREIN Deep Leaning Challenge

By [Arian Gallardo](http://github.com/ariangc).

Pontificia Universidad Catolica del Peru (PUCP).

### Tabla de contenidos
0. [Introduccion](#introduccion)
0. [Problema presentado](#problema-presentado)
0. [Solucion](#solucion)
0. [Results](#results)
0. [API](#api)

### Introduction

Este repositorio contiene los archivos que corresponden a la solucion del reto presentado para el proceso de seleccion del Hub de Innovacion del Grupo Breca (BREIN), para Febrero 2020. 

### Problema presentado

En este reto se busca ayudar a una empresa retail a mejorar su proceso de manejo de inventarios. La empresa esta buscando una manera de reducir el esfuerzo humano en la clasificacion de sus productos.

A traves de un clasificador de imagenes, podemos ayudar a la compania a analizar su inventario.

Todos los datos se pueden encontrar en el siguiente enlace: [Data Reto Brein](https://www.dropbox.com/s/kub6cebbsgiotla/reto_deep_learning.rar?dl=0)

### Solucion

Se uso el modelo ResNet-18 descrito en el paper "Deep Residual Learning for Image Recognition" (http://arxiv.org/abs/1512.03385). Este modelo, asi como la familia de ResNets, fueron usados en [ILSVRC](http://image-net.org/challenges/LSVRC/2015/) y [COCO](http://mscoco.org/dataset/#detections-challenge2015), competencias de Computer Vision en 2015, en los que ganaron el 1er puesto en: Clasificacion de ImageNet, Deteccion de ImageNet, Localizacion de Imagenet, Deteccion en COCO, y Segmentacion en COCO.

### Resultados

0. Curvas de accuracy sobre el dataset (20 epocas, learning rate = 0.0001, optimizador = Adam)
	![Training acc curves](https://raw.githubusercontent.com/ariangc/breinchallenge/master/models/pytorch_resnet18/train_val_acc.jpg)

0. Curvas de loss sobre el dataset (20 epocas, learning rate = 0.0001, optimizador = Adam)
	![Training loss curves](https://raw.githubusercontent.com/ariangc/breinchallenge/master/models/pytorch_resnet18/train_val_loss.jpg)


# Reemplazar todo lo demas

### Models

0. Visualizations of network structures (tools from [ethereon](http://ethereon.github.io/netscope/quickstart.html)):
	- [ResNet-50] (http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006)
	- [ResNet-101] (http://ethereon.github.io/netscope/#/gist/b21e2aae116dc1ac7b50)
	- [ResNet-152] (http://ethereon.github.io/netscope/#/gist/d38f3e6091952b45198b)

0. Model files:
	- ~~MSR download: [link] (http://research.microsoft.com/en-us/um/people/kahe/resnet/models.zip)~~
	- OneDrive download: [link](https://onedrive.live.com/?authkey=%21AAFW2-FVoxeVRck&id=4006CBB8476FF777%2117887&cid=4006CBB8476FF777)

### Results
0. Curves on ImageNet (solid lines: 1-crop val error; dashed lines: training error):
	![Training curves](https://cloud.githubusercontent.com/assets/11435359/13046277/e904c04c-d412-11e5-9260-efc5b8301e2f.jpg)

0. 1-crop validation error on ImageNet (center 224x224 crop from resized image with shorter side=256):

	model|top-1|top-5
	:---:|:---:|:---:
	[VGG-16](http://www.vlfeat.org/matconvnet/pretrained/)|[28.5%](http://www.vlfeat.org/matconvnet/pretrained/)|[9.9%](http://www.vlfeat.org/matconvnet/pretrained/)
	ResNet-50|24.7%|7.8%
	ResNet-101|23.6%|7.1%
	ResNet-152|23.0%|6.7%
	
0. 10-crop validation error on ImageNet (averaging softmax scores of 10 224x224 crops from resized image with shorter side=256), the same as those in the paper:

	model|top-1|top-5
	:---:|:---:|:---:
	ResNet-50|22.9%|6.7%
	ResNet-101|21.8%|6.1%
	ResNet-152|21.4%|5.7%
	
### Third-party re-implementations

Deep residual networks are very easy to implement and train. We recommend to see also the following third-party re-implementations and extensions:

0. By Facebook AI Research (FAIR), with **training code in Torch and pre-trained ResNet-18/34/50/101 models for ImageNet**: [blog](http://torch.ch/blog/2016/02/04/resnets.html), [code](https://github.com/facebook/fb.resnet.torch)
0. Torch, CIFAR-10, with ResNet-20 to ResNet-110, training code, and curves: [code](https://github.com/gcr/torch-residual-networks)
0. Lasagne, CIFAR-10, with ResNet-32 and ResNet-56 and training code: [code](https://github.com/Lasagne/Recipes/tree/master/papers/deep_residual_learning)
0. Neon, CIFAR-10, with pre-trained ResNet-32 to ResNet-110 models, training code, and curves: [code](https://github.com/apark263/cfmz)
0. Torch, MNIST, 100 layers: [blog](https://deepmlblog.wordpress.com/2016/01/05/residual-networks-in-torch-mnist/), [code](https://github.com/arunpatala/residual.mnist)
0. A winning entry in Kaggle's right whale recognition challenge: [blog](http://blog.kaggle.com/2016/02/04/noaa-right-whale-recognition-winners-interview-2nd-place-felix-lau/), [code](https://github.com/felixlaumon/kaggle-right-whale)
0. Neon, Place2 (mini), 40 layers: [blog](http://www.nervanasys.com/using-neon-for-scene-recognition-mini-places2/), [code](https://github.com/hunterlang/mpmz/)
0. MatConvNet, CIFAR-10, with ResNet-20 to ResNet-110, training code, and curves: [code](https://github.com/suhangpro/matresnet)
0. TensorFlow, CIFAR-10, with ResNet-32,110,182 training code and curves:
[code](https://github.com/ppwwyyxx/tensorpack/tree/master/examples/ResNet)
0. MatConvNet, reproducing CIFAR-10 and ImageNet experiments (supporting official MatConvNet), training code and curves: [blog](https://zhanghang1989.github.io/ResNet/), [code](https://github.com/zhanghang1989/ResNet-Matconvnet)
0. Keras, ResNet-50: [code](https://github.com/raghakot/keras-resnet)

Converters:

0. MatConvNet: [url](http://www.vlfeat.org/matconvnet/pretrained/#imagenet-ilsvrc-classification)
0. TensorFlow: [url](https://github.com/ry/tensorflow-resnet)

