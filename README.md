# Image-classification-ZSL

A CNN model with zero shot learning based on cifar100 data set.

The target of this Project is to create a CNN model to classify images, even images not included in the training data set. 

This Project was created with <b> Python, tensorflow, keras, pandas, numpy and more liberais</b>. 

### Project Setup (Windows):

1. make sure you have python on your computer (if not install python 3.6.1 from here [Python download](https://www.python.org/downloads/windows/))
2. Make sure Python is in path (if not follow this guide [Add python to path](https://datatofish.com/add-python-to-windows-path/))
3. Make sure pip is in path (if not follow this guide [Add pip to path](https://appuals.com/fix-pip-is-not-recognized-as-an-internal-or-external-command/))
5. Clone repository.
6. Run installationWin.bat and wait until console closes.
7. That’s it, you are all set up to run.

### Project Setup (Linux):

1. make sure you have python on your computer (if not install python from here [Python download](https://docs.python-guide.org/starting/install3/linux/))
3. Make sure you have pip on your computer (if not follow this guide [pip download]](https://itsfoss.com/install-pip-ubuntu/))
5. Clone repository.
6. Run installationLinux.sh and wait until console closes.
7. That’s it, you are all set up to run.

### Project Run:

#### Train:
In the console run:
1. cd src (go to src directory).
2. python image_classification.py --task train --model 'model_saving_name.h5'
the created model will be saved in src\model_saving_name.h5

#### Test:
In the console run:
1. cd src (go to src directory).
2. python image_classification.py --task test --image 'image_path_to_clasify' --model 'model_saving_name.h5'
the 5 classification results will be printed to console.

### Examples of tests:

<p align="center"><img src="https://github.com/leorrose/Image-classification-ZSL/blob/master/test%20images/forest.jpg" width="400" hieght="400" alt="forest"/></p>
<p align="center">Prediction: insect, species, snake, flower, spider</p>

<p align="center"><img src="https://github.com/leorrose/Image-classification-ZSL/blob/master/test%20images/car.jpg" width="400" hieght="400" alt="car"/></p>
<p align="center">Prediction: motorcycle, truck, bicycle, tractor, bike</p>

<p align="center"><img src="https://github.com/leorrose/Image-classification-ZSL/blob/master/test%20images/bycicle.jpg" width="400" hieght="400" alt="bycicle"/></p>
<p align="center">Prediction: motorcycle, bicycle, bike, motorbike, motorcycles</p>

<p align="center"><img src="https://github.com/leorrose/Image-classification-ZSL/blob/master/test%20images/lynx.jpg" width="400" hieght="400" alt="lynx"/></p>
<p align="center">Prediction: tiger, cat, car, vehicle, motorcycle</p>


Please let me know if you find bugs or something that needs to be fixed.

Hope you enjoy.

