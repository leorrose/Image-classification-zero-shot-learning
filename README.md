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
**The image:**

<img src="https://github.com/leorrose/Image-classification-ZSL/blob/master/test%20images/forest.jpg" width="250" hieght="250" alt="forest"/>

**The prediction:** 

insect, species, snake, flower, spider

**The image:**

<img src="https://github.com/leorrose/Image-classification-ZSL/blob/master/test%20images/car.jpg" width="250" hieght="250" alt="car"/>

**The prediction:**

motorcycle, truck, bicycle, tractor, bike

**The image:**

<img src="https://github.com/leorrose/Image-classification-ZSL/blob/master/test%20images/bycicle.jpg" width="250" hieght="250" alt="bycicle"/>

**The prediction:**

motorcycle, bicycle, bike, motorbike, motorcycles

**The image:**

<img src="https://github.com/leorrose/Image-classification-ZSL/blob/master/test%20images/lynx.jpg" width="250" hieght="250" alt="lynx"/>

**The prediction:**

truck, car, vehicle, motorcycle, bicycle




