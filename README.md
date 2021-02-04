# Image-classification-ZSL

A CNN model with zero shot learning based on cifar100 data set.

The target of this Project is to create a CNN model to classify images, even images not included in the training data set. 

This Project was created with <b> Python(3.8.7), tensorflow, keras, pandas, numpy and more liberais</b>. 

### Project Setup and Run:
1. Clone this repository.
2. Open cmd/shell/terminal and go to project folder: `cd Image-classification-ZSL`
3. Install project dependencies: `pip install -r requirements.txt`
4. Run the python script with input image: `python  ./src/image_classification.py python "path_to_img"`
5. Enjoy the application. 

### Examples:

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

### Citations

```sh
@inproceedings{mikolov2018advances,
  title={Advances in Pre-Training Distributed Word Representations},
  author={Mikolov, Tomas and Grave, Edouard and Bojanowski, Piotr and Puhrsch, Christian and Joulin, Armand},
  booktitle={Proceedings of the International Conference on Language Resources and Evaluation (LREC 2018)},
  year={2018}
}
```
