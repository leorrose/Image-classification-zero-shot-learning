# Image-classification-zero-shot-learning

A deep neural network model with zero shot learning based on cifar100 data set.

The target of this Project is to create a model to classify images, even images not included in the training data set. 

This Project was created with <b> Python(3.8.7), tensorflow, keras, pandas, numpy and more libraries</b>.

## Project Research

In order to understand the steps and what we did you are welcome to look at [the research jupyter notebook](https://github.com/leorrose/Image-classification-ZSL/blob/master/research_notebook.ipynb).

## Project Setup and Run:
1. Clone this repository.
2. Open cmd/shell/terminal and go to project folder: `cd Image-classification-zero-shot-learning`
3. Install project dependencies: `pip install -r requirements.txt`
4. Run the python script with input image: `python  ./src/image_classification.py python "path_to_img"`
5. Enjoy the application.

## Examples:
| | |
|:-------------------------:|:-------------------------:|
|<p align="center"><img src="https://github.com/leorrose/Image-classification-zero-shot-learning/blob/master/demo_images/chimpanzee.jpg" width="400" hieght="400" alt="chimpanzee"/></p>|<p align="center"><img src="https://github.com/leorrose/Image-classification-zero-shot-learning/blob/master/demo_images/computer_mouse.jpg" width="400" hieght="400" alt="computer mouse"/></p>|
|<p align="center">Prediction: chimpanzee, chimp, monkey, baboon, orangutan</p>| <p align="center">Prediction: telephone, phone, telephon, telephones, land-line</p>|
|<p align="center"><img src="https://github.com/leorrose/Image-classification-zero-shot-learning/blob/master/demo_images/petunia.jpg" width="400" hieght="400" alt="petunia"/></p>|<p align="center"><img src="https://github.com/leorrose/Image-classification-zero-shot-learning/blob/master/demo_images/rhino.jpg" width="400" hieght="400" alt="rhino"/></p>|
|<p align="center">Prediction: rose, flower, tulip, carnation, marigold</p>|<p align="center">Prediction: elephant, tiger, lion, tusker, leopard</p>|
|<p align="center"><img src="https://github.com/leorrose/Image-classification-zero-shot-learning/blob/master/demo_images/starfish.jpg" width="400" hieght="400" alt="starfish"/></p>|<p align="center"><img src="https://github.com/leorrose/Image-classification-zero-shot-learning/blob/master/demo_images/tansy.jpg" width="400" hieght="400" alt="tansy"/></p>|
|<p align="center">Prediction: woodlouse, snake, crab, leatherjacket, blobfish</p>|<p align="center">Prediction: orange, purple, yellow, pink, red</p>|





Please let me know if you find bugs or something that needs to be fixed.

Hope you enjoy.

## Citations

```sh
@inproceedings{mikolov2018advances,
  title={Advances in Pre-Training Distributed Word Representations},
  author={Mikolov, Tomas and Grave, Edouard and Bojanowski, Piotr and Puhrsch, Christian and Joulin, Armand},
  booktitle={Proceedings of the International Conference on Language Resources and Evaluation (LREC 2018)},
  year={2018}
}
```
