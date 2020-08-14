
# ***Developing Patch Based Adversarial Attacks***

This repository allows one to generate, evaluate, and visualize a patch based adversarial attack for the Yolov3 object detector, a copy of which is included in this repository. Some of the code in this repository has been pulled directly from other repositories, as cited in my research report.

This is the code associated with my research, funded by Swarthmore College's SOAR-NSE undergraduate research program, and produced with assistance from Profesor Allan Moser.


## *About the Repository*
- The requirements.txt file contains a listing of all required packages
- The script adversarial_attack.py is the adversarial attack generator 
- The config.gin file contains the options for adversarial_attack.py 
- The data directory contains training data
- The results directory contains results from past trainings and will be populated as adversarial_attack.py is run
- The tf_logs directory is set up for tensorboard logging
- The yolov3 directory contains the yolov3 object detector that is attacked.

## Adversarial Patch

<p align="center">
  <img width="400" height="400" src="duplicate.png">
</p>


## Requirements

Python 3.7 or later with all `requirements.txt` dependencies installed. To install all dependencies run:
```bash
$ pip install -U -r requirements.txt
```

## Training
- Set the main logic command in the config.gin to 'train'
```bash
$ python adversarial_attack.py config.gin
```
<img src="training1.png" alt="Training" width="500"/>


## Visualization
- Set the main logic command in the config.gin to 'visualize'
```bash
$ python adversarial_attack.py config.gin
```
<img src="visualization1.png" alt="Visualization" width="1200"/>

## Evaluation
- Set the main logic command in the config.gin to 'evaluate'
- Set the secondary logic command if desired
```bash
$ python adversarial_attack.py config.gin
```
## Hyper Parameter Optimization
- Set the main logic command in the config.gin to 'optimize'
- Set the optimization parameters in the config.gin
```bash
$ python adversarial_attack.py config.gin
```
<img src="optimization1.png" alt="Optimization" width="300"/> <img src="optimization2.png" alt="Visualization" width="300"/> <img src="optimization3.png" alt="optimization" width="300"/>

## Questions?
Feel free to email me at ianmcdiarmidsterling at gmail dot com

## License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

- **[MIT license](http://opensource.org/licenses/mit-license.php)**

## Acknowledgments
Special thanks to:
- Professor Allan Moser
- David Sterling PhD


