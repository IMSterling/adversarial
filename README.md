# ->Developing Patch Based Adversarial Attacks<-
## Swarthmore College 
##### Ian McDiarmid-Sterling

This repository allows one to generate, evaluate, and visualize a patch based adversarial attack for the Yolov3 object detector, a copy of which is included in this repository. Some of the code in this repository has been pulled directly from other repositories, as cited in my report.

This is the code associated with my report, generated with funding from Swarthmore College's SOAR-NSE undergraduate research program, and with assistance from Profesor Allan Moser 

--- About the repository ---

- The requirements.txt file contains all the modules that are needed for the scripts

- The script adversarial_attack.py is the adversarial attack generator and must be called with the config.gin file when run

- The config.gin file contains the options for adversarial_attack.py 

- The data directory contains training data

- The results directory contains results from past runs and will be populated as adversarial_attack.py is run

- The tf_logs directory is setup for tensorboard logging

- The yolov3 directory contains the yolov3 object detector that is attacked.


Enjoy ! 
Feel free to email me with questions -- ianmcdiarmidsterling at gmail dot com
