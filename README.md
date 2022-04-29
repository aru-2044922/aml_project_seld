
# Sound event localization and detection (SELD) task
[Sound event localization and detection (SELD)](https://www.aane.in/research/computational-audio-scene-analysis-casa/sound-event-localization-detection-and-tracking) is the combined task of identifying the temporal onset and offset of a sound event, tracking the spatial location when active, and further associating a textual label describing the sound event.
   
## Getting Started

This repository consists of multiple Python scripts forming one big architecture used to train the SELD network.
* The `batch_feature_extraction.py` is a standalone wrapper script, that extracts the features, labels, and normalizes the training and test split features for a given dataset. Make sure you update the location of the downloaded datasets before.
* The `parameter.py` script consists of all the training, model, and feature parameters. If a user has to change some parameters, they have to create a sub-task with unique id here. Check code for examples.
* The `feature_class.py` script has routines for labels creation, features extraction and normalization.
* The `data_generator.py` script provides feature + label data in generator mode for training.
* The `crnn_model.py` script implements the SELDnet architecture.
* The `evaluation_metrics.py` script, implements the core metrics from sound event detection evaluation module http://tut-arg.github.io/sed_eval/ and the DOA metrics explained in the paper.
* The `seld.py` is a wrapper script that trains the SELDnet. The training stops when the SELD error (check paper) stops improving.


### Training the SELDnet

In order to quickly train the network follow the steps below.

* Update the dataset path in `parameter.py` script. For the above example, you will change `dataset_dir='base_folder/'`. Also provide a directory path `feat_label_dir` in the same `parameter.py` script where all the features and labels will be dumped.

* Extract features from the downloaded dataset by running the `batch_feature_extraction.py` script. First, update the parameters in the script, check the python file for more comments. You can now run the script as shown below. This will dump the normalized features and labels here. Since feature extraction is a one-time thing, this script is standalone and does not use the `parameter.py` file.

```
python batch_feature_extraction.py
```

You can now train the network using default parameters using
```
python seld.py
```

* Additionally, you can add/change parameters by using a unique identifier \<task-id\> in if-else loop as seen in the `parameter.py` script and call them as following
```
python seld.py <task-id> <job-id>
```
Where \<job-id\> is a unique identifier which is used for output filenames (models, training plots). You can use any number or string for this.
```
