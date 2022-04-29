
# Sound event localization and detection (SELD) task
[Sound event localization and detection (SELD)](https://www.aane.in/research/computational-audio-scene-analysis-casa/sound-event-localization-detection-and-tracking) is the combined task of identifying the temporal onset and offset of a sound event, tracking the spatial location when active, and further associating a textual label describing the sound event.
   
## Training the network
The network can be re-created and training using either the provided jupyter notebook or the python scripts, see further details below. Use the jupter notebook to run the tuning experiments used to choose the final architecture, and use the python scripts to train using the final architecture.

### Jupter Notebook
* In order to use the provided Jupyter Notebook, update the folder path in `working_dir` variable to reflect the folder that contains the dataset

* Run the cells sequentially to follow the experiments created in order to select the final architecture. The last cell trains using the final architecture.

### Python Scripts
In order to train the network with the final architecture, follow the steps below.

* Update the dataset path in `parameter.py` script. For the above example, you will change `dataset_dir='....'`. Also provide a directory path `feat_label_dir` in the same `parameter.py` script where all the features and labels will be dumped.

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
