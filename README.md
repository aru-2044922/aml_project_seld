
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

In order to quickly train SELDnet follow the steps below.

* For the chosen dataset (Ambisonic or Microphone), download the respective zip file. This contains both the audio files and the respective metadata. Unzip the files under the same 'base_folder/', ie, if you are Ambisonic dataset, then the 'base_folder/' should have two folders - 'foa_dev/' and 'metadata_dev/' after unzipping.

* Now update the respective dataset path in `parameter.py` script. For the above example, you will change `dataset_dir='base_folder/'`. Also provide a directory path `feat_label_dir` in the same `parameter.py` script where all the features and labels will be dumped. Make sure this folder has sufficient space. For example if you use the baseline configuration, you will need about 160 GB in total just for the features and labels.

* Extract features from the downloaded dataset by running the `batch_feature_extraction.py` script. First, update the parameters in the script, check the python file for more comments. You can now run the script as shown below. This will dump the normalized features and labels here. Since feature extraction is a one-time thing, this script is standalone and does not use the `parameter.py` file.

```
python batch_feature_extraction.py
```

You can now train the SELDnet using default parameters using
```
python seld.py
```

* Additionally, you can add/change parameters by using a unique identifier \<task-id\> in if-else loop as seen in the `parameter.py` script and call them as following
```
python seld.py <task-id> <job-id>
```
Where \<job-id\> is a unique identifier which is used for output filenames (models, training plots). You can use any number or string for this.

In order to get baseline results on the development set for Microphone array recordings, you can run the following command
```
python seld.py 2
```
Similarly, for Ambisonic format baseline results, run the following command
```
python seld.py 4
```

* By default, the code runs in `quick_test = True` mode. This trains the network for 2 epochs on only 2 mini-batches. Once you get to run the code sucessfully, set `quick_test = False` in `parameter.py` script and train on the entire data.

* The code also plots training curves, intermediate results and saves models in the `model_dir` path provided by the user in `parameter.py` file.

* In order to visualize the output of SELDnet and for submission of results, set `dcase_output=True` and provide `dcase_dir` directory. This will dump file-wise results in the directory, which can be individually visualized using `misc_files/visualize_SELD_output.py` script.

* Finally, the average development dataset score across the four folds can be obtained using `calculate_SELD_metrics.py` script. Provide the directory where you dumped the file-wise results above and the reference metadata folder. Check the comments in the script for more description.

## Results on development dataset


| Dataset | Error rate | F score| DOA error | Frame recall |
| ----| --- | --- | --- | --- |
| Ambisonic | 0.34 | 79.9 % | 28.5&deg; | 85.4 % |
| Microphone Array |0.35 | 80.0 % | 30.8&deg; | 84.0 % |

**Note:** The reported baseline system performance is not exactly reproducible due to varying setups. However, you should be able to obtain very similar results.

## DOA estimation: regression vs classification

The DOA estimation can be approached as both a regression or a classification task. In the baseline, it is handled as regression task. In case you plan to use a classification approach check the `test_SELD_metrics.py` script in misc_files folder. It implements a classification version of DOA and also uses a corresponding metric function.


## Submission

* Before submission, make sure your SELD results are correct by visualizing the results using `misc_files/visualize_SELD_output.py` script
* Make sure the file-wise output you are submitting is produced at 20 ms hop length. At this hop length a 60 s audio file has 3000 frames.
* Calculate your development score for the four splits using the `calculate_SELD_metrics.py` script. Check if the average results you are obtaining here is comparable to the results you were obtaining during training.

For more information on the submission file formats [check the website](http://dcase.community/challenge2019/task-sound-event-localization-and-detection#submission)

## License

Except for the contents in the `metrics` folder that have [MIT License](metrics/LICENSE.md). The rest of the repository is licensed under the [TAU License](LICENSE.md).

## Acknowledgments

The research leading to these results has received funding from the European Research Council under the European Unions H2020 Framework Programme through ERC Grant Agreement 637422 EVERYSOUND.
