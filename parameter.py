# Ideally, do not change the values of the default parameters. Create separate cases with unique <task-id> as seen in
# the code below (if-else loop) and use them. This way you can easily reproduce a configuration on a later time.

import os


def get_params(argv):
    dirname = os.path.dirname(__file__)
    print("SET: {}".format(argv))
    # ########### default parameters ##############
    params = dict(
        quick_test=True,     # To do quick test. Trains/test on small subset of dataset, and # of epochs

        # INPUT PATH
        dataset_dir = os.path.join(dirname, 'dataset/'),   # Base folder containing the foa/mic and metadata folders

        # OUTPUT PATH
        feat_label_dir = os.path.join(dirname, 'dataset/feat_label/'),  # Directory to dump extracted features and labels
        model_dir = os.path.join(dirname, 'models/'),   # Dumps the trained models and training curves in this folder
        dcase_output=False,     # If true, dumps the results recording-wise in 'dcase_dir' path.
                               # Set this true after you have finalized your model, save the output, and submit
        dcase_dir=os.path.join(dirname, 'results/'),  # Dumps the recording-wise network output in this folder

        # DATASET LOADING PARAMETERS
        mode='dev',         # 'dev' - development or 'eval' - evaluation dataset
        dataset='foa',       # 'foa' - ambisonic or 'mic' - microphone signals

        # DNN MODEL PARAMETERS
        sequence_length=128,        # Feature sequence length
        batch_size=16,              # Batch size
        dropout_rate=0.2,             # Dropout rate, constant for all layers
        nb_cnn2d_filt=[64, 128, 256],           # Number of CNN nodes, constant for each layer
        f_pool_size=[4, 4, 2],        # CNN pooling, length of list = number of CNN layers, list value = pooling per layer
        t_pool_size=[1, 1, 1],        # CNN time pooling
        rnn_size=[256, 256],        # RNN contents, length of list = number of layers, list value = number of nodes
        fnn_size=[256, 128],             # FNN contents, length of list = number of layers, list value = number of nodes
        loss_weights=[1., 50.],     # [sed, doa] weight for scaling the DNN outputs
        nb_epochs=100,               # Train for maximum epochs
        epochs_per_fit=5,           # Number of epochs per fit

    )
    params['patience'] = int(0.1 * params['nb_epochs'])     # Stop training if patience is reached

    # ########### User defined parameters ##############
    if argv == '1':
        print("USING DEFAULT PARAMETERS\n")

    elif argv == '2':
        print("QUICK TEST MODE\n")
        params['quick_test'] = True

    else:
        print('ERROR: unknown argument {}'.format(argv))
        exit()

    for key, value in params.items():
        print("\t{}: {}".format(key, value))
    return params
