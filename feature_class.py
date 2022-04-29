# Contains routines for labels creation, features extraction and normalization
#


import os
import numpy as np
import scipy.io.wavfile as wav
from sklearn import preprocessing
import joblib
import matplotlib.pyplot as plot
import librosa
import tensorflow as tf
plot.switch_backend('agg')


class FeatureClass:
    def __init__(self, dataset_dir='', feat_label_dir='', dataset='foa', is_eval=False):
        """

        :param dataset: string, dataset name, supported: foa - ambisonic or mic- microphone format
        :param is_eval: if True, does not load dataset labels.
        """

        # Input directories
        self._feat_label_dir = feat_label_dir
        self._dataset_dir = dataset_dir
        self._dataset_combination = '{}_{}'.format(dataset, 'eval' if is_eval else 'dev')
        self._aud_dir = os.path.join(self._dataset_dir, self._dataset_combination)

        self._desc_dir = None if is_eval else os.path.join(self._dataset_dir, 'metadata_dev')

        # Output directories
        self._label_dir = None
        self._feat_dir = None
        self._feat_dir_norm = None

        # Local parameters
        self._is_eval = is_eval

        # This is the sample rate of the audio files (number of values in one second)
        # So audio file has length_in_seconds x sample_rate values
        self._fs = 48000

        self._hop_len_s = 0.02 # 20ms --> When generating feature, move window from left to right using this hop length
        self._hop_len = int(self._fs * self._hop_len_s) # 960 --> number of samples in the 0.02 hop length (this is the hop length in samples instead of seconds)
        self._frame_res = self._fs / float(self._hop_len) # 50 --> number of hops in one second of audio
        self._nb_frames_1s = int(self._frame_res)

        self._win_len = 2 * self._hop_len # 1920 --> length of the window to process per time (over_lap = window - hop_length)
        self._nfft = self._next_greater_power_of_2(self._win_len) # 2048 -->  CONFUSED --> length of the windowed signal after padding with zeros (win_len <= n_fft)

        self._nb_mel_bins=64
        self._mel_wts = librosa.filters.mel(sr=self._fs, n_fft=self._nfft, n_mels=self._nb_mel_bins).T

        self._dataset = dataset
        self._eps = np.spacing(np.float(1e-16))
        self._nb_channels = 4  # number of audio channels

        # Sound event classes dictionary # DCASE 2016 Task 2 sound events
        self._unique_classes = dict()
        self._unique_classes = \
            {
                'clearthroat': 2,
                'cough': 8,
                'doorslam': 9,
                'drawer': 1,
                'keyboard': 6,
                'keysDrop': 4,
                'knock': 0,
                'laughter': 10,
                'pageturn': 7,
                'phone': 3,
                'speech': 5
            }

        self._doa_resolution = 10 # doas values are in intervals of 10 degress
        self._azi_list = range(-180, 180, self._doa_resolution)  # list of all possible azimuth values
        self._length = len(self._azi_list)  # number of all possible azimuth value
        self._ele_list = range(-40, 50, self._doa_resolution) # list of all possible elevation values

        self._audio_max_len_samples = 60 * self._fs # 2880000

        # For regression task only
        self._default_azi = 180
        self._default_ele = 50

        if self._default_azi in self._azi_list:
            print('ERROR: chosen default_azi value {} should not exist in azi_list'.format(self._default_azi))
            exit()
        if self._default_ele in self._ele_list:
            print('ERROR: chosen default_ele value {} should not exist in ele_list'.format(self._default_ele))
            exit()

        self._max_frames = int(np.ceil(self._audio_max_len_samples / float(self._hop_len))) # 3000 -> maximum number of frames expected in our extracted features

    def _load_audio(self, audio_path):
        fs, audio = wav.read(audio_path) # read audio
        audio = audio[:, :self._nb_channels] / 32768.0 + self._eps # normalize values
        if audio.shape[0] < self._audio_max_len_samples:
            zero_pad = np.zeros((self._audio_max_len_samples - audio.shape[0], audio.shape[1]))
            audio = np.vstack((audio, zero_pad)) # pad audio to make it as long as the maximum allowed length
        elif audio.shape[0] > self._audio_max_len_samples:
            audio = audio[:self._audio_max_len_samples, :] # extract maximum allowed length from the audio file
        return audio, fs

    # INPUT FEATURES
    @staticmethod
    def _next_greater_power_of_2(x):
        return 2 ** (x - 1).bit_length()

    def _spectrogram(self, audio_input):
        _nb_ch = audio_input.shape[1] # number of channels in audio recording
        nb_bins = self._nfft // 2
        spectra = np.zeros((self._max_frames, nb_bins + 1, _nb_ch), dtype=complex)
        for ch_cnt in range(_nb_ch):
            stft_ch = librosa.core.stft(np.asfortranarray(audio_input[:, ch_cnt]), n_fft=self._nfft, hop_length=self._hop_len,
                                        win_length=self._win_len, window='hann') # get spectrogram for each audio channel
            spectra[:, :, ch_cnt] = stft_ch[:, :self._max_frames].T
        return spectra

    def _get_spectrogram_for_file(self, audio_filename):
        audio_in, fs = self._load_audio(os.path.join(self._aud_dir, audio_filename))
        audio_spec = self._spectrogram(audio_in)
        return audio_spec

    def _get_mel_spectrogram(self, linear_spectra):
        mel_feat = np.zeros((linear_spectra.shape[0], self._nb_mel_bins, linear_spectra.shape[-1]))
        for ch_cnt in range(linear_spectra.shape[-1]):
            mag_spectra = np.abs(linear_spectra[:, :, ch_cnt])**2
            mel_spectra = np.dot(mag_spectra, self._mel_wts)
            log_mel_spectra = librosa.power_to_db(mel_spectra)
            mel_feat[:, :, ch_cnt] = log_mel_spectra
        mel_feat = mel_feat.reshape((linear_spectra.shape[0], self._nb_mel_bins * linear_spectra.shape[-1]))
        return mel_feat

    def _get_foa_intensity_vectors(self, linear_spectra):
        IVx = np.real(np.conj(linear_spectra[:, :, 0]) * linear_spectra[:, :, 3])
        IVy = np.real(np.conj(linear_spectra[:, :, 0]) * linear_spectra[:, :, 1])
        IVz = np.real(np.conj(linear_spectra[:, :, 0]) * linear_spectra[:, :, 2])

        normal = np.sqrt(IVx**2 + IVy**2 + IVz**2) + self._eps
        IVx = np.dot(IVx / normal, self._mel_wts)
        IVy = np.dot(IVy / normal, self._mel_wts)
        IVz = np.dot(IVz / normal, self._mel_wts)

        # we are doing the following instead of simply concatenating to keep the processing similar to mel_spec and gcc
        foa_iv = np.dstack((IVx, IVy, IVz))
        foa_iv = foa_iv.reshape((linear_spectra.shape[0], self._nb_mel_bins * 3))
        if np.isnan(foa_iv).any():
            print('Feature extraction is generating nan outputs')
            exit()
        return foa_iv



    # OUTPUT LABELS
    def read_desc_file(self, desc_filename, in_sec=False):
        # extract labels from metadata file
        desc_file = {
            'class': list(), 'start': list(), 'end': list(), 'ele': list(), 'azi': list()
        }
        fid = open(desc_filename, 'r')
        next(fid)
        for line in fid:
            split_line = line.strip().split(',')
            desc_file['class'].append(split_line[0])
            if in_sec:
                # return onset-offset time in seconds
                desc_file['start'].append(float(split_line[1]))
                desc_file['end'].append(float(split_line[2]))
            else:
                # return onset-offset time in frames
                desc_file['start'].append(int(np.floor(float(split_line[1])*self._frame_res)))
                desc_file['end'].append(int(np.ceil(float(split_line[2])*self._frame_res)))
            desc_file['ele'].append(int(split_line[3]))
            desc_file['azi'].append(int(split_line[4]))
        fid.close()
        return desc_file

    def _get_doa_labels_regr(self, _desc_file):
        # create arrays for azimth and elevation angles. Insert default values
        azi_label = self._default_azi*np.ones((self._max_frames, len(self._unique_classes))) # 3000, 11
        ele_label = self._default_ele*np.ones((self._max_frames, len(self._unique_classes))) # 3000, 11
        for i, ele_ang in enumerate(_desc_file['ele']):
            start_frame = _desc_file['start'][i]
            end_frame = self._max_frames if _desc_file['end'][i] > self._max_frames else _desc_file['end'][i]
            azi_ang = _desc_file['azi'][i]
            class_ind = self._unique_classes[_desc_file['class'][i]]
            # if value is within permitted rangle then insert value into the frame range (rows) of the active class (column)
            if (azi_ang >= self._azi_list[0]) and (azi_ang <= self._azi_list[-1]) and (ele_ang >= self._ele_list[0]) and (ele_ang <= self._ele_list[-1]):
                azi_label[start_frame:end_frame + 1, class_ind] = azi_ang
                ele_label[start_frame:end_frame + 1, class_ind] = ele_ang
            else:
                print('bad_angle {} {}'.format(azi_ang, ele_ang))
        doa_label_regr = np.concatenate((azi_label, ele_label), axis=1) # return item of shape 3000, 22
        return doa_label_regr

    def _get_se_labels(self, _desc_file):
        # create an array of (3000, 11) with all values as 0
        se_label = np.zeros((self._max_frames, len(self._unique_classes)))

        # since we have the start and end frames for the active sound
        # we fill in those array spaces with 1 (rows of start to end frame, column of active class)
        for i, se_class in enumerate(_desc_file['class']):
            start_frame = _desc_file['start'][i]
            end_frame = self._max_frames if _desc_file['end'][i] > self._max_frames else _desc_file['end'][i]
            se_label[start_frame:end_frame + 1, self._unique_classes[se_class]] = 1
        return se_label

    def get_labels_for_file(self, _desc_file):
        """
        Reads description csv file and returns classification based SED labels and regression based DOA labels

        :param _desc_file: csv file
        :return: label_mat: labels of the format [sed_label, doa_label],
        where sed_label is of dimension [nb_frames, nb_classes] which is 1 for active sound event else zero
        where doa_labels is of dimension [nb_frames, 2*nb_classes], nb_classes each for azimuth and elevation angles,
        if active, the DOA values will be in degrees, else, it will contain default doa values given by
        self._default_ele and self._default_azi
        """

        se_label = self._get_se_labels(_desc_file)
        doa_label = self._get_doa_labels_regr(_desc_file)

        # place SE and DOA arrays side by side (because they have the same number of rows -> max_frame)
        label_mat = np.concatenate((se_label, doa_label), axis=1)
        return label_mat # 3000, 33

     # ------------------------------- EXTRACT FEATURE AND PREPROCESS IT -------------------------------
    def extract_all_feature(self):
        # setting up folders
        self._feat_dir = self.get_unnormalized_feat_dir()
        create_folder(self._feat_dir)

        # extraction starts
        print('Extracting spectrogram:')
        print('\t\taud_dir {}\n\t\tdesc_dir {}\n\t\tfeat_dir {}'.format(
            self._aud_dir, self._desc_dir, self._feat_dir))

        for file_cnt, file_name in enumerate(os.listdir(self._aud_dir)):
            wav_filename = '{}.wav'.format(file_name.split('.')[0])
            spect = self._get_spectrogram_for_file(wav_filename)

            #extract mel
            mel_spect = self._get_mel_spectrogram(spect) # 3000, 256

            # extract intensity vectors
            foa_iv = self._get_foa_intensity_vectors(spect)  # 3000, 192
            feat = np.concatenate((mel_spect, foa_iv), axis=-1)

            print('{}: {}, {}'.format(file_cnt, file_name, feat.shape )) # feat.shape --> 3000, 448 --> (3000 frames with 448 features)
            np.save(os.path.join(self._feat_dir, '{}.npy'.format(wav_filename.split('.')[0])), feat)

    def preprocess_features(self):
        # Setting up folders and filenames
        self._feat_dir = self.get_unnormalized_feat_dir()
        self._feat_dir_norm = self.get_normalized_feat_dir()
        create_folder(self._feat_dir_norm)
        normalized_features_wts_file = self.get_normalized_wts_file()
        spec_scaler = None

        # pre-processing starts
        if self._is_eval:
            spec_scaler = joblib.load(normalized_features_wts_file)
            print('Normalized_features_wts_file: {}. Loaded.'.format(normalized_features_wts_file))

        else:
            print('Estimating weights for normalizing feature files:')
            print('\t\tfeat_dir: {}'.format(self._feat_dir))

            spec_scaler = preprocessing.StandardScaler()
            for file_cnt, file_name in enumerate(os.listdir(self._feat_dir)):
                print('{}: {}'.format(file_cnt, file_name))
                feat_file = np.load(os.path.join(self._feat_dir, file_name))

                # fit -> get mean and std from all data set
                # part_fit -> we already have mean and std but also consider this value in the calculationn
                #             It is helpful when saving memory, so that we dont have to load all the data at once
                spec_scaler.partial_fit(feat_file)
                del feat_file
            joblib.dump(
                spec_scaler,
                normalized_features_wts_file
            )
            print('Normalized_features_wts_file: {}. Saved.'.format(normalized_features_wts_file))

        print('Normalizing feature files:')
        print('\t\tfeat_dir_norm {}'.format(self._feat_dir_norm))
        for file_cnt, file_name in enumerate(os.listdir(self._feat_dir)):
            print('{}: {}'.format(file_cnt, file_name))
            feat_file = np.load(os.path.join(self._feat_dir, file_name))
            feat_file = spec_scaler.transform(feat_file)
            np.save(
                os.path.join(self._feat_dir_norm, file_name),
                feat_file
            )
            del feat_file

        print('normalized files written to {}'.format(self._feat_dir_norm))

    def augment_mel(self):
        # setting up folders
        self._feat_dir = self.get_unnormalized_feat_dir()
        self._feat_dir_norm = self.get_normalized_feat_dir()

        # setup normalization files
        normalized_features_wts_file = self.get_normalized_wts_file()
        spec_scaler = joblib.load(normalized_features_wts_file)

        print('--- Augmentation started ---')

        sample = ()
        all_feature_files = os.listdir(self._feat_dir)
        # run data augmentations equally across all splits
        count = 0
        for i in range(0, 4):
            split_feature_files = list(filter(lambda x: (x[5] == "{}".format(i + 1)), all_feature_files))
            aug_arr = np.random.randint(0, len(split_feature_files) - 1, size = len(split_feature_files)//4) # array of random indexes to augment
            for file_cnt, file_index in enumerate(aug_arr):
                count += 1
                file_name = split_feature_files[file_index]
                feat_file = np.load(os.path.join(self._feat_dir, file_name))
                # print("{}. {} was augmented and duplicated".format(file_cnt + 1, feature_files[file_index]))

                # separate features
                mel_spec = feat_file[:, :(self._nb_mel_bins * self._nb_channels)]
                foa_iv = feat_file[:, (self._nb_mel_bins * self._nb_channels):]

                # augment mel_spec
                mel_spec_after_freq_mk = self.freq_mask(mel_spec, param=20)
                mel_spec_after_time_mk = self.time_mask(mel_spec_after_freq_mk, param=20)

                # join features
                feat = np.concatenate((mel_spec_after_time_mk, foa_iv), axis=-1)

                # scale and save normlized features
                feat = spec_scaler.transform(feat)
                new_file_name = "{}_{}.npy".format(file_name.split('.')[0], "aug")
                np.save(os.path.join(self._feat_dir_norm, new_file_name),feat)

                # save accompanying label
                non_aug_label = np.load(os.path.join(self.get_label_dir(), file_name)) # load
                np.save(os.path.join(self.get_label_dir(), new_file_name), non_aug_label) # save with new name

                # return mel spectrogram of last augmented features
                if (i == 3 and file_cnt == len(aug_arr) - 1):
                    sample = (np.transpose(mel_spec), np.transpose(mel_spec_after_time_mk.numpy()))
            print ("Number of created augmented files in split {}: {}".format(i+1, len(aug_arr)))
        return sample

    def freq_mask(self, input, param, name=None):
        """
        Apply masking to a spectrogram in the freq domain.
        """
        input = tf.convert_to_tensor(input)
        freq_max = tf.shape(input)[1] # 256
        for i in range(0, 2):
            f = tf.random.uniform(shape=(), minval=0, maxval=param, dtype=tf.dtypes.int32)
            f0 = tf.random.uniform(
                shape=(), minval=0, maxval=freq_max - f, dtype=tf.dtypes.int32
            )
            indices = tf.reshape(tf.range(freq_max), (1, -1))
            condition = tf.math.logical_and(
                tf.math.greater_equal(indices, f0), tf.math.less(indices, f0 + f)
            )
            input = tf.where(condition, tf.cast(0, input.dtype), input)
        return input

    def time_mask(self, input, param, name=None):
        """
        Apply masking to a spectrogram in the time domain.
        """
        input = tf.convert_to_tensor(input)
        time_max = tf.shape(input)[0]
        for i in range(0, 2):
            t = tf.random.uniform(shape=(), minval=0, maxval=param, dtype=tf.dtypes.int32)
            t0 = tf.random.uniform(
                shape=(), minval=0, maxval=time_max - t, dtype=tf.dtypes.int32
            )
            indices = tf.reshape(tf.range(time_max), (-1, 1))
            condition = tf.math.logical_and(
                tf.math.greater_equal(indices, t0), tf.math.less(indices, t0 + t)
            )
            input = tf.where(condition, tf.cast(0, input.dtype), input)
        return input


    # ------------------------------- EXTRACT LABELS AND PREPROCESS IT -------------------------------
    def extract_all_labels(self):
        self._label_dir = self.get_label_dir()

        print('Extracting labels:')
        print('\t\taud_dir {}\n\t\tdesc_dir {}\n\t\tlabel_dir {}'.format(
            self._aud_dir, self._desc_dir, self._label_dir))
        create_folder(self._label_dir)

        for file_cnt, file_name in enumerate(os.listdir(self._desc_dir)):
            print('{}: {}'.format(file_cnt, file_name))
            wav_filename = '{}.wav'.format(file_name.split('.')[0])
            desc_file = self.read_desc_file(os.path.join(self._desc_dir, file_name))
            label_mat = self.get_labels_for_file(desc_file)
            np.save(os.path.join(self._label_dir, '{}.npy'.format(wav_filename.split('.')[0])), label_mat)



    # ------------------------------- Misc public functions -------------------------------
    def get_classes(self):
        return self._unique_classes

    def get_normalized_feat_dir(self):
        return os.path.join(
            self._feat_label_dir,
            '{}_norm'.format(self._dataset_combination)
        )

    def get_unnormalized_feat_dir(self):
        return os.path.join(
            self._feat_label_dir,
            '{}'.format(self._dataset_combination)
        )

    def get_label_dir(self):
        if self._is_eval:
            return None
        else:
            return os.path.join(
                self._feat_label_dir, '{}_label'.format(self._dataset_combination)
            )

    def get_normalized_wts_file(self):
        return os.path.join(
            self._feat_label_dir,
            '{}_wts'.format(self._dataset)
        )

    def get_default_azi_ele_regr(self):
        return self._default_azi, self._default_ele

    def nb_frames_1s(self):
        return self._nb_frames_1s

    def get_nb_mel_bins(self):
        return self._nb_mel_bins
    
    def get_sample_rate(self):
        return self._fs
    
    def get_hop_len(self):
        return self._hop_len

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        print('{} folder does not exist, creating it.'.format(folder_name))
        os.makedirs(folder_name)