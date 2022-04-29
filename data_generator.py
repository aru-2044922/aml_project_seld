#
# Data generator for training the SELDnet
#

import os
import numpy as np
import feature_class
from IPython import embed
from collections import deque
import random


class DataGenerator(object):
    def __init__(
            self, dataset='foa', feat_label_dir='', is_eval=False, split=1, batch_size=32, seq_len=64,
            shuffle=True, per_file=False
    ):
        self._per_file = per_file
        self._is_eval = is_eval
        self._splits = np.array(split)
        self._batch_size = batch_size # 16
        self._seq_len = seq_len # 128 --> number of frames to process per time
        self._shuffle = shuffle
        self._feat_cls = feature_class.FeatureClass(feat_label_dir=feat_label_dir, dataset=dataset, is_eval=is_eval)
        self._label_dir = self._feat_cls.get_label_dir()
        self._feat_dir = self._feat_cls.get_normalized_feat_dir()

        self._filenames_list = list()
        self._nb_frames_file = 0     # 3000 --> Using a fixed number of frames in feat files. Updated in _get_label_filenames_sizes()
        
        self._nb_mel_bins = self._feat_cls.get_nb_mel_bins()
        self._nb_ch = None

        self._label_len = None  # total length of label - DOA + SED
        self._doa_len = None    # DOA label length
        self._class_dict = self._feat_cls.get_classes()
        self._nb_classes = len(self._class_dict.keys())
        self._default_azi, self._default_ele = self._feat_cls.get_default_azi_ele_regr()
        self._get_filenames_list_and_feat_label_sizes()

        self._batch_seq_len = self._batch_size*self._seq_len #2048
        self._circ_buf_feat = None
        self._circ_buf_label = None

        # get total number of batches for this split
        if self._per_file:
            self._nb_total_batches = len(self._filenames_list)
        else:
            self._nb_total_batches = int(np.floor((len(self._filenames_list) * self._nb_frames_file /
                                               float(self._seq_len * self._batch_size)))) # 146 --> 100 files with 3000 frames, sequence length of 128 and batch size of 16

        print(
            '\tDatagen_mode: {}, nb_files: {}, nb_classes:{}\n'
            '\tnb_frames_file: {}, feat_len: {}, nb_ch: {}, label_len:{}\n'.format(
                'eval' if self._is_eval else 'dev', len(self._filenames_list),  self._nb_classes,
                self._nb_frames_file, self._nb_mel_bins, self._nb_ch, self._label_len
                )
        )

        print(
            '\tDataset: {}, split: {}\n'
            '\tbatch_size: {}, seq_len: {}, shuffle: {}\n'
            '\tTotal batches in dataset: {}\n'
            '\tlabel_dir: {}\n '
            '\tfeat_dir: {}\n'.format(
                dataset, split,
                self._batch_size, self._seq_len, self._shuffle,
                self._nb_total_batches,
                self._label_dir, self._feat_dir
            )
        )

    def get_data_sizes(self):
        feat_shape = (self._batch_size, self._nb_ch, self._seq_len, self._nb_mel_bins)
        if self._is_eval:
            label_shape = None
        else:
            label_shape = [
                (self._batch_size, self._seq_len, self._nb_classes),
                (self._batch_size, self._seq_len, self._nb_classes*2)
            ]
        # feat_shape = batch_size, channel_count*2, sequence_length, feature_length = 16,8,128,1024
        # label_shape = (batch_size, sequence_length, class_count), (batch_size, sequence_length, class_count * 2)= (16,128,11),(16,128,22)
        return feat_shape, label_shape

    def get_total_batches_in_data(self):
        return self._nb_total_batches

    def _get_filenames_list_and_feat_label_sizes(self):
        # only take files in the specified split
        for filename in os.listdir(self._feat_dir):
            if int(filename[5]) in self._splits: # check which split the file belongs to
                self._filenames_list.append(filename)

        # get number of frames in a sample audio (feature extraction)
        temp_feat = np.load(os.path.join(self._feat_dir, self._filenames_list[0]))
        self._nb_frames_file = temp_feat.shape[0]
        self._nb_ch = temp_feat.shape[1] // self._nb_mel_bins
        
        if not self._is_eval:
            temp_label = np.load(os.path.join(self._label_dir, self._filenames_list[0])) # get label for a sample audio
            self._label_len = temp_label.shape[-1] # get the length of the labels (sed and doa[azi & ele] -> class*3) # 33
            self._doa_len = (self._label_len - self._nb_classes)//self._nb_classes # 2 (if there are 11 classes)

        if self._per_file:
            self._batch_size = int(np.ceil(temp_feat.shape[0]/float(self._seq_len)))

        return

    def generate(self):
        """
        Generates batches of samples
        :return: 
        """

        while 1:
            if self._shuffle:
                random.shuffle(self._filenames_list)

            self._circ_buf_feat = deque()
            self._circ_buf_label = deque()

            file_cnt = 0
            # Batch Sequence Length: 2048 (128 * 16) (seq_len * batch_size) 
            # Total Batches: 292 (number of files in batch * frames_per_file / batch_sequence_length) (200 * 3000 / 2048) --> if split uses two sets of training files
            for i in range(self._nb_total_batches):
                
                # load feat and label to circular buffer. Always maintain atleast one batch worth feat and label in the
                # circular buffer. If not keep refilling it.
                while len(self._circ_buf_feat) < self._batch_seq_len:
                    temp_feat = np.load(os.path.join(self._feat_dir, self._filenames_list[file_cnt]))
                    temp_label = np.load(os.path.join(self._label_dir, self._filenames_list[file_cnt]))

                    # iterate through all 3000 frames by 448 ( 448 is 64 mel bins by 7 features; 4 for mel and 3 for intensity) in the file
                    # add the frame info (mag and phase features for all channels)
                    # add the label info (33 which is 11 for SED and 22 for DOA)
                    for row_cnt, row in enumerate(temp_feat):
                        self._circ_buf_feat.append(row)
                        self._circ_buf_label.append(temp_label[row_cnt])

                    # If self._per_file is True, this returns the sequences belonging to a single audio recording
                    if self._per_file:
                        extra_frames = self._batch_seq_len - temp_feat.shape[0]
                        extra_feat = np.ones((extra_frames, temp_feat.shape[1])) * 1e-6

                        extra_labels = np.zeros((extra_frames, temp_label.shape[1]))
                        extra_labels[:, self._nb_classes:2 * self._nb_classes] = self._default_azi
                        extra_labels[:, 2 * self._nb_classes:] = self._default_ele

                        for row_cnt, row in enumerate(extra_feat):
                            self._circ_buf_feat.append(row)
                            self._circ_buf_label.append(extra_labels[row_cnt])

                    file_cnt = file_cnt + 1

                # Read one batch size from the circular buffer
                # feat -> extract batch -> (2048, 448) instead of the full 3000, 8192
                # label -> extract batch -> (2048, 33) instead of the full 3000, 33
                feat = np.zeros((self._batch_seq_len, self._nb_mel_bins * self._nb_ch))
                label = np.zeros((self._batch_seq_len, self._label_len))
                for j in range(self._batch_seq_len):
                    feat[j, :] = self._circ_buf_feat.popleft()
                    label[j, :] = self._circ_buf_label.popleft()
                feat = np.reshape(feat, (self._batch_seq_len, self._nb_mel_bins, self._nb_ch))
                # feat -> 2048, 64, 7

                # Split to sequences
                feat = self._split_in_seqs(feat)
                # feat -> 16, 128, 64, 7

                label = self._split_in_seqs(label)
                # label -> 16, 128, 33

                # Get azi/ele in radians
                # Normalize azi values and convert it to rad
                azi_rad = label[:, :, self._nb_classes:2 * self._nb_classes] * np.pi / self._default_azi

                # rescaling the elevation data from [-def_elevation def_elevation] to [-180 180] to keep them in the
                # range of azimuth angle
                ele_rad = label[:, :, 2 * self._nb_classes:] * np.pi / self._default_ele

                label = [
                    label[:, :, :self._nb_classes],  # SED labels
                    np.concatenate((azi_rad, ele_rad), -1)  # DOA labels in radians
                        ]

                yield feat, label

    def _split_in_seqs(self, data):
        if len(data.shape) == 1:
            if data.shape[0] % self._seq_len:
                data = data[:-(data.shape[0] % self._seq_len), :]
            data = data.reshape((data.shape[0] // self._seq_len, self._seq_len, 1))
        elif len(data.shape) == 2:
            if data.shape[0] % self._seq_len:
                data = data[:-(data.shape[0] % self._seq_len), :]
            data = data.reshape((data.shape[0] // self._seq_len, self._seq_len, data.shape[1]))
        elif len(data.shape) == 3:
            if data.shape[0] % self._seq_len:
                data = data[:-(data.shape[0] % self._seq_len), :, :]
            data = data.reshape((data.shape[0] // self._seq_len, self._seq_len, data.shape[1], data.shape[2]))
        else:
            print('ERROR: Unknown data dimensions: {}'.format(data.shape))
            exit()
        return data

    def get_default_elevation(self):
        return self._default_ele

    def get_nb_classes(self):
        return self._nb_classes

    def nb_frames_1s(self):
        return self._feat_cls.nb_frames_1s()