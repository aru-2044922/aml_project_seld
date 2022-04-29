# Extracts the features, labels, and normalizes the development and evaluation split features.

import feature_class
import os


dirname = os.path.dirname(__file__)
dataset_dir = os.path.join(dirname, 'dataset/')   # Base folder containing the foa/mic and metadata folders
feat_label_dir = os.path.join(dirname, 'dataset/feat_label/')  # Directory to dump extracted features and labels


# -------------- Extract features and labels for development set -----------------------------
dev_feat_cls = feature_class.FeatureClass(dataset_dir=dataset_dir, feat_label_dir=feat_label_dir)

# Extract features and normalize them
dev_feat_cls.extract_all_feature() # saves result
dev_feat_cls.preprocess_features() # saves scaled features

# Extract labels
dev_feat_cls.extract_all_labels()

# Perform data augmentation to increase files by 25%
sample_feature = dev_feat_cls.augment_mel()

# Show sample feature before and after augmentation
'''plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

fig, ax = plt.subplots()
img = librosa.display.specshow(sample_feature[0], x_axis='time', y_axis='mel', ax=ax, sr=dev_feat_cls.get_sample_rate(), hop_length=dev_feat_cls.get_hop_len())
ax.set(title='Mel spectrogram (before augmentation)')
fig.colorbar(img, ax=ax, format="%+2.f dB")

fig, ax = plt.subplots()
img = librosa.display.specshow(sample_feature[1], x_axis='time', y_axis='mel', ax=ax, sr=dev_feat_cls.get_sample_rate(), hop_length=dev_feat_cls.get_hop_len())
ax.set(title='Mel spectrogram (after frequency and time masking)')
fig.colorbar(img, ax=ax, format="%+2.f dB")'''

