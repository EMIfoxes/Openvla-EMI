import tensorflow as tf
import tensorflow_datasets as tfds
# TFRECORD = "/media/lxx/Elements/project/openvla-oft/modified_libero_rlds/libero_10_no_noops/1.0.0/liber_o10-train.tfrecord-00000-of-00032"

# raw_dataset = tf.data.TFRecordDataset(TFRECORD)
# for raw in raw_dataset.take(1):
#     example = tf.train.Example()
#     example.ParseFromString(raw.numpy())
#     # print(example)          # 文本格式，能看到所有 key 与类型
#     print(example.features.feature.keys())   # 只看 key

import dlimp as dl, matplotlib.pyplot as plt

tfrec = '/media/lxx/Elements/project/openvla-oft/modified_libero_rlds/libero_10_no_noops/1.0.0/liber_o10-train.tfrecord-00000-of-00032'
# ds = dl.TFRecordDataset(tfrec).decode()   # 自动解 bytes->np.uint8
# sample = ds[0]                            # dict, 已经带 'image'

# plt.imshow(sample['image'])               # (H,W,3) RGB
# plt.axis('off')
# plt.show()

name = "libero_10_no_noops"

builder = tfds.builder(name, data_dir=tfrec)