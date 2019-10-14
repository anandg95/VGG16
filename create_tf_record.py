# real imagenet is too big for this project. I will use tiny imagenet, with 200 classes
import pickle
import numpy as np
from tqdm import tqdm
import glob
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

imagenet_tiny_dir = "/home/anand/Work/trials_runs/ML/learn_here/tiny-imagenet-200"
classes_list = f"{imagenet_tiny_dir}/wnids.txt"

class_name_to_index = {}
n_classes = 200
train_eval_split = 0.8

total_images_train = 80000
total_images_eval = 20000


def create_byte_feature(value):
    feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    return feature


def create_int_feature(value):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    return feature


def create(im, label, writer):
    feature = {}
    feature["image"] = create_byte_feature(im)
    feature["label"] = create_int_feature(label)
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(tf_example.SerializeToString())


def build_index():
    with open(classes_list, "r") as f:
        lines = f.read().split("\n")
    for i, lbl_name in enumerate(lines[:n_classes]):
        class_name_to_index[lbl_name] = i


def access_images_in_folder(label_id):
    folder_path = f"{imagenet_tiny_dir}/train/{label_id}/images/*.JPEG"
    all_images = glob.glob(folder_path)
    for img in all_images[: int(train_eval_split * len(all_images))]:
        im = cv2.imread(img)
        # im = im.resize((224, 224)) # resize in input_fn
        # im_np = np.array(im)
        create(
            cv2.imencode(".jpeg", im)[1].tostring(),
            class_name_to_index[label_id],
            train_writer,
        )
    for img in all_images[int(train_eval_split * len(all_images)) :]:
        im = cv2.imread(img)
        # im = im.resize((224, 224))
        # im_np = np.array(im)
        create(
            cv2.imencode(".jpeg", im)[1].tostring(),
            class_name_to_index[label_id],
            eval_writer,
        )


def ransack_the_folders():
    for folder in tqdm(list(class_name_to_index.keys())):
        access_images_in_folder(folder)


if __name__ == "__main__":
    train_writer = tf.io.TFRecordWriter("./dataset/train/data.tfrecord")
    eval_writer = tf.io.TFRecordWriter("./dataset/eval/data.tfrecord")
    build_index()
    ransack_the_folders()
