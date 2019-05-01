import math
import os
import re

import cv2
import numpy as np
import tensorflow as tf

import util

tf.enable_eager_execution()


class Dataset:
    def __init__(self, force_preprocessing=False):
        self.db_path = "db/"
        self.unprocessed_path = "faces/"
        self.preprocessed_path = "preprocessed/"
        self.image_index = []
        self.age_index = []
        self.gender_index = []
        self.target_image_size = 250
        self.train_record_path = self.db_path + 'train.tfrec'
        self.test_record_path = self.db_path + 'test.tfrec'
        self.val_record_path = self.db_path + 'val.tfrec'

        if not os.path.isfile(
                self.db_path + self.preprocessed_path + "lock") or \
                force_preprocessing:
            print("Started preprocessing data...")
            self.preprocess_data()
            self.load_indexes()
            self.save_dataset()
        else:
            self.load_indexes()
            self.save_dataset()

    def load_indexes(self):
        with open('db/preprocessed/image.index', 'r') as f_handle:
            formatted_image_index = f_handle.readlines()
            for formatted_image_input in formatted_image_index:
                self.image_index.append(formatted_image_input[:-1])

        with open('db/preprocessed/age.index', 'r') as f_handle:
            formatted_age_index = f_handle.readlines()
            for formatted_age_input in formatted_age_index:
                # values = formatted_age_input[:-1]
                # values = values.split(" ")
                self.age_index.append(formatted_age_input[:-1])
                # self.age_index.append((float(values[0]), float(values[1])))

        with open('db/preprocessed/gender.index', 'r') as f_handle:
            formatted_gender_index = f_handle.readlines()
            for formatted_gender_input in formatted_gender_index:
                self.gender_index.append(int(formatted_gender_input[:-1]))

    def resize_image(self, img_path_in_db):
        img = cv2.imread(self.db_path + self.unprocessed_path + img_path_in_db)
        old_dimensions = img.shape[:2]

        ratio = float(self.target_image_size) / max(old_dimensions)
        new_dimensions = tuple([int(x * ratio) for x in old_dimensions])

        img = cv2.resize(img, (new_dimensions[1], new_dimensions[0]))

        delta_w = self.target_image_size - new_dimensions[1]
        delta_h = self.target_image_size - new_dimensions[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        color = [0, 0, 0]
        new_img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                     cv2.BORDER_CONSTANT,
                                     value=color)

        # Ensure directory exists or is created !
        save_path = self.db_path + self.preprocessed_path + \
                    img_path_in_db.replace("/", "_")
        directory = os.path.dirname(save_path)
        try:
            os.makedirs(directory)
        except FileExistsError:
            pass  # It's okay, nothing to be done

        cv2.imwrite(save_path, new_img)

        return save_path

    @staticmethod
    def get_age_interval(mean):
        age_intervals = [3, 7, 14, 24, 37, 47, 59, 150]
        for i in range(len(age_intervals)):
            if mean <= age_intervals[i]:
                return i
        return 3

    def preprocess_data(self):
        # Retrieve every image entry in the database
        data_indexes = ["fold_0_data.txt", "fold_1_data.txt", "fold_2_data.txt",
                        "fold_3_data.txt", "fold_4_data.txt"]
        entries = []
        for i in range(0, len(data_indexes)):
            index = open("db/" + data_indexes[i], 'r')
            next(index)  # Skip header row
            entries.extend(index.readlines())

        image_index = []
        gender_index = []
        age_index = []  # Format : (`mean`, `standard_deviation`)

        # Resize them to fixed size + filling data indexes
        for i in range(0, len(entries)):
            util.print_progress_bar(i + 1, len(entries), 'Resizing images : ')
            entry = entries[i]
            entry = entry.split("\t")
            save_path = self.resize_image(
                entry[0] + "/" + "coarse_tilt_aligned_face." + entry[2] + "." +
                entry[1])
            image_index.append(save_path)
            try:
                if entry[4] == "f" or entry[4] == "m":
                    gender_index.append(1 if entry[4] == "f" else 0)
                else:
                    raise ValueError
                if entry[3][0] == '(':
                    interval = re.sub("\(|\)|\s", "", entry[3]).split(",")
                    bottom = int(interval[0])
                    top = int(interval[1])
                    mean = ((bottom + top) / 2, (top - bottom) / 2)
                    age_index.append(self.get_age_interval(mean[0]))
                else:
                    age_index.append(self.get_age_interval(int(entry[3])))
            except ValueError:  # ignore [incorrectly formatted / incomplete]
                # lines
                image_index.pop()

        # Flushing indexes to disk
        with open('db/preprocessed/image.index', 'w+') as f_handle:
            for item in image_index:
                f_handle.write('%s\n' % item)

        with open('db/preprocessed/age.index', 'w+') as f_handle:
            for item in age_index:
                f_handle.write('%s\n' % item)

        with open('db/preprocessed/gender.index', 'w+') as f_handle:
            for item in gender_index:
                f_handle.write('%s\n' % item)

        # Work out mean image
        mean_img = np.zeros((self.target_image_size, self.target_image_size, 3),
                            np.uint32)
        for i in range(0, len(image_index)):
            util.print_progress_bar(i + 1, len(image_index),
                                    'Working out mean : ')
            img = cv2.imread(image_index[i])
            mean_img += img
        mean_img = np.true_divide(mean_img, len(image_index))
        cv2.imwrite("db/preprocessed/mean.jpg", mean_img)

        # Work out standard deviation image
        standard_deviation_img = np.zeros(
            (self.target_image_size, self.target_image_size, 3), np.uint32)
        for i in range(0, len(image_index)):
            util.print_progress_bar(i + 1, len(image_index),
                                    'Working out standard deviation : ')
            img = cv2.imread(image_index[i])
            standard_deviation_img += np.power(img - mean_img, 2).astype(
                np.uint32)
        standard_deviation_img = np.sqrt(
            np.true_divide(standard_deviation_img, len(image_index) - 1))
        cv2.imwrite("db/preprocessed/standard_deviation.jpg",
                    standard_deviation_img)

        # Normalizing images
        absolute_min = math.inf
        absolute_max = -math.inf
        for i in range(0, len(image_index)):
            util.print_progress_bar(i + 1, len(image_index),
                                    'Retrieving image scale : ')
            img = ((cv2.imread(image_index[i]).astype(
                np.int16) - mean_img) / standard_deviation_img)
            img_min = np.amin(img)
            if img_min < absolute_min:
                absolute_min = img_min
            img_max = np.amax(img)
            if img_max > absolute_max:
                absolute_max = img_max
            np.save(image_index[i] + ".npy", img.astype(np.int16))

        for i in range(0, len(image_index)):
            util.print_progress_bar(i + 1, len(image_index),
                                    'Normalizing images : ')
            img = np.load(image_index[i] + ".npy").astype(np.float64)
            img -= absolute_min  # Start @ 0
            img *= 255 / (absolute_max - absolute_min)  # Occupy full scale
            cv2.imwrite(image_index[i], img)
            os.remove(image_index[i] + ".npy")

        f_handle = open(self.db_path + self.preprocessed_path + "lock", "w+")
        f_handle.close()
        print("All done.")

    @staticmethod
    def preprocess_image(image):
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize_images(image, [250, 250])
        image /= 255.0  # normalize to [0,1] range

        return image

    def save_dataset(self):
        ds = tf.data.Dataset.from_tensor_slices(self.image_index).map(
            tf.read_file)

        BATCH_SIZE = 32
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        DATASET_SIZE = len(self.image_index)
        train_size = int(0.7 * DATASET_SIZE)
        test_size = int(0.15 * DATASET_SIZE)

        out_ds = tf.data.Dataset.from_tensor_slices(self.age_index)
        ds = tf.data.Dataset.zip((ds, out_ds))

        # Setting a shuffle buffer size as large as the dataset ensures that
        # the data is completely shuffled.
        ds = ds.shuffle(buffer_size=DATASET_SIZE)
        ds = ds.repeat()
        ds = ds.batch(BATCH_SIZE)
        # `prefetch` lets the dataset fetch batches, in the background while
        # the model is training.
        ds = ds.prefetch(buffer_size=AUTOTUNE)

        train_dataset = ds.take(train_size)
        test_dataset = ds.skip(train_size)
        test_dataset = test_dataset.take(test_size)
        val_dataset = test_dataset.skip(test_size)

        tfrec = tf.data.experimental.TFRecordWriter(
            self.train_record_path)
        tfrec.write(train_dataset)
        tfrec = tf.data.experimental.TFRecordWriter(
            self.test_record_path)
        tfrec.write(test_dataset)
        tfrec = tf.data.experimental.TFRecordWriter(
            self.val_record_path)
        tfrec.write(val_dataset)
