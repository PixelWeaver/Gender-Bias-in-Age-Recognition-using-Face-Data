import cv2
import os
import numpy as np
import math

target_size = 500


# Print iterations progress
# FROM https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def print_progress_bar(iteration, total, prefix='Progress :', suffix='Complete', decimals=1, length=50, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def resize_image(img_path):
    img = cv2.imread("db/faces/" + img_path)
    old_dimensions = img.shape[:2]

    ratio = float(target_size) / max(old_dimensions)
    new_dimensions = tuple([int(x * ratio) for x in old_dimensions])

    img = cv2.resize(img, (new_dimensions[1], new_dimensions[0]))

    delta_w = target_size - new_dimensions[1]
    delta_h = target_size - new_dimensions[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                 value=color)

    # Ensure directory exists or is created !
    save_path = "db/preprocessed/" + img_path.replace("/", "_")
    directory = os.path.dirname(save_path)
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass  # It's okay, nothing to be done

    cv2.imwrite(save_path, new_img)

    return save_path


def preprocess_data():
    # Retrieve every image entry in the database
    data_indexes = ["fold_0_data.txt", "fold_1_data.txt", "fold_2_data.txt", "fold_3_data.txt", "fold_4_data.txt"]
    entries = []
    for i in range(0, len(data_indexes)):
        index = open("db/" + data_indexes[i], 'r')
        next(index)  # Skip header row
        entries.extend(index.readlines())

    data_index = []  # Format : list of [`image path in db`, `age`, `gender`]

    # Resize them to fixed size + filling data_index
    for i in range(0, len(entries)):
        print_progress_bar(i + 1, len(entries), 'Resizing images : ')
        entry = entries[i]
        entry = entry.split("\t")
        save_path = resize_image(entry[0] + "/" + "coarse_tilt_aligned_face." + entry[2] + "." + entry[1])
        data_index.append([save_path, entry[3], entry[4]])

    # Work out mean image
    mean_img = np.zeros((target_size, target_size, 3), np.uint32)
    for i in range(0, len(data_index)):
        print_progress_bar(i + 1, len(data_index), 'Working out mean : ')
        img = cv2.imread(data_index[i][0])
        mean_img += img
    mean_img = np.true_divide(mean_img, len(data_index))
    cv2.imwrite("db/preprocessed/mean.jpg", mean_img)

    # Work out standard deviation image
    standard_deviation_img = np.zeros((target_size, target_size, 3), np.uint32)
    for i in range(0, len(data_index)):
        print_progress_bar(i + 1, len(data_index), 'Working out standard deviation : ')
        img = cv2.imread(data_index[i][0])
        standard_deviation_img += np.power(img - mean_img, 2).astype(np.uint32)
    standard_deviation_img = np.sqrt(np.true_divide(standard_deviation_img, len(data_index) - 1))
    cv2.imwrite("db/preprocessed/standard_deviation.jpg", standard_deviation_img)

    # Normalizing images
    absolute_min = math.inf
    absolute_max = -math.inf
    normalized_images = []
    for i in range(0, len(data_index)):
        print_progress_bar(i + 1, len(data_index), 'Normalizing images : ')
        img = ((cv2.imread(data_index[i][0]).astype(np.int64) - mean_img) / standard_deviation_img)
        img_min = np.amin(img)
        if img_min < absolute_min:
            absolute_min = img_min
        amax = np.amax(img)
        if amax > absolute_max:
            absolute_max = amax
        normalized_images.append(img)

    for i in range(0, len(data_index)):
        normalized_images[i] -= absolute_min  # Start @ 0
        normalized_images[i] *= 255 / (absolute_max - absolute_min)  # Occupy full scale
        cv2.imwrite(data_index[i][0], normalized_images[i])

    print("All done !")


preprocess_data()
