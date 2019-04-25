image_index = []
with open('db/preprocessed/image.index', 'r') as f_handle:
    formatted_image_index = f_handle.readlines()
    for formatted_image_input in formatted_image_index:
        image_index.append(formatted_image_input[:-1])

age_index = []
with open('db/preprocessed/age.index', 'r') as f_handle:
    formatted_age_index = f_handle.readlines()
    for formatted_age_input in formatted_age_index:
        values = formatted_age_input[:-1]
        values = values.split(" ")

        age_index.append((float(values[0]), float(values[1])))

gender_index = []
with open('db/preprocessed/gender.index', 'r') as f_handle:
    formatted_gender_index = f_handle.readlines()
    for formatted_gender_input in formatted_gender_index:
        gender_index.append(int(formatted_gender_input[:-1]))

train_count = 500
test_count = 50

for i in range(0, train_count):
    x_train = cv2.imread(image_index[i])
    y_train = age_index[i]

for i in range(train_count, test_count):
    x_test = cv2.imread(image_index[i])
    y_test = age_index[i]
