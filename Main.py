import glob
from scipy import misc
from keras.layers import *
from keras.models import Sequential

VEHICLE_IMAGES_DIR = 'TrainingData/Vehicles'
NON_VEHICLE_IMAGES_DIR = 'TrainingData/NonVehicles'


def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Flatten())
    #model.add(Dense(128))
    #model.add(Activation('relu'))
    #model.add(Dense(5))
    model.add(Dense(1))
    return model


model = create_model()

vehicle_image_files = glob.iglob(VEHICLE_IMAGES_DIR + '/**/*.png', recursive=True)
non_vehicle_image_files = glob.iglob(NON_VEHICLE_IMAGES_DIR + '/**/*.png', recursive=True)

images = []
labels = []

for img_file in vehicle_image_files:
    img = (misc.imread(img_file) - 128.0) / 128.0
    images.append(img)
    labels.append(1)
for img in non_vehicle_image_files:
    img = (misc.imread(img_file) - 128.0) / 128.0
    images.append(img)
    labels.append(0)

images = np.array(images)
labels = np.array(labels)

from sklearn.preprocessing import LabelBinarizer
from keras.utils import to_categorical
label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit_transform(labels)

model.compile('adam', 'binary_crossentropy', ['accuracy'])
history = model.fit(images, y_one_hot, epochs=1, validation_split=0.2)
exit(0)

with open('small_test_traffic.p', 'rb') as f:
    data_test = pickle.load(f)

X_test = data_test['features']
y_test = data_test['labels']

# preprocess data
X_normalized_test = np.array((X_test - 128.0) / 128.0 )
y_one_hot_test = label_binarizer.fit_transform(y_test)

print("Testing")

# TODO: Evaluate the test data in Keras Here
metrics = model.evaluate(X_normalized_test, y_one_hot_test)
# TODO: UNCOMMENT CODE
for metric_i in range(len(model.metrics_names)):
    metric_name = model.metrics_names[metric_i]
    metric_value = metrics[metric_i]
    print('{}: {}'.format(metric_name, metric_value))