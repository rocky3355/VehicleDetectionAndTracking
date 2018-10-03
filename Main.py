import cv2
import glob
from scipy import misc
from keras.layers import *
from keras.models import Sequential

VEHICLE_IMAGES_DIR = 'TrainingData/Vehicles'
NON_VEHICLE_IMAGES_DIR = 'TrainingData/NonVehicles'


def create_model():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(64, 64, 3)))
    model.add(Conv2D(24, (5, 5), subsample=(2, 2), activation='relu'))
    model.add(Conv2D(36, (5, 5), subsample=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), subsample=(2, 2), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model


model = create_model()

vehicle_image_files = glob.iglob(VEHICLE_IMAGES_DIR + '/**/*.png', recursive=True)
non_vehicle_image_files = glob.iglob(NON_VEHICLE_IMAGES_DIR + '/**/*.png', recursive=True)

images = []
labels = []

for img_file in vehicle_image_files:
    images.append(misc.imread(img_file))
    labels.append(1)
for img in non_vehicle_image_files:
    images.append(misc.imread(img_file))
    labels.append(0)

# TODO: Use generator?
images = np.array(images)
labels = np.array(labels)

history = model.fit(images, labels, epochs=2, validation_split=0.2, shuffle=True)
#model.save('model.h5')

img1 = misc.imread('TestImages/Car01.jpg')
img2 = misc.imread('TestImages/Car02.jpg')
img1 = cv2.resize(img1, (64, 64))
img2 = cv2.resize(img2, (64, 64))
test_images = np.array([img1, img2])
test_labels = np.array([1, 1])

metrics = model.evaluate(test_images, test_labels)
for metric_i in range(len(model.metrics_names)):
    print('Loss: {}'.format(metrics))