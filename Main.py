import cv2
import glob
import keras
from scipy import misc
from keras.layers import *
from keras.models import Sequential
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label


LOAD_MODEL = True
MODEL_IMG_SIZE = (64, 64)
MODEL_FILE_NAME = 'model.h5'
VEHICLE_IMAGES_DIR = 'TrainingData/Vehicles'
NON_VEHICLE_IMAGES_DIR = 'TrainingData/NonVehicles'


class Window:
    def __init__(self, bbox, weight):
        self.bbox = bbox
        self.weight = weight


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
    model.compile(optimizer='adam', loss='mse')
    return model


def load_model():
    model = keras.models.load_model(MODEL_FILE_NAME)
    return model


def train_model(model):
    vehicle_image_files = glob.iglob(VEHICLE_IMAGES_DIR + '/**/*.png', recursive=True)
    non_vehicle_image_files = glob.iglob(NON_VEHICLE_IMAGES_DIR + '/**/*.png', recursive=True)

    images = []
    labels = []

    for img_file in vehicle_image_files:
        images.append(misc.imread(img_file))
        labels.append(1)
    for img_file in non_vehicle_image_files:
        images.append(misc.imread(img_file))
        labels.append(0)

    # TODO: Use generator? (Actually not needed)
    images = np.array(images)
    labels = np.array(labels)

    model.fit(images, labels, epochs=2, validation_split=0.2, shuffle=True)
    model.save(MODEL_FILE_NAME)


def test_model():
    img1 = cv2.resize(misc.imread('TestImages/Car01.jpg'), MODEL_IMG_SIZE)
    img2 = cv2.resize(misc.imread('TestImages/Car02.jpg'), MODEL_IMG_SIZE)
    img3 = cv2.resize(misc.imread('TestImages/test2.jpg'), MODEL_IMG_SIZE)
    img4 = cv2.resize(misc.imread('TestImages/test3.jpg'), MODEL_IMG_SIZE)
    test_images = np.array([img1, img2, img3, img4])
    test_labels = np.array([1, 1, 0, 0])
    metrics = model.evaluate(test_images, test_labels)
    for metric_i in range(len(model.metrics_names)):
        print('Loss: {}'.format(metrics))


def draw_boxes(img, windows, color=(0, 0, 255), thick=6):
    for window in windows:
        cv2.rectangle(img, window[0], window[1], color, thick)


def slide_window(img_shape, x_start_stop, y_start_stop, xy_window, xy_overlap, weight):
    img_width = img_shape[1]
    img_height = img_shape[0]

    x_start = x_start_stop[0]
    x_stop = x_start_stop[1]
    y_start = y_start_stop[0]
    y_stop = y_start_stop[1]

    if x_start is None:
        x_start = 0
    if x_stop is None:
        x_stop = img_width
    if y_start is None:
        y_start = 0
    if y_stop is None:
        y_stop = img_height

    #print('Boundaries: ({0}, {1}); ({2}, {3})'.format(x_start, y_start, x_stop, y_stop))

    x_span = x_stop - x_start
    y_span = y_stop - y_start
    #print('Span: {0}, {1}'.format(x_span, y_span))

    x_step = xy_window[0] * (1 - xy_overlap[0])
    y_step = xy_window[1] * (1 - xy_overlap[1])
    #print('Step: {0}, {1}'.format(x_step, y_step))

    # TODO: What about rounding?
    x_windows = np.int(x_span / x_step)
    y_windows = np.int(y_span / y_step)
    #print('#Windows: {0}, {1}'.format(x_windows, y_windows))

    window_list = []
    for y in range(y_windows):
        for x in range(x_windows):
            left_top = (np.int(x_start + x * x_step), np.int(y_start + y * y_step))
            right_bottom = (np.int(left_top[0] + xy_window[0]), np.int(left_top[1] + xy_window[1]))
            if right_bottom[0] <= img_width and right_bottom[1] <= img_height:
                window = Window((left_top, right_bottom), weight)
                window_list.append(window)

    return window_list


def create_search_windows(img_shape):
    overlap = (0.75, 0.75)
    near_windows = slide_window(img_shape, (None, None), (300, None), (256, 256), overlap, 1)
    mid_windows = slide_window(img_shape, (None, None), (400, 550), (128, 128), overlap, 1)
    far_windows = slide_window(img_shape, (None, None), (400, 450), (64, 64), overlap, 2)

    # TODO: Add weights for the window sizes: Smaller windows will get a higher weight
    windows = near_windows + mid_windows + far_windows

    #image = misc.imread('TestImages/test11.jpg')
    #draw_boxes(image, near_windows, (0, 0, 255), 6)
    #draw_boxes(image, mid_windows, (0, 255, 0), 6)
    #draw_boxes(image, far_windows, (255, 0, 0), 6)
    #misc.imsave('windows.jpg', image)
    #exit(0)

    return windows


def get_window_images(img, windows):
    window_imgs = []
    for window in windows:
        bbox = window.bbox
        window_img = img[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
        window_img = cv2.resize(window_img, MODEL_IMG_SIZE)
        #misc.imsave('test.jpg', window_img)
        window_imgs.append(window_img)
    window_imgs = np.array(window_imgs)
    return window_imgs


def draw_labeled_bboxes(img, labels):
    for car_number in range(1, labels[1]+1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)


iteration = 0
heat_map = None
last_labels = None
max_hmap = 0
HEAT_MAP_THRESHOLD = 30
HEAT_MAP_ITERATIONS = 3

def find_vehicles(img):
    global heat_map, iteration, last_labels, max_hmap

    #img = misc.imread('TestImages/test12.jpg')
    if heat_map is None:
        heat_map = np.zeros_like(img[:, :, 0]).astype(np.float)

    window_images = get_window_images(img, windows)

    result = model.predict(window_images)
    bool_result = list(map(lambda x: x[0] > 0.8, result))

    for idx in range(len(bool_result)):
        if bool_result[idx]:
            window = windows[idx]
            bbox = window.bbox
            heat_map[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]] += window.weight

    iteration += 1
    if iteration == HEAT_MAP_ITERATIONS:
        max = np.max(heat_map)
        if max > max_hmap:
            max_hmap = max
        #heat_map = np.clip(heat_map, 0, 255)
        #misc.imsave('heatmap.jpg', heat_map)
        heat_map[heat_map < HEAT_MAP_THRESHOLD] = 0
        #misc.imsave('heatmap_threshold.jpg', heat_map)

        labels = label(heat_map)
        #misc.imsave('labels.jpg', labels[0])

        draw_labeled_bboxes(img, labels)
        # misc.imsave('result.jpg', img)
        last_labels = labels

        iteration = 0
        heat_map = None
    elif last_labels is not None:
        draw_labeled_bboxes(img, last_labels)
    return img


if LOAD_MODEL:
    model = load_model()
else:
    model = create_model()
    train_model(model)

#test_model()

img_shape = (720, 1280)
windows = create_search_windows(img_shape)

# Load the video
input_video = VideoFileClip('Videos/project_video.mp4')
# Apply the lane finding algorithm to each frame
output_video = input_video.fl_image(find_vehicles)
# Save the video
output_video.write_videofile('output.mp4', audio=False)

print(max_hmap)
