# Vehicle Detection and Tracking
The goal of this project was to implement a pipeline for detecting vehicles in a video stream and to draw bounding boxes around them.

## The pipeline
I decided to go for the machine learning approach, as this is the state of the art way to tackle such kind of problems. The pipeline consists of the following steps:

### 1. Setup
- Create Keras model
- Load training images
- Train the model
- Save the model

If the model has been created already, the saved model will be loaded instead of executing these steps. I used the same model that I already used for the Behavioral Cloning project, except the cropping layer had to be removed and the input size changed. I used the Adam optimizer, a validation split of 0.2, shuffling and trained for 5 epochs. The training images are the ones provided for this project.

|Layer        |Description                                                |
| ----------- | --------------------------------------------------------- |
|Lambda       | Input: 64x64x3, Normalizes the images from -0.5 to 0.5    |
|Convolution2D| Filters: 24, Kernel: 5x5, Subsample: 2x2, Activation: Relu|
|Convolution2D| Filters: 36, Kernel: 5x5, Subsample: 2x2, Activation: Relu|
|Convolution2D| Filters: 48, Kernel: 5x5, Subsample: 2x2, Activation: Relu|
|Dropout      | Keeping probability: 0.5                                  |
|Convolution2D| Filters: 64, Kernel: 3x3, Activation: Relu                |
|Convolution2D| Filters: 64, Kernel: 3x3, Activation: Relu                |
|Dropout      | Keeping probability: 0.5                                  |
|Flatten      | Flattens the data to a one-dimensional array              |
|Dense        | Output: 100                                               |
|Dense        | Output: 50                                                |
|Dense        | Output: 10                                                |
|Dense        | Output: 1                                                 |

### 2. Image processing
Each image of the video stream will be processed by the following algorithm:
- Use sliding windows to get excerpts from the image
- Scale those excerpts to fit the model input size
- Let the model predict if the input either displays a car or not
- If the model predicts a car, an image-sized heatmap is increased at the area of the excerpt
- This is repeated for a certain amount of frames, the heatmap status is not being reset inbetween
- The heatmap is thresholded in order to filter for the "hot" spots that depict cars
- scipy is used to create labels for the distinct cars
- Bounding boxes are being drawn around the labels
- The image is fed to the output video stream

## Results

## Conclusion
