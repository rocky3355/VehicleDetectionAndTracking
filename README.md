# Vehicle Detection and Tracking
The goal of this project was to implement a pipeline for detecting vehicles in a video stream and to draw bounding boxes around them.

## The pipeline
The pipeline consists of the following steps:
### 1. Setup
- Create Keras model for deep learning
- Load training images
- Train the model
- Save the model
If the model has been created already, the saved model will be loaded instead of executing these steps.......
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
