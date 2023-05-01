# Face-Mask-Detection

Break Down of the steps

1)Mounting Google Drive: This step is used to mount your Google Drive in Google Colab to access files stored in it.

2)Importing libraries: The necessary libraries are imported, including OpenCV (cv2), TensorFlow, and Keras.

3)Defining paths and categories: The data path is set, and the categories are listed.

4)Creating label dictionary: A dictionary is created to map categories to numerical labels.

5)Resizing and preprocessing images: The images are loaded, resized to a fixed size, converted to grayscale, and appended to the data list. Corresponding labels are also appended to the target list.

6)Preprocessing data: The data and target lists are converted to NumPy arrays and normalized.

7)Saving data and target: The preprocessed data and target arrays are saved as numpy files.

8)Loading data and target: The saved numpy files are loaded.

9)Building the model: A sequential model is created in Keras. It consists of two convolutional layers with activation functions, followed by max-pooling layers. Then, there is a flatten layer, dropout layer, and two fully connected (dense) layers with ReLU and softmax activations.

10)Compiling the model: The model is compiled with categorical cross-entropy loss, Adam optimizer, and accuracy metric.

11)Splitting the data: The data is split into training and testing sets using the train_test_split function.

12)Training the model: The model is trained on the training data for a specified number of epochs, with a ModelCheckpoint to save the best model based on validation loss.

13)Plotting the training history: The loss and accuracy curves are plotted using the training history.

14)Evaluating the model: The model is evaluated on the test data.

15)Loading the trained model: The best model from the checkpoints is loaded.

16)Face detection and mask prediction: The code captures video frames from the webcam, detects faces using Haar cascades, resizes and normalizes the face image, and predicts whether a mask is present or not using the trained model. The bounding box and label are drawn on the face in the video frame.
