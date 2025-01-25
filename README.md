# MNIST-Data2
MNIST Digit Classification Using Convolutional Neural Networks (CNN)

Overview This project aims to classify handwritten digits from the MNIST dataset using Convolutional Neural Networks (CNN). The model uses Keras and TensorFlow for building and training the model. It achieves high accuracy with early stopping to avoid overfitting.

The dataset is divided into:
train.csv: Contains the training data with 42000 samples and 785 features (one for each pixel in the 28x28 grayscale image and the label column).
test.csv: Contains the test data with 28000 samples and 784 features.
How to Run

Data Loading and Preprocessing:

Load the MNIST dataset and normalize the pixel values.
Split the data into training and validation sets.
Convert labels to categorical format (one-hot encoded).
Model Architecture:

The model uses CNN layers with the following architecture:
Conv2D Layer (32 filters)
MaxPooling2D
Conv2D Layer (64 filters)
MaxPooling2D
Flatten Layer
Dense Layer (128 neurons) with Dropout
Output Layer with 10 neurons (for each digit) and softmax activation.
Model Training:

The model is trained using the Adam optimizer and categorical cross-entropy loss function.
Early stopping is used to prevent overfitting.
Model Evaluation:

The model is evaluated on the validation dataset and achieves high accuracy (around 99%).
Prediction and Submission:

Predictions are made on the test dataset, and the results are saved to a CSV file for submission.
Model Performance

Test Loss: 0.0398
Test Accuracy: 99.27%
Conclusion
The CNN model achieved an impressive accuracy of 99.27% on the validation set. The model is capable of accurately predicting handwritten digits and can be extended for other image classification tasks.
