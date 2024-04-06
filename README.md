## PJMRecognizer
### About the project
This project is an attempt to produce an application dedicated to learning the Polish sign alphabet: <br />

<p align="center">
  <img src="https://github.com/worthy11/PJMRecognizer/img/polski-alfabet-palcowy.jpg" alt="Polish sign alphabet"/>
</p>

Currently only the backbone network responsible for classifying letters from weebcam feed has been implemented.

### Architecture
The program uses OpenCV and Mediapipe to retrieve video feed and detect hands in subsequent frames. The underlying network consists of the following layers: <br />

|INDEX|TYPE|UNITS|ACTIVATION|
|-|:-:|:-:|:-:|
|0|INPUT|441|-|
|1|HIDDEN|256|RELU|
|2|HIDDEN|128|RELU|
|3|OUTPUT|26|SOFTMAX|

The default training parameters for the network are as follows:
- Number of epochs: **10**
- Batch size: **1**
- Optimizer: Adam
- Loss function: **Categorical Cross Entropy**
- Minimal learning rate: **0.00001**
- Learning rate reduction factor on plateau: **0.5** <br />

The input vector is a flattened 21x21 array representing the distances between each pair of the 21 hand landmarks produced by Mediapipe's hand recognizer, where a value located at `input[i][j]` represents the distance (proportional to the hand's height and width in the frame) between landmarks `i` and `j`, given by the square root of the squared difference of their respective `x` and `y` coordinates.: <br />

<p align="center">
  <img src="https://github.com/worthy11/PJMRecognizer/img/hand_landmarks.png" alt="Hand landmark indexing in Mediapipe"/>
</p>

### Usage
Required packages are listed in `requirements.txt`. <br />

The program currently has three modes of operation:
- **Recognition** - default mode; classifies the detected hand landmarks as one of 26 letters
- **Training** - press `t` to retrain the model, followed by your preferece to overwrite the previous model weights 
- **New sample creation** - press `n` when your hand is in position to save a new sample, followed by a label (a lowercase letter excluding special Polish characters); press any key other than a valid label to cancel. Press `SPACE` outside of sample creation mode to toggle between saving to the training and testing datasets. <br />

Additional functionalities include:
- **Landmark display** - press `ENTER` to toggle
- **Confidence of prediction** - indicated by the color of the displayed label (the greener the letter, the higher the confidence)