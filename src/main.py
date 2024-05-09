from data_loader import *
from model import *
from functions import *
from tester import *

# Model able to accept new samples and train during runtime; newly learned information can be verified immediately after training
def main():
    recognizer = Model(classes, epochs=10, batch_size=1, learning_rate=0.00001)
    capture = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hand_recognizer = mp_hands.Hands()
    mp_drawing = mp.solutions.drawing_utils
    work_mode = 114 # Recognize ('r') by default
                    # Press 't' to retrain the model
                    # Press 'n' followed by the label to add new sample
    save_mode = 'training' # Add samples to training set by default
                           # Press SPACE to toggle
    display_landmarks = True # Landmarks displayed by default
                             # Press ENTER to toggle
    while 1:
        not_empty, img = capture.read()
        if not_empty:
            h, w, c = img.shape
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hand_recognizer.process(imgRGB)

            match work_mode:
                case 114:
                    # Recognize
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            sample = ConvertToNumpy(hand_landmarks)
                            sample = np.expand_dims(sample, axis=-1)
                            sample = np.expand_dims(sample, axis=0)
                            label, confidence = recognizer.Predict(sample)  # Feed to network
                            if confidence < .5: # If network is not confident enough, use simple prediction
                                sample = sample[0, :, 0].reshape((21, 21))
                                label_1, confidence_1 = RecognizeLetter(sample)
                                if confidence_1 > confidence:
                                    confidence = confidence_1
                                    label = label_1
                            color = (0, confidence*255, (1-confidence)*255) # Set color depending on confidence
                            cv2.putText(img, label, (w // 2, 70), cv2.FONT_HERSHEY_PLAIN, 5, color, 5) # Display result
                            if display_landmarks:
                                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                case 110:
                    # Add new samples
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            sample = ConvertToNumpy(hand_landmarks)

                            print('Waiting for label')
                            label = cv2.waitKey() - 97
                            if chr(label) not in letters: # If not a lowercase letter
                                print('Invalid label. Cancelled saving sample')

                            else:
                                filepath = 'pjm_' + save_mode + '_set.csv'
                                if AddSample(sample, letters.find(chr(label)), filepath):
                                    print('Saved sample with label {}'.format(chr(label)))
                                else:
                                    print('Failed to save sample')
                        
                case 116:
                    # Retrain model
                    print('Save to checkpoint? [y/n]')
                    choice = input()
                    if choice != 'y' and choice != 'n':
                        print('Failed to retrain model')
                    else:
                        if choice == 'n':
                            choice = False
                        else:
                            choice = True
                        recognizer.TrainModel(choice)
                        print('Retrained model successfully')

                # DO NOT USE - remove sample manually if you make a mistake
                # case 100:
                #     # Delete last sample
                #     filepath = 'pjm_' + save_mode + '_set.csv'
                #     if DeleteLastSample(filepath):
                #         print('Removed sample from {} set'.format(filepath))
                #     else:
                #         print('Failed to removed sample from {} set'.format(filepath))

            work_mode = 114
                        
            cv2.imshow('Webcam', img)
            key = cv2.waitKey(1)
            if key == 32:
                if save_mode == 'testing':
                    save_mode = 'training'
                    print('Switching to training set')
                else:
                    save_mode = 'testing'
                    print('Switching to testing set')
            elif key == 13:
                display_landmarks = not display_landmarks
            elif key != -1:
                work_mode = key

main()