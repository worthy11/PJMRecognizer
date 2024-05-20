import cv2
import numpy as np
import mediapipe as mp
import csv
import os

letters = "abcdefghiklmnoprsuwyz"
classes: dict[int, str] = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    9: 'K',
    10: 'L',
    11: 'M',
    12: 'N',
    13: 'O',
    14: 'P',
    15: 'R',
    16: 'S',
    17: 'U',
    18: 'W',
    19: 'Y',
    20: 'Z'
}
landmarks = {
    0: 'WRIST',
    1: 'THUMB_CMC',
    2: 'THUMB_MCP',
    3: 'THUMB_IP',
    4: 'THUMB_TIP',
    5: 'INDEX_FINGER_MCP',
    6: 'INDEX_FINGER_PIP',
    7: 'INDEX_FINGER_DIP',
    8: 'INDEX_FINGER_TIP',
    9: 'MIDDLE_FINGER_MCP',
    10: 'MIDDLE_FINGER_PIP',
    11: 'MIDDLE_FINGER_DIP',
    12: 'MIDDLE_FINGER_TIP',
    13: 'RING_FINGER_MCP',
    14: 'RING_FINGER_PIP',
    15: 'RING_FINGER_DIP',
    16: 'RING_FINGER_TIP',
    17: 'PINKY_MCP',
    18: 'PINKY_PIP',
    19: 'PINKY_DIP',
    20: 'PINKY_TIP'
}

def ComputeDistances(arr: np.array, ndims: int) -> np.array:
    w = np.max(arr[:, 0]) - np.min(arr[:, 0])
    h = np.max(arr[:, 1]) - np.min(arr[:, 1])
    distances = np.empty(441)
    for i in range(21):
        for j in range(21):
            dx = (arr[j][0] - arr[i][0]) / w
            dy = (arr[j][1] - arr[i][1]) / h
            distances[i*21+j] = np.sqrt(dx**2 + dy**2)
    return distances

def ConvertToNumpy(hand_landmarks) -> np.array:
    coords = list()
    
    for landmark in hand_landmarks.landmark:
        coords.append([landmark.x, landmark.y])
    received = np.array(coords)
    observed = ComputeDistances(received, 2)
    return observed

# Predictions are made by comparing distances between landmarks with an established base (data/base.npy)
def RecognizeLetter(sample: np.array) -> str:
    expected = np.load('src/data/base.npy')
    errors = np.empty(len(classes)-1)

    for letter in range(len(classes)-1):
        errors[letter] = np.sum(((expected[letter] - sample) / 441)**2)
    label_idx = np.argmin(errors)
    if label_idx > 15: # Q not handled yet
        label_idx += 1
    label = classes[label_idx]
    min_error = np.min(errors)
    confidence = (np.sum(errors) - min_error) / np.sum(errors)
    return label, confidence

def AddSample(sample: np.array, label: str, filepath: str='pjm_testing_set.csv') -> bool:
    sample = list([str(_) for _ in sample])
    with open('src/data/'+filepath, 'a', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        if os.path.getsize('./src/data/{}'.format(filepath)) == 0:
            columns = list()
            columns.append('label')
            for _, data in enumerate(sample):
                columns.append('dist{}'.format(_))
            writer.writerow(columns)
        writer.writerow([label] + sample)
    return 1

# Base for simple predictions - not recommended to use, but if you must:
# Show the entire alphabet (skip special polish characters and Q - there should be 25 letters in total) in order
# Press SPACE each time you wish to save a letter
def CreateBase() -> None:
    capture = cv2.VideoCapture(0)
    img: cv2.Mat
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hand = mp_hands.Hands()
    
    coords = list()
    letters = list()
    curr_record = 0
    record = False

    while(True):
        not_empty, img = capture.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = img.shape

        if not_empty:
            results = hand.process(imgRGB)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    if record:
                        for point in hand_landmarks.landmark:
                            coords.append([point.x * w, point.y * h])
                        arr = np.array(coords)
                        letters.append(arr)
                        coords.clear()
                        print('Saved letter {} successfully'.format(curr_record))
                    mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Image', img)
        record = False
        key = cv2.waitKey(1)
        if key == 32:
            record = True
            curr_record += 1
        elif key == 114:
            coords.pop()
        elif key == 113:
            break

    letters = np.array(letters)
    distances = ComputeDistances(letters, 3)
    np.save('src/data/base.npy', distances)
    print('Base created successfully')