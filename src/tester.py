import cv2
import numpy as np

def test(sample):
    opencv_net = cv2.dnn.readNetFromONNX('./pjmrecognizer.onnx')
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
        17: 'T',
        18: 'U',
        19: 'W',
        20: 'Y',
        21: 'Z'
    }
    
    opencv_net.setInput(sample)
    out = opencv_net.forward()
    print(out)

    imagenet_class_id = np.argmax(out)
    confidence = out[0][imagenet_class_id]
    print("* class ID: {}, label: {}".format(imagenet_class_id, classes[imagenet_class_id]))
    print("* confidence: {:.4f}\n".format(confidence))