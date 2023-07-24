import mediapipe as mp
import cv2
import os
import pickle
import matplotlib.pyplot as plt

# Useful in order to detect and draw all the landmarks
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Creates an instance of the Hands class from the mediapipe library.
# The Hands class is responsible for detecting hand landmarks in images and videos.
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
# static_image_mode=True: This parameter specifies that we are using the hand detection model for static images,
# not for live video streams.
# min_detection_confidence=0.3: This parameter sets the minimum confidence threshold for hand detection.
# Any detection with a confidence score below 0.3 will be ignored.

DATA_DIR = "./data"

data = []  # images
labels = []  # category for each of the image (symbols)

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):  # [:1]:
        data_aux = []  # x and y of data
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Once you have created the hands instance, you can use it to detect hands in static images by passing the
        # images to the process() method of the Hands class.
        results = hands.process(img_rgb)
        # This will return the hand landmarks (e.g., the positions of the fingertips, palm, etc.) detected in the image.

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:  # more than one hands could be detected
                '''
                # Show landmarks on img:
                mp_drawing.draw_landmarks(
                    img_rgb,  # img to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                '''

                for i in range(len(hand_landmarks.landmark)):
                    # position of landmarks --> x, y, z
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
            plt.figure(), plt.imshow(img_rgb), plt.show()
            data.append(data_aux)
            labels.append(dir_)  # the categories(symbols) - 3

f = open('data.pickle', 'wb')  # 'wb' --> writing, bytes
pickle.dump({'data': data, 'labels': labels}, f)  # create a dictionary containing these two keys
f.close()  # close the file

'''
The Pickle library is used for serializing (pickling) and deserializing (unpickling) data objects in Python programs. 
Pickle is the process of writing data to a stream and then reading it back, storing and restoring the data as Python 
objects. Pickle transforms data into Python objects such as arrays, lists, dictionaries, classes, etc., allowing data 
to be transmitted to files, databases or over the network. It includes functions such as dump(), dumps(), load() and 
loads(). You can write a Python object to a file using pickle and then restore it later.
'''

'''
In this code, data_aux, data and labels variables are used as follows:

data_aux: This variable is a temporary list that holds the handshake data for a single image. This list is created for 
each image, and the hand sign data is added to it. This data_aux list is then added to the data list.

data: This variable is a list that holds the hand sign data. Each of its elements is a child list, like data_aux, and 
contains the landmark data detected in an image. The data list contains the landmark data detected in each image.

labels: This variable is a list that holds the category information of the images. The category (symbol) of each image 
is added to the labels list. This is used to label the hand signal data according to their category.

As an example, suppose data and labels are as follows:
data = [
    [x1_1, y1_1, x2_1, y2_1, ..., x_n1, y_n1], # Hand signal data for image 1
    [x1_2, y1_2, x2_2, y2_2, ..., x_n2, y_n2], # Hand signal data for 2nd image
    ...
    [x1_m, y1_m, x2_m, y2_m, ..., x_nm, y_nm] # Hand signal data for mth image
]

labels = [
    'symbol_1', # category of the 1st image
    'symbol_2', # category of the 2nd image
    ...
    'symbol_k' # category of k. image
]

Here, each sub-list in the data list represents the hand signal data detected in an image. The labels list contains the 
category information of the images and each category represents the symbol name.

For example, if there is an image labeled as 'symbol_1' in the labels list, the hand signal data for this image will be 
found in the corresponding index in the data list.
'''