import cv2
import mediapipe as mp
import pickle
import numpy as np

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Creates an instance of the Hands class from the mediapipe library.
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Load model.pickle
model_dict = pickle.load(open('./model.pickle', 'rb'))
model = model_dict['model']

# Labels
labels_dict = {0: 'A', 1: 'B', 2: 'L'}

if not cap.isOpened():
    print("Failed to capture")

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    h, w, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # img to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)

        x1 = int(min(x_) * w) - 10
        y1 = int(min(y_) * h) - 10

        x2 = int(max(x_) * w) - 10
        y2 = int(max(y_) * h) - 10

        # prediction --> list of only one element
        prediction = model.predict([np.asarray(data_aux)])

        predicted_character = labels_dict[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4 )
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break  # stop recording when press q

cap.release()  # Stop capture
cv2.destroyAllWindows()  # Close windows
