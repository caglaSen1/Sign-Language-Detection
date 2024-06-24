# Sign-Language-Detection
This project focuses on detecting and recognizing sign language gestures using computer vision techniques. Utilizing the MediaPipe library for hand landmark detection, the system processes video frames to identify and extract hand landmarks. These landmarks are then used to classify different sign language symbols through a trained RandomForest model.

The project involves several steps:

* Data Collection: Capturing hand gesture images for different sign language symbols using a webcam and storing them in a structured directory.
* Landmark Detection: Using MediaPipe to detect hand landmarks (x, y coordinates) from the collected images.
* Data Preparation: Extracting and labeling the hand landmarks data, then saving it for model training.
* Model Training: Training a RandomForestClassifier on the extracted hand landmarks data to recognize sign language symbols.
* Real-time Detection: Implementing a real-time system that captures video from a webcam, processes each frame to detect hand landmarks, and uses the trained model to classify the detected gestures.

By detecting hand landmarks and mapping them to predefined sign language symbols, this project enables real-time recognition and interaction, providing a valuable tool for communication using sign language.
