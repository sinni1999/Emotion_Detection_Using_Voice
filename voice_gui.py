import os
import glob
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tkinter as tk
from tkinter import filedialog
from pydub import AudioSegment
import sounddevice as sd
from PIL import Image, ImageTk 

import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = '/path/to/ffmpeg'
import warnings
warnings.filterwarnings("ignore", message=".*Couldn't find ffmpeg or avconv.*")



def extract_features(file_path):
    audio, sr = librosa.load(file_path, res_type='kaiser_fast')
    features = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).T, axis=0)
    return features

data = []
labels = []

# Provide the correct directory path here
directory_path = r"C:\MAchine Learning Project\Emotion_Detection_Using_Voice\TESS Toronto emotional speech set data"


for folder_name in os.listdir(directory_path):
    folder_path = os.path.join(directory_path, folder_name)
    if os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            print(file_path)
            # Extract features and labels here
            features = extract_features(file_path)
            data.append(features)
            labels.append(folder_name)  # Assuming folder name represents labels

    
        # Process the file (e.g., load data, extract features, etc.)
        # Append data and labels accordingly




import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, TimeDistributed
from sklearn.preprocessing import LabelEncoder

# Check if data and labels_encoded have enough samples
if len(data) == 0 or len(labels) == 0:
    print("Error: Data or labels_encoded is empty.")
else:
    # Assuming you have data and labels_encoded defined earlier
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, random_state=42)

    # Convert data to numpy arrays and add a new axis
    X_train = np.array(X_train)[:, np.newaxis, :]
    X_test = np.array(X_test)[:, np.newaxis, :]

    # Define the model architecture
    model = Sequential()
    model.add(TimeDistributed(Dense(256, activation='relu'), input_shape=(1, X_train.shape[2])))
    model.add(Dropout(0.5))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(len(label_encoder.classes_), activation='softmax'))

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test, y_test)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)




def predict_emotion(audio_file):
    features = extract_features(audio_file)
    features = features[np.newaxis, np.newaxis, :]  
    print("Features shape:", features.shape)
    print("Features:", features)

    predicted_probabilities = model.predict(features)
    print("Predicted probabilities shape:", predicted_probabilities.shape)
    print("Predicted probabilities:", predicted_probabilities)

    predicted_label_index = np.argmax(predicted_probabilities)
    print("Predicted label index:", predicted_label_index)

    predicted_emotion = label_encoder.classes_[predicted_label_index]
    print("Predicted emotion:", predicted_emotion)


    # Emotion mapping for TESS dataset
    emotion_mapping = {
        'YAF_angry': 'ANGRY',
        'YAF_disgust': 'DISGUST',
        'YAF_fear': 'FEAR',
        'YAF_happy': 'HAPPY',
        'YAF_neutral': 'NEUTRAL',
        'YAF_pleasant_surprised': 'SURPRISED',
        'YAF_sad': 'SAD',
        'OAF_angry': 'ANGRY',
        'OAF_disgust': 'DISGUST',
        'OAF_Fear': 'FEAR',
        'OAF_happy': 'HAPPY',
        'OAF_neutral': 'NEUTRAL',
        'OAF_Pleasant_surprised': 'SURPRISED',
        'OAF_Sad': 'SAD',
    }


    recognizable_emotion = emotion_mapping.get(predicted_emotion)
    return recognizable_emotion


import tkinter as tk

class EmotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice Prediction App")
        self.root.configure(bg='skyblue')
        self.prediction_history = []  
        self.show_home_page()

        
        
    def show_home_page(self):
        self.clear_window()
        
        label = tk.Label(self.root, text="Voice Prediction App", font=('Helvetica bold', 16))
        label.pack(pady=20)
        
        button = tk.Button(self.root, text="Audio Prediction", command=self.show_audio_page, bg='lightblue')
        button.pack()
        
        button_history = tk.Button(self.root, text="Prediction History", command=self.show_history_page, bg='lightgreen')
        button_history.pack(pady=10)
        
        about_button = tk.Button(self.root, text="About The App", command=self.show_about_page, bg='lightblue')
        about_button.pack(pady=10)
        
    def show_audio_page(self):
        self.clear_window()
        
        canvas = tk.Canvas(self.root, width=500, height=500, bg='skyblue')
        canvas.pack()
        
        label1 = tk.Label(self.root, text='Voice Emotion Prediction', font=('Helvetica bold', 26))
        canvas.create_window(250, 50, window=label1)
        
        def upload_audio():
            file_path = filedialog.askopenfilename(filetypes=[("TESS toronto emotional speech data set", "*.wav")])
            if file_path:
                predicted_emotion = predict_emotion(file_path)
                label2.config(text=predicted_emotion)
                
                self.prediction_history.append((os.path.basename(file_path), predicted_emotion))
                
                
                
        button1 = tk.Button(self.root, text='Upload Audio', command=upload_audio, bg='lightgreen')
        canvas.create_window(250, 150, window=button1)
        
        label2 = tk.Label(self.root, text='Predicted Emotion Will Be Displayed Here')
        canvas.create_window(250, 200, window=label2)
        
       
        
        back_button = tk.Button(self.root, text="Back to Home", command=self.show_home_page)
        canvas.create_window(250, 400, window=back_button)
        
    def show_history_page(self):
        self.clear_window()
        
        canvas = tk.Canvas(self.root, width=500, height=500, bg='lightgreen')
        canvas.pack()
        
        label = tk.Label(self.root, text="Prediction History", font=('Helvetica bold', 16))
        canvas.create_window(250, 50, window=label)
        
        if self.prediction_history:
            for index, (file_name, predicted_emotion) in enumerate(self.prediction_history, start=1):
                history_text = f"{index}. File: {file_name}, Emotion: {predicted_emotion}"
                history_label = tk.Label(self.root, text=history_text)
                canvas.create_window(250, 100 + index * 30, window=history_label)
        else:
            no_history_label = tk.Label(self.root, text="No prediction history available.")
            canvas.create_window(250, 150, window=no_history_label)
        
        back_button = tk.Button(self.root, text="Back to Home", command=self.show_home_page)
        canvas.create_window(250, 450, window=back_button)
        
    def show_about_page(self):
        self.clear_window()
        
        canvas = tk.Canvas(self.root, width=500, height=500, bg='skyblue')
        canvas.pack()
        
        label = tk.Label(self.root, text="About The Software", font=('Helvetica bold', 16))
        canvas.create_window(250, 50, window=label)
        
        about_text = ("Hello Everyone !! "
                      " Speech Emotion Recognition is a software that recognizes the emotion of the user."
                      " All of the audio files in this software should be inputted with '.wav' extension."
                     
                      )
        
        about_label = tk.Label(self.root, text=about_text, wraplength=400)
        canvas.create_window(250, 150, window=about_label)
        
        back_button = tk.Button(self.root, text="Back to Home", command=self.show_home_page)
        canvas.create_window(250, 400, window=back_button)
        
    def clear_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionApp(root)
    root.mainloop()




