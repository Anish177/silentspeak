from flask import Flask, redirect, render_template, Response
import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array, load_img
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# import pyttsx3

app = Flask(__name__)
# camera = cv2.VideoCapture(0)  # Use 0 for default camera


# # Load your pre-trained model
# model = load_model('model.h5')

# # Mapping of indices to gestures
# gesture_mapping = {
#     0: 'A', 1: 'B', 2: 'C'
# }

# def detect_gesture(frame):
#     # Resize the frame to match the input shape expected by your model
#     resized_frame = cv2.resize(frame, (64, 64))
#     # Convert the frame to an array
#     resized_frame_array = img_to_array(resized_frame)
#     # Expand the dimensions to match the model's expected input shape
#     input_frame = np.expand_dims(resized_frame_array, axis=0)
#     # Preprocess the input frame
#     preprocessed_frame = preprocess_input(input_frame)
#     # Perform prediction
#     prediction = model.predict(preprocessed_frame)[0]
#     predicted_class = np.argmax(prediction)
#     gesture_label = gesture_mapping.get(predicted_class, 'Unknown')
#     return gesture_label


# def generate_frames():
#     while True:
#         success, frame = camera.read()
#         if not success:
#             break

#         # Perform gesture detection on each frame
#         gesture_label = detect_gesture(frame)

#         # Draw the gesture label on the frame
#         cv2.putText(frame, f'Gesture: {gesture_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         pyttsx3.speak(gesture_label)

        
#         # Encode the frame in JPEG format
#         _, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/try_gestures')
def gestures():
    return redirect('http://localhost:9967/')

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/community', endpoint='path2')
def community():
    return render_template('community.html')

@app.route('/learn', endpoint='path3')
def learn():
    return render_template('learn.html')

@app.route('/self_paced', endpoint='path4')
def self():
    return render_template('self_paced.html')

@app.route('/wtg', endpoint='path5')
def wtg():
    return render_template('wtg.html')

@app.route('/videos', endpoint='path6')
def videos():
    return render_template('videos.html')

@app.route('/call', endpoint='path7')
def call():
    return redirect("http://127.0.0.1:5500/index.html")
# @app.route('/back', endpoint='path7')
# def back():
#     camera.release()
#     camera = cv2.VideoCapture(0)  # Use 0 for default camera
#     return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)