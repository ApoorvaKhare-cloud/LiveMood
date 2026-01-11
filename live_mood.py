import cv2
from deepface import DeepFace

# Open camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Define colors for emotions
emotion_colors = {
    'happy': (0, 255, 255),
    'sad': (255, 0, 0),
    'angry': (0, 0, 255),
    'surprise': (255, 255, 0),
    'fear': (128, 0, 128),
    'disgust': (0, 128, 0),
    'neutral': (200, 200, 200)
}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Analyze emotion
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        dominant_emotion = result[0]['dominant_emotion']
        face_region = result[0]['region']  # get face rectangle

        x, y, w, h = face_region['x'], face_region['y'], face_region['w'], face_region['h']

        # Draw rectangle around face
        color = emotion_colors.get(dominant_emotion, (0, 255, 0))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

        # Draw emotion text above rectangle
        cv2.putText(frame, dominant_emotion.upper(), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)
    except Exception as e:
        # In case face not detected, skip
        pass

    # Optional: Add a semi-transparent overlay for aesthetics
    overlay = frame.copy()
    alpha = 0.2
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (50, 50, 50), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.imshow("LiveMood - Emotion Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

