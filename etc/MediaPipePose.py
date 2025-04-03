import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture("sample_video.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            h, w, _ = frame.shape
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    cv2.imshow("Pose Detection", frame)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()