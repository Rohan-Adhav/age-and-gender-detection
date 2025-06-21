import cv2 as cv
import time
from collections import deque, Counter

def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    h, w = frame.shape[:2]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300),
                                [104, 117, 123], swapRB=False)
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frameOpencvDnn, bboxes

# Load models
faceNet = cv.dnn.readNet("opencv_face_detector_uint8.pb", "opencv_face_detector.pbtxt")
ageNet = cv.dnn.readNet("age_net.caffemodel", "age_deploy.prototxt")
genderNet = cv.dnn.readNet("gender_net.caffemodel", "gender_deploy.prototxt")

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
           '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

cap = cv.VideoCapture(0)
padding = 20
gender_history = deque(maxlen=20)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frameFace, bboxes = getFaceBox(faceNet, frame)
    if not bboxes:
        cv.imshow("Age Gender Demo", frameFace)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    for bbox in bboxes:
        face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                     max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]

        # Correct swapRB
        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227),
                                    MODEL_MEAN_VALUES, swapRB=True)

        # Gender prediction
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender_confidence = genderPreds[0].max()
        gender_index = genderPreds[0].argmax()
        predicted_gender = genderList[gender_index]

        print("Raw gender prediction:", genderPreds[0])
        print(f"Predicted: {predicted_gender}, Confidence: {gender_confidence:.2f}")

        # Use confidence threshold
        if gender_confidence >= 0.65:
            gender_history.append(predicted_gender)
        else:
            gender_history.append("Uncertain")

        # Most frequent recent gender prediction
        if gender_history:
            final_gender = Counter(gender_history).most_common(1)[0][0]
        else:
            final_gender = "Uncertain"

        # Age prediction
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age_index = agePreds[0].argmax()
        age = ageList[age_index]

        print(f"Predicted Age: {age}, Confidence: {agePreds[0][age_index]:.2f}")

        label = f"{final_gender}, {age}"
        cv.putText(frameFace, label, (bbox[0], bbox[1] - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv.imshow("Age Gender Demo", frameFace)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
