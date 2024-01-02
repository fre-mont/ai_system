# python project.py -c haarcascade_frontalface_default.xml -m output/emotion_model.hdf5

# 패키지 임포트
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import time 
import sys
import os 

# 코드 실행 시 표준 출력 비활성화
sys.stdout = open(os.devnull, 'w')

# 파일 실행 시 입력 인자 받기
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
                help="path to where the face cascade resides")
ap.add_argument("-m", "--model", required=True,
                help="path to pre-trained smile detector CNN")
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
args = vars(ap.parse_args())


### 얼굴 좌표에 따른 오프셋 적용
def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)


### input 전처리 
def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x


### 감정 라벨 
# emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

### 부정적 감정은 Angry로 모두치환 
emotion_labels = ["Angry", "Angry", "Angry", "Happy", "Sad", "Neutral", "Neutral"]
emotion_window = []

# 얼굴 추적 cascade와 smile 추적 CNN 모델 로드하기
detector = cv2.CascadeClassifier(args["cascade"])
# model = load_model(args["model"])
model = load_model(args["model"])

# 모델 입력 사이즈 정의
emotion_target_size = model.input_shape[1:3]

### 비디오 파일이 주어지지 않았다면 웹캠을 사용
if not args.get("video", False):
    camera = cv2.VideoCapture(0)
### 비디오 로드 
else:
    camera = cv2.VideoCapture(args["video"])


# 5초 동안 웹캠 틀어 감정 인식 

# while True:

start_time = time.time()
while time.time() - start_time < 5:

    ### 비디오 프레임 읽기
    (grabbed, frame) = camera.read()
    
    ##### 실패 시 종료
    if args.get("video") and not grabbed:
        break
#### 프레임 크기 조정
    #frame = camera.read()[1]
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gray = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frameClone = frame.copy()

    ##### 얼굴 검출
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                          minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for face_coordinates in faces:
        #### 얼굴 좌표에 따른 오프셋 적용
        x1, x2, y1, y2 = apply_offsets(face_coordinates, (20, 40))
        gray_face = gray[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        
        ### 감정 예측
        emotion_prediction = model.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        
        emotion_window.append(emotion_text)

    ### 인풋 프레임에서 얼굴 검출
    rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)
    # 얼굴 바운딩 박스 
    for (fX, fY, fW, fH) in rects:

        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (28, 28))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        ### 감정 라벨 표시 (추론 결과)
        cv2.putText(frameClone, emotion_text, (fX, fY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                      (0, 0, 255), 2)

    ### 카메라 화면 표시
    cv2.imshow("Face", frameClone)
    # q누르면 끔 
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()


text = emotion_text 

# 파일에 문자열 저장
with open("result.txt", "w") as file:
    file.write(text)
