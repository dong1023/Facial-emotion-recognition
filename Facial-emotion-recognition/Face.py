import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 加载情绪识别模型
model = load_model('model_v6_23.hdf5')

# 定义情绪标签o
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    # 读取视频流的帧
    ret, frame = cap.read()

    # 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 人脸检测
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    # 对于每张检测到的人脸
    for (x, y, w, h) in faces:
        # 提取人脸区域并进行预处理（缩放、归一化）
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = face_roi / 255.0
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = np.expand_dims(face_roi, axis=-1)

        # 进行情绪预测
        emotion_scores = model.predict(face_roi)[0]

        # 获取每种情绪的得分
        emotion_dict = dict(zip(emotion_labels, emotion_scores))

        # 在图像上绘制情绪得分
        for i, (emotion, score) in enumerate(emotion_dict.items()):
            text = f'{emotion}: {score:.2f}'
            cv2.putText(frame, text, (x, y-30-i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 绘制人脸矩形框
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # 显示结果图像
    cv2.imshow('Emotion Recognition', frame)

    # 按下'q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 关闭摄像头和窗口
cap.release()
cv2.destroyAllWindows()
