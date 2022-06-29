import cv2 
import numpy as np
#загружаем необходимые классификаторы XML.
face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_detect = cv2.CascadeClassifier('haarcascade_eye.xml')
# загружаем изображение
img=cv2.imread('7.jpg')
# переводим в оттенки серого
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Находим лицо
faces = face_detect.detectMultiScale(gray, 1.3, 5)
# Рисуем прямоугольники на лице
for (x,y,w,h) in faces:
    img=cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
# Находим глаза
    eyes = eye_detect.detectMultiScale(roi_gray)
# Рисуем прямоугольники над глазами
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
# Результаты программы выводим на экран изображение
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

    


