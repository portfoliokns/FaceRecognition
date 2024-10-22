import face_recognition as recognition
import cv2 as cv2

# 画像の読み込み
image = recognition.load_image_file("./test.jpeg")

# 顔の位置を算出
face_locations = recognition.face_locations(image)

# RGB2BGR 
image_cv2 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# 描画
color = (255, 255,255)
for (top, right, bottom, left) in face_locations:
  cv2.rectangle(image_cv2, (left,top), (right, bottom), color, 2)

print(f'画像の中には{len(face_locations)}人の人がいます')

cv2.imshow("Detected Faces", image_cv2)
cv2.waitKey(0)
cv2.destroyAllWindows()