import face_recognition as recognition
import cv2 as cv2

# モザイク処理
def mosaic(img, x, y, w, h, size):
    # モザイクをかける領域を取得
    (x1, y1, x2, y2) = (x, y, x+w, y+h) #モザイク処理をかける領域を指定
    
    img_rec = img[y1:y2, x1:x2] #スライスでモザイク処理をかける領域を取得
    
    # モザイク処理：縮小ー＞拡大
    img_small = cv2.resize(img_rec, (size, size)) #sizeと同じ値に画像を縮小
    img_mos = cv2.resize(img_small, (w, h), interpolation=cv2.INTER_AREA) #元のサイズに拡大

    # 画像にモザイク画像を重ねる
    img_out = img.copy()
    img_out[y1:y2, x1:x2] = img_mos #スライス機能で値を上書き
    return img_out

# 画像の読み込み
image = recognition.load_image_file("./test.jpeg")

# 顔の位置を算出
face_locations = recognition.face_locations(image)

# RGB2BGR 
image_cv2 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# 描画
for (top, right, bottom, left) in face_locations:
  x, y, w, h, size = left, top, right - left, bottom - top, 5
  image_cv2 = mosaic(image_cv2, x, y, w, h, size)

print(f'画像の中には{len(face_locations)}人の人がいます')

cv2.imshow("Detected Faces", image_cv2)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("new.png",image_cv2)


