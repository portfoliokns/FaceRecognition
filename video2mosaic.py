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

# 動画の読み込み
cascade_file= cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
clas = cv2.CascadeClassifier(cascade_file)
cap = cv2.VideoCapture("./videoplayback.mp4")

# 保存用のVideoWriterオブジェクトを作成
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
output_file = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v'はMP4形式
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # 顔の位置を算出
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = recognition.face_locations(rgb_frame)

    if face_locations:
        # 描画
        top, right, bottom, left = face_locations[0]
        x, y, w, h, size = left, top, right - left, bottom - top, 20
        image_cv2 = mosaic(frame, x, y, w, h, size)

    else:
        image_cv2 = frame.copy()

    out.write(image_cv2)

cap.release()
out.release()
cv2.destroyAllWindows()
