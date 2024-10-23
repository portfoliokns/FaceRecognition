import face_recognition as recognition
import cv2 as cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# テキストを顔に追加する関数
def draw_japanese_text(img, text, x, y):
    # OpenCVの画像をPillowの画像に変換
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # フォントを設定（日本語フォントのパスを指定）
    font = ImageFont.truetype("NotoSansJP-VariableFont_wght.ttf", size=30)

    # テキストを描画
    draw.text((x, y), text, font=font, fill=(255, 255, 255))

    # Pillowの画像をOpenCVの画像に変換
    img_out = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img_out

# 動画の読み込み
cascade_file= cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
clas = cv2.CascadeClassifier(cascade_file)
cap = cv2.VideoCapture("./test.mov")

# 保存用のVideoWriterオブジェクトを作成
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
output_file = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v'はMP4形式
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

mosaic_para = 20
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # 顔の位置を算出
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = recognition.face_locations(rgb_frame)
    image_cv2 = frame.copy()

    # 描画
    for (top, right, bottom, left) in face_locations:
        x, y, w, h = left, top, right - left, bottom - top
        image_cv2 = draw_japanese_text(image_cv2, "ここに文字を入れて", x, y)  # 顔の上に「顔」と表示

    out.write(image_cv2)

cap.release()
out.release()
cv2.destroyAllWindows()
