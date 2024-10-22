import face_recognition as recognition

# 画像の読み込み
image = recognition.load_image_file("./test.jpeg")

# 顔の位置を算出
face_locations = recognition.face_locations(image)

print(f'画像の中には{len(face_locations)}人の人がいます')
print(f'顔認識における位置情報は次のとおりになっています{face_locations}')