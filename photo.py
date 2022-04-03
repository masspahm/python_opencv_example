import cv2
import sys
import os

if len(sys.argv) < 2:
    sys.exit("Usage: python photo.py <image_path> <output_path>")

if not os.path.exists(sys.argv[2]):
    os.makedirs(sys.argv[2])

imagePath = sys.argv[1]
cascPath = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

image = cv2.imread(imagePath)
row, col = image.shape[:2]
if row > 900 or col > 900:
    row = round(row/4)
    col = round(col/4)
resized = cv2.resize(image, (col, row),  interpolation=cv2.INTER_NEAREST)

row, col = resized.shape[:2]

gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(50, 50),
)

rx = 113*3
ry = 151*3
filename = 'masspahmi.jpg'
resultPath = sys.argv[2]

cv2.imwrite(resultPath + "/resized_"+filename, resized)
cv2.imwrite(resultPath + "/ori_"+filename, image)
if len(faces) < 1:
    print("Error: Face not found")
    sys.exit(1)
if len(faces) > 1:
    print("Error: Found more than one face")
    sys.exit(1)

for (x, y, w, h) in faces:
    cx = x + w/2
    cy = y + h/2
    startX = int(cx - round(rx/2))
    startY = int(cy - round(ry/2))
    width = int(startX+rx)
    height = int(startY+ry)
    faces = resized[startY:height, startX:width]
    cv2.imwrite(resultPath + "/hasil_crop_"+filename, faces)

    newgray = cv2.cvtColor(faces, cv2.COLOR_BGR2GRAY)

    (thresh, im_bw) = cv2.threshold(newgray, 128, 255, cv2.THRESH_BINARY)
    im_bw = cv2.threshold(newgray, thresh, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite(resultPath + "/hitam_putih_"+filename, im_bw)

    blue = faces.copy()
    blue[:, :, 1] = 0
    blue[:, :, 2] = 0
    cv2.imwrite(resultPath + "/warna_biru_"+filename, blue)

    green = faces.copy()
    green[:, :, 0] = 0
    green[:, :, 2] = 0
    cv2.imwrite(resultPath + "/warna_hijau_"+filename, green)

    red = faces.copy()
    red[:, :, 0] = 0
    red[:, :, 1] = 0
    cv2.imwrite(resultPath + "/warna_merah_"+filename, red)

    bordersize = 10
    border = cv2.copyMakeBorder(
        faces,
        top=bordersize,
        bottom=bordersize,
        left=bordersize,
        right=bordersize,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255]
    )
    cv2.imwrite(resultPath + "/bingkai_putih_"+filename, border)
