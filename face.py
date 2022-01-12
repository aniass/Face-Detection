import cv2

face_path = 'C:\\Users\\PC\\Anaconda3\\envs\\opencv\\Library\\etc\\haarcascades\\haarcascade_frontalface_default.xml'
eye_path = 'C:\\Users\\PC\\Anaconda3\\envs\\opencv\\Library\\etc\\haarcascades\\haarcascade_eye.xml'
smile_path = 'C:\\Users\\PC\\Anaconda3\\envs\\opencv\\Library\\etc\\haarcascades\\haarcascade_smile.xml'

face_cascade = cv2.CascadeClassifier(face_path)
eye_cascade = cv2.CascadeClassifier(eye_path)
smile_cascade = cv2.CascadeClassifier(smile_path)


def convert_color(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def detection(gray, images):
    faces_detect = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(0, 0),
                                                 flags=cv2.CASCADE_SCALE_IMAGE)
    # Draw a rectangle around the each faces, eye and smile
    for (x, y, w, h) in faces_detect:
        cv2.rectangle(images, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = images[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 2)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        smile = smile_cascade.detectMultiScale(roi_gray, 1.5, 22)
        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh),
                          (255, 0, 130), 2)
            cv2.putText(images, "Happy", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)
    return images


girl_img = cv2.imread('picture.jpg')
girl_img2 = cv2.cvtColor(girl_img, cv2.COLOR_BGR2RGB)
gray_img = cv2.cvtColor(girl_img, cv2.COLOR_BGR2GRAY)

picture = detection(gray_img, girl_img2)
image = convert_color(picture)

cv2.imshow('Image', image)
cv2.imwrite('Image.png', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
