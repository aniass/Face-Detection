import cv2

FACE_CASCADE_PATH = 'models\haarcascade_frontalface_default.xml'
EYE_CASCADE_PATH = 'models\haarcascade_eye.xml'
SMILE_CASCADE_PATH = 'models\haarcascade_smile.xml'

face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)
smile_cascade = cv2.CascadeClassifier(SMILE_CASCADE_PATH)


def convert_color(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def detect_features(gray, images):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(0, 0), flags=cv2.CASCADE_SCALE_IMAGE)
    
    # Draw a rectangle around the each faces, eye and smile
    for (x, y, w, h) in faces:
        cv2.rectangle(images, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = images[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 2)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.5, 22)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (255, 0, 130), 2)
            cv2.putText(images, "Happy", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return images


def main():
    girl_img = cv2.imread('picture.jpg')
    girl_img_rgb = cv2.cvtColor(girl_img, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(girl_img, cv2.COLOR_BGR2GRAY)

    result_img = detect_features(gray_img, girl_img_rgb)
    converted_img = convert_color(result_img)

    cv2.imshow('Image', converted_img)
    cv2.imwrite('Image.png', converted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
