import cv2

def capture_list():
    imlist = []
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("test")

    img_counter = 0

    while img_counter<4:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Exiting")
            return imlist
        elif k%256 == 32:
            # SPACE pressed
            imlist.append(frame)
            print(type(frame))
            img_counter += 1
            print("Captured intersection {}".format(img_counter))
    cam.release()
    cv2.destroyAllWindows()
    return imlist
