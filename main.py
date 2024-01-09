import cv2, os
import numpy as np

def display_only_orange(image_path, lower_hsv, upper_hsv):
    img = cv2.imread(image_path)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Create mask of orange things
    orange_mask = cv2.inRange(img_hsv, lower_hsv, upper_hsv)
    kernel = np.ones((5, 5), np.uint8)
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel)
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Rule out items too small
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 4000:
            cv2.drawContours(orange_mask, [contour], 0, 0, -1)

    result_image = cv2.bitwise_and(img, img, mask=orange_mask)
    
    # Check for discs
    cnts = cv2.findContours(orange_mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)[-2]
    cnt = sorted(cnts, key=cv2.contourArea)[-1]
    
    epsilon = cv2.arcLength(cnt, True) * 0.0005
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    cv2.drawContours(result_image, [approx], -1, (255,0,0), 3, cv2.LINE_AA)
    
    # Check for stacked disks
    cnts, heirs = cv2.findContours(orange_mask, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    for heir in heirs[0]:
        if heir[3] > 0:
            cv2.drawContours(result_image, [cnts[heir[3]]], -1, (255,0,0), 3)

    return result_image

if __name__ == "__main__":
    listDir = os.listdir("./ref")
    for filename in listDir:
        if filename.endswith('.png'):
            img = display_only_orange(f"./ref/{filename}", (5, 100, 100), (30, 255, 255))
            img_scaled = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow(filename[:-4], img_scaled)

    cv2.waitKey(0)
    cv2.destroyAllWindows()