import cv2

def main():
   
    img = cv2.imread("img.jpg", cv2.IMREAD_GRAYSCALE)
    height, width = img.shape[:2]
    histogram = [0] * 256

    for i in range(height):
        for j in range(width):
            # Piksel değerini histograma ekle
            histogram[img[i, j]] += 1

    for i in range(256):
        cv2.line(histogram_img, (i, 0), (i, histogram[i]), (0, 255, 0), 1)

    cv2.imshow("Histogram", histogram_img)
    cv2.waitKey(0)
