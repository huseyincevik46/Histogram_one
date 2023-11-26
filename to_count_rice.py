import cv2
import numpy as np

# Resmi oku
image = cv2.imread(r"C:\Users\cevik\Desktop/images_pirinc_1.jpg")

# Gri tonlamaya dönüştür
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Eşikleme uygula
_, thresholded = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

# Morfolojik işlemler uygula (istenmeyen arka planları temizle)
kernel = np.ones((5, 5), np.uint8)
morphology_result = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

# Etiketleme ve sayma
_, labels, stats, centroids = cv2.connectedComponentsWithStats(morphology_result, connectivity=8)

# Toplam pirinç sayısını al
num_rice = len(stats) - 1  # İlk etiket arka plan olduğu için çıkarılır

# Sonuçları ekrana yazdır
print("Pirinç sayısı:", num_rice)

# Görüntüleri göster
cv2.imshow('Original Image', image)
cv2.imshow('Thresholded Image', thresholded)
cv2.imshow('Morphology Result', morphology_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
