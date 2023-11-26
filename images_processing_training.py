import cv2
# import numpy as np
# foto2 = cv2.imread("./goruntu/kanal.png")#klasörden görüntü okuma
# cv2.imshow("Kırmızı",foto2)
foto = cv2.imread("baboon.bmp")
cv2.imshow("el",foto)
# cv2.waitKey()
B = foto[:,:,0]
G = foto[:,:,1]
R = foto[:,:,2]
# cv2.imshow("Mavi",B)
# cv2.imshow("Yesil",G)
# cv2.imshow("Kirmizi",R)
# cv2.waitKey()
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
imgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B
plt.imshow(imgGray, cmap='gray')
plt.show()




import cv2
import numpy as np
# foto2 = cv2.imread("./goruntu/kanal.png")#klasörden görüntü okuma
# cv2.imshow("Kırmızı",foto2)
foto = cv2.imread("hand.jpg")
# cv2.imshow("el",foto)
B = foto[:,:,0]
G = foto[:,:,1]
R = foto[:,:,2]
# # cv2.imshow("el1",B)
# # cv2.imshow("el2",G)
cv2.imshow("el3",R)
cv2.waitKey()
print(foto.shape) # yükseklik, genişlik ve kanal sayısı
print(R.shape)
x=4 #sütun no
y = 3 #satır no
kanal = 0
yogunluk_b= foto[y, x, kanal]
print("yoğunluk:",yogunluk_b)
yogunluk_r = R[y,x]
print("yoğunluk_gray:",yogunluk_r)
print("merhaba")
maksimum_yogunluk =np.max(B)
minimum_yogunluk =np.min(B)
print("Maksimum yoğunluk: ",maksimum_yogunluk)
print("Minimum yoğunluk: ",minimum_yogunluk)
print(foto[y,x])#tam koordinatın R G B değerleri döner
# görüntünün belirli piksllerini elde etme ve ekranda gösterme
parca2 = R[20:180, 40:280]
print(parca2)
cv2.imshow("parca",parca2)
cv2.waitKey()



3.1 py
import cv2
import numpy as np
from matplotlib import pyplot as plt
# renkli = cv2.imread("hand.jpg")
gri = cv2.imread("hand.jpg",0)
cv2.imshow("hand_original",gri)
cv2.waitKey()
# hist_color = cv2.calcHist([renkli], [2],None,[256],[0,256])#renkli görüntü histogram
hesaplama(2.parametre kanallar)
hist_gray = cv2.calcHist([gri], [0],None,[256],[0,256])#gri görüntü histogram hesaplama
# plt.figure(1)
# plt.plot(hist_color)
# plt.show()
plt.figure(2)
plt.plot(hist_gray)
plt.show()
# B = renkli[:,:,0]
# hist_B = cv2.calcHist([B], [0],None,[256],[0,256])#gri görüntü histogram hesaplama
# print(np.sum(hist_B))
# plt.plot(hist_B)
# plt.show()
# #alternatif gösterim
# plt.figure(3)
# plt.hist(hist_gray.ravel(),256,[0,256])
# plt.show()

3.2py
import cv2
import numpy as np
from matplotlib import pyplot as plt
foto1 = cv2.imread("hand.jpg",0)
import matplotlib.image as mpimg
# Michalson’in denklemi ile Contrast iyileştirme
CM = (np.max(foto1) - np.min(foto1))/(np.max(foto1) + np.min(foto1))
yeni = CM*foto1
plt.imshow(yeni, cmap='gray')
plt.show()
cv2.imshow("hand_original",foto1)
cv2.waitKey()
# İyileştirilen görüntü ile orijinalin histogram karşılaştırması
hist_gray = cv2.calcHist([foto1], [0],None,[256],[0,256])#gri görüntü histogram hesaplama
hist_gray_yeni = cv2.calcHist([yeni], [0],None,[256],[0,256])#gri görüntü histogram hesaplama
plt.figure(1)
plt.plot(hist_gray)
plt.figure(2)
plt.plot(hist_gray_yeni)
plt.show()
#np kütüphanesi ile histogram oluşturma (sonuçlar opencv ile aynı!!!)
# a=np.histogram(foto1,256,[0,256])#numpy fonksiyonu ile histogramı hesaplama (Hangi pikselden
kaç tane var görünür (Debug da)
# print(a[0])#histogram değerleri
# print(np.sum(a[0]))#histogram değerlerinin toplamı (kümülatif ile örtüşmeli)
# plt.figure(1)
# plt.plot(a[0])
# plt.show()
# # Cumulative histogram
# plt.figure(3)
# plt.hist(foto1.ravel(), bins = 256, cumulative = True)
# plt.xlabel('Intensity Value')
# plt.ylabel('Count')
# plt.show()

3.3py
import cv2
import matplotlib.pyplot as plt
# RGB bantlarının histogramlarını tek plot ta çizdirme
image = cv2.imread('hand.jpg')
for i, col in enumerate(['b', 'g', 'r']):
 hist = cv2.calcHist([image], [i], None, [256], [0, 256])#ikinci parametre renk kanalı için
 plt.plot(hist, color=col)
 plt.xlim([0, 256])
plt.show()




# görüntü inverting, positive/negative transforms
import cv2
import numpy as np
from matplotlib import pyplot as plt
image = cv2.imread('hand.jpg',0)
inverted = np.invert(image)
cv2.imshow("original",image)
cv2.imshow("inverted",inverted)
# cv2.waitKey()
negimage = 255 - image
cv2.imshow("negimage",negimage)
cv2.waitKey()
# [h,w] = image.shape
# new_image = np.zeros([h,w], dtype=np.uint8)
# for i in range(h):
# for j in range(w):
# new_image[i,j] = 255 - image[i,j]
#
# # print(image[0,0])
# cv2.imshow("Manuel_inverted",new_image)
# cv2.waitKey()


# eşikleme
import cv2
import numpy as np
from matplotlib import pyplot as plt
image1 = cv2.imread('el.png')
img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
ret, thresh1 = cv2.threshold(img, 140, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img, 140, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img, 140, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img, 140, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img, 140, 255, cv2.THRESH_TOZERO_INV)
cv2.imshow('Binary Threshold', thresh1)
cv2.imshow('Binary Threshold Inverted', thresh2)
cv2.imshow('Truncated Threshold', thresh3)
cv2.imshow('Set to 0', thresh4)
cv2.imshow('Set to 0 Inverted', thresh5)
cv2.waitKey()
#Belirli aralıklar arası manuel eşikleme
# myResult = cv2.inRange(img, 100, 130)
# cv2.imshow("esikleme",myResult)
# cv2.waitKey()

# Log dönüşümü
import cv2
import numpy as np
from matplotlib import pyplot as plt
image1 = cv2.imread('lion.jpg')
cv2.imshow("Original", image1)
# print(image1.dtype)
c = 255 / np.log(1 + np.max(image1))
log_image = c*np.log(1+image1)
log_image = np.array(log_image, dtype = np.uint8)#imshow ile gösterebilmek
için uint8 yapmalıyız
cv2.imshow("Logaritmik Dönüşüm", log_image)#değerler çok küçük, birşey
görünmez. 255 e bölmek gerekiyor
# log_image = 255/(c*np.log(1+image1))
# log_image = np.array(log_image, dtype = np.uint8)#imshow ile
gösterebilmek için uint8 yapmalıyız
# cv2.imshow("Logaritmik Dönüşüm_255", log_image)
cv2.waitKey()


import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('hand.jpg',0)
# img = cv2.medianBlur(img,5)
ret,th1=cv2.threshold(img,80,255,cv2.THRESH_BINARY)
th2=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BIN
ARY,11,2)
th3=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
cv2.THRESH_BINARY,11,2)
#giriş görüntüsü, max değer, adaptif metot, eşikleme tipi,blok boyutu
titles = ['Original Image', 'Global Thresholding (v = 90)',
'Adaptif ortalama esikleme', 'Adaptif Gaussian esikleme']
images = [img, th1, th2, th3]
for i in range(4):
 plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
 plt.title(titles[i])
 plt.xticks([]),plt.yticks([])
plt.show()



4.5py
#Gamma dönüşümü
import cv2
import matplotlib.pyplot as plt
import numpy as np
img = cv2.imread('lion.jpg')
cv2.imshow("orijinal", img)

# 4 gamma değerini uygulama.
for gamma in [0.1, 0.5, 1.16, 2.2]:

# Gamma uygulama
gamma_corrected = np.array(255 * (img/255) ** gamma, dtype='uint8')

# Düzeltilen görüntüleri kaydetme
# cv2.imwrite('gamma_transformed' + str(gamma) + '.jpg',
gamma_corrected)
#Ekrana gösterme
cv2.imshow("Gamma Dönüşüm", gamma_corrected)
cv2.waitKey()

#Gamma fonksiyonunu çizdirme
gama = 1/5
a = np.linspace(0, 1, 50)
print("a = ", a)
g = a ** gama
plt.plot(a, g)
plt.show()
4.6py
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
img = cv2.imread('lion.jpg')

# GÖRÜNTÜYÜ GRİ SEVİYEYE DÖNÜŞTÜR
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gamma değerini otomatik hesaplama= log(mid*255)/log(mean)
mid = 0.5
mean = np.mean(gray)
gamma = math.log(mid*255)/math.log(mean)

# GAMMA DÜZELTMESİ
img_gamma1 = np.power(img, gamma).clip(0,255).astype(np.uint8)
cv2.imshow('orijinal', img)
cv2.imshow('sonuc', img_gamma1)
cv2.waitKey(0)
# hist1 = cv2.calcHist([img], [0], None, [256], [0, 256])
# hist2 = cv2.calcHist([img_gamma1], [0], None, [256], [0, 256])
# plt.figure(1)
# plt.plot(hist1)
# plt.figure(2)
# plt.plot(hist2)
# plt.show()
4.7py

#Gamma dönüşümü
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
img = cv2.imread('lion.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hue, sat, val = cv2.split(hsv)
# cv2.imshow("val bandi", hsv)
mid = 0.5
mean = np.mean(val)
gamma = math.log(mid*255)/math.log(mean)
print(gamma)

# gamma düzeltmesi value bandında
val_gamma = np.power(val, gamma).clip(0,255).astype(np.uint8)

# iyileştirilen value bandı ile görüntünün diğer bantlarını birleştirme
hsv_gamma = cv2.merge([hue, sat, val_gamma])
img_gamma2 = cv2.cvtColor(hsv_gamma, cv2.COLOR_HSV2BGR)
cv2.imshow('orijinal', img)
cv2.imshow('SONUC', img_gamma2)
cv2.waitKey(0)
4.8py

#histogram eşitleme
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

# görüntüyü yeniden boyutlandırma yapacaksak
# image2 = cv2.resize(image, (500, 600))
img = cv2.imread('lion.jpg',0)
equ = cv2.equalizeHist(img)

# iki görüntüyü tek pencerede gösterme
res = np.hstack((img, equ))
# show image input vs output
cv2.imshow("image", res)
cv2.waitKey(0)
cv2.destroyAllWindows()
