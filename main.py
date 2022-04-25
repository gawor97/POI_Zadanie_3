from pandas import DataFrame
from itertools import product
import numpy as np
from PIL import Image
from skimage.feature import greycomatrix, greycoprops
from glob import glob
from os.path import sep, join, splitext

cechy_tekstur = ('dissimilarity', 'contrast', 'correlation', 'energy', 'homogeneity', 'ASM')

distances = (1, 3, 5)

angles = (0, np.pi/4, np.pi/2, 3*np.pi/4)

def get_full_names():
    dist_str = ('1', '2', '5')
    angles_str = '0deg, 45deg, 90deg, 135deg'.split(',')
    return ['_'.join(f) for f in product(cechy_tekstur, dist_str, angles_str)]

#Funckcja obliczająca macierz GLCM
def get_glcm_feature_array(patch):
    patch_64 = (patch / np.max(patch) * 63).astype('uint8')
    glcm = greycomatrix(patch_64, distances, angles, 64, True, True)
    feature_vector = []
    for feature in cechy_tekstur:
        feature_vector.extend(list(greycoprops(glcm, feature).flatten()))
    return feature_vector

#Katalogi z zdjeciami
texture_folder = "Textures"
samples_folder = "TextureSamples"
paths = glob(texture_folder + "\\*\\*.jpg")

fil2 = [p.split(sep) for p in paths]
_, categories, files = zip(*fil2)

size = 128, 128 # rozmiar probek zdjec

features = []
for category, infile in zip(categories, files):
    img = Image.open(join(texture_folder, category, infile))
    xr = np.random.randint(0, img.width-size[0], 10)            #
    yr = np.random.randint(0, img.height-size[1], 10)           # losowanie 10 położeń próbek ze zdjęcia
    base_name, _ = splitext(infile)                             # wydzielenie nazwy zdjęcia
    for i, (x, y) in enumerate(zip(xr, yr)):
        img_sample = img.crop((x, y, x+size[0], y+size[1]))
        img_sample.save(join(samples_folder, category, f'{base_name:s}_{i:02d}.jpg'))   # zapis do pliku
        img_grey = img.convert('L')                             # konwersja do skali szarości
        feature_vector = get_glcm_feature_array(np.array(img_grey))     # generowanie cech tekstury
        feature_vector.append(category)                         # dołączenie nazwy kategorii do wektora
        features.append(feature_vector)                         # dołączenie wektora do wszystkich wektorów

full_feature_names = get_full_names()
full_feature_names.append('Category')

df = DataFrame(data=features, columns=full_feature_names)
df.to_csv('textures_data.csv', sep=',', index=False)        # zapis do pliku

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

features = pd.read_csv('textures_data.csv', sep=',')    # otwarcie pliku z danymi

data = np.array(features)       # zapis danych do tablicy
x = (data[:, :-1]).astype('float64')    # do X zapisuje wszystkie kolumny BEZ OSTATNIEJ (-1), astype - zamiana typu
y = data[:, -1]

x_transform = PCA(n_components=3)
xt = x_transform.fit_transform(x)

red = y == 'drewno'
blue = y == 'scany'
green = y == 'plytki'

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xt[red, 0], xt[red, 1], xt[red, 2], c="r")
ax.scatter(xt[blue, 0], xt[blue, 1], xt[blue, 2], c="b")
ax.scatter(xt[green, 0], xt[green, 1], xt[green, 2], c="g")

classifier = svm.SVC(gamma='auto')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)


classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(acc)

cm = confusion_matrix(y_test, y_pred, normalize='true')

print(cm)

disp = plot_confusion_matrix(classifier, x_test, y_test, cmap=plt.cm.Blues)
plt.show()
