import cv2
import numpy as np


def select_points(image, window_name="Sélection des points", num_points=8):
    """
    Permet à l'utilisateur de sélectionner num_points points sur l'image en cliquant.
    Les points sélectionnés sont affichés sous forme de petits cercles.
    """
    points = []

    # Fonction callback pour gérer les clics
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow(window_name, image)

    cv2.imshow(window_name, image)
    cv2.setMouseCallback(window_name, click_event)
    print("Veuillez sélectionner {} points dans la fenêtre '{}'.".format(num_points, window_name))

    # Boucle jusqu'à ce que le nombre de points soit atteint
    while True:
        cv2.imshow(window_name, image)
        key = cv2.waitKey(1) & 0xFF
        if len(points) >= num_points:
            break

    cv2.destroyWindow(window_name)
    return np.array(points, dtype=np.float32)


# Chargement d'un couple d'images (par exemple, la première paire)
img_gauche = cv2.imread('gauche_1.jpg')
img_droite = cv2.imread('droite_1.jpg')

# Copier les images pour ne pas altérer l'original lors de l'affichage des points
pts_gauche = select_points(img_gauche.copy(), window_name="Image Gauche", num_points=8)
pts_droite = select_points(img_droite.copy(), window_name="Image Droite", num_points=8)

# Calcul de la matrice fondamentale avec l'algorithme des 8 points
F, mask = cv2.findFundamentalMat(pts_gauche, pts_droite, cv2.FM_8POINT)
print("Matrice Fondamentale (F) :\n", F)

# Si vous avez déjà calibré chaque caméra individuellement, vous possédez les matrices intrinsèques K_gauche et K_droite.
# Vous pouvez alors calculer la matrice essentielle :
# E = K_droite.T * F * K_gauche
# Exemple (en supposant que K_gauche et K_droite sont définies) :
# E = K_droite.T @ F @ K_gauche
# print("Matrice Essentielle (E) :\n", E)

# Pour récupérer la rotation et la translation à partir de E, utilisez cv2.recoverPose :
# (La fonction récupère aussi une estimation de la pose qui est cohérente avec les points homologues)
# retval, R, T, mask_pose = cv2.recoverPose(E, pts_gauche, pts_droite, K_gauche)
# print("Rotation (R) :\n", R)
# print("Translation (T) :\n", T)
