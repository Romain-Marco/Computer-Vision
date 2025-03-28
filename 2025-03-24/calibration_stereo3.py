import cv2
import numpy as np
import glob


def select_points(image, window_name="Sélection des points", num_points=8):
    """
    Permet à l'utilisateur de sélectionner num_points points sur l'image en cliquant.
    Les points sélectionnés sont affichés sous forme de petits cercles.
    """
    points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow(window_name, image)

    cv2.imshow(window_name, image)
    cv2.setMouseCallback(window_name, click_event)
    print("Veuillez sélectionner {} points dans la fenêtre '{}'.".format(num_points, window_name))

    while True:
        cv2.imshow(window_name, image)
        key = cv2.waitKey(1) & 0xFF
        if len(points) >= num_points:
            break
    cv2.destroyWindow(window_name)
    return np.array(points, dtype=np.float32)


# -------------------------------
# Calibration individuelle des caméras
# -------------------------------

# Paramètres du damier
nb_lignes = 7
nb_colonnes = 9
taille_case = 15  # en mm

# Préparation de l'objet 3D pour le damier
objp = np.zeros((nb_lignes * nb_colonnes, 3), np.float32)
objp[:, :2] = np.mgrid[0:nb_colonnes, 0:nb_lignes].T.reshape(-1, 2) * taille_case

# Listes pour stocker les points 3D et les points détectés sur chaque image
objpoints = []  # Points 3D réels
imgpoints_g = []  # Points détectés dans les images de la caméra gauche
imgpoints_d = []  # Points détectés dans les images de la caméra droite

images_g = sorted(glob.glob('images_calibration/cam_gauche/*.jpg'))
images_d = sorted(glob.glob('images_calibration/cam_droite/*.jpg'))

# Pour récupérer la taille d'image, on lit la première image de la caméra gauche
if len(images_g) == 0:
    raise ValueError("Aucune image trouvée dans 'images_calibration/cam_gauche/'")
img_temp = cv2.imread(images_g[0])
gray_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)
image_size = gray_temp.shape[::-1]  # largeur, hauteur

for fname_g, fname_d in zip(images_g, images_d):
    img_g = cv2.imread(fname_g)
    img_d = cv2.imread(fname_d)
    gray_g = cv2.cvtColor(img_g, cv2.COLOR_BGR2GRAY)
    gray_d = cv2.cvtColor(img_d, cv2.COLOR_BGR2GRAY)

    ret_g, corners_g = cv2.findChessboardCorners(gray_g, (nb_colonnes, nb_lignes), None)
    ret_d, corners_d = cv2.findChessboardCorners(gray_d, (nb_colonnes, nb_lignes), None)

    if ret_g and ret_d:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2_g = cv2.cornerSubPix(gray_g, corners_g, (11, 11), (-1, -1), criteria)
        corners2_d = cv2.cornerSubPix(gray_d, corners_d, (11, 11), (-1, -1), criteria)

        objpoints.append(objp)
        imgpoints_g.append(corners2_g)
        imgpoints_d.append(corners2_d)

# Calibration individuelle pour chaque caméra
ret_g, K_g, dist_g, rvecs_g, tvecs_g = cv2.calibrateCamera(objpoints, imgpoints_g, image_size, None, None)
ret_d, K_d, dist_d, rvecs_d, tvecs_d = cv2.calibrateCamera(objpoints, imgpoints_d, image_size, None, None)

print("\n--- Calibration Individuelle Gauche ---")
print("Matrice Intrinsèque Gauche :\n", K_g)
print("Distorsion Gauche :\n", dist_g)

print("\n--- Calibration Individuelle Droite ---")
print("Matrice Intrinsèque Droite :\n", K_d)
print("Distorsion Droite :\n", dist_d)

# -------------------------------
# Calibration stéréo manuelle via l'algorithme des 8 points
# -------------------------------

# Choix d'une paire d'images pour la calibration stéréo (ici la première image de chaque série)
img_g = cv2.imread('camerag.jpg')
img_d = cv2.imread('camerad.jpg')

# Sélection manuelle de 8 points homologues sur chaque image
pts_g = select_points(img_g.copy(), window_name="Image Gauche", num_points=8)
pts_d = select_points(img_d.copy(), window_name="Image Droite", num_points=8)

# Calcul de la matrice fondamentale avec l'algorithme des 8 points
F, mask = cv2.findFundamentalMat(pts_g, pts_d, cv2.FM_8POINT)
print("\n--- Matrice Fondamentale ---")
print(F)

# Calcul de la matrice essentielle à partir de F et des matrices intrinsèques
# La formule est : E = K_droite^T * F * K_gauche
E = K_d.T @ F @ K_g
print("\n--- Matrice Essentielle ---")
print(E)

# Récupération de la pose relative (rotation et translation) entre les caméras
retval, R, T, mask_pose = cv2.recoverPose(E, pts_g, pts_d, K_g)
print("\n--- Pose relative entre les caméras ---")
print("Rotation (R) :\n", R)
print("Translation (T) :\n", T)

# -------------------------------
# Sauvegarde des paramètres de calibration dans un fichier YAML
# -------------------------------

fs = cv2.FileStorage('calibration_stereo.yml', cv2.FILE_STORAGE_WRITE)
fs.write("K_gauche", K_g)
fs.write("dist_gauche", dist_g)
fs.write("K_droite", K_d)
fs.write("dist_droite", dist_d)
fs.write("F", F)
fs.write("E", E)
fs.write("R", R)
fs.write("T", T)
fs.release()

cv2.destroyAllWindows()
