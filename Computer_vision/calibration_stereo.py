import cv2
import numpy as np
import glob

# Paramètres du damier
nb_lignes = 7
nb_colonnes = 9
taille_case = 15  # en mm

objp = np.zeros((nb_lignes*nb_colonnes,3), np.float32)
objp[:,:2] = np.mgrid[0:nb_colonnes,0:nb_lignes].T.reshape(-1,2)*taille_case

# Préparation des listes pour les points
objpoints = []  # 3D réels
imgpoints_g = []  # image gauche
imgpoints_d = []  # image droite

images_g = sorted(glob.glob('images_calibration/cam_gauche/*.jpg'))
images_d = sorted(glob.glob('images_calibration/cam_droite/*.jpg'))

for fname_g, fname_d in zip(images_g, images_d):
    img_g = cv2.imread(fname_g)
    img_d = cv2.imread(fname_d)
    gray_g = cv2.cvtColor(img_g, cv2.COLOR_BGR2GRAY)
    gray_d = cv2.cvtColor(img_d, cv2.COLOR_BGR2GRAY)

    ret_g, corners_g = cv2.findChessboardCorners(gray_g, (nb_colonnes, nb_lignes), None)
    ret_d, corners_d = cv2.findChessboardCorners(gray_d, (nb_colonnes, nb_lignes), None)

    if ret_g and ret_d:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2_g = cv2.cornerSubPix(gray_g, corners_g, (11,11), (-1,-1), criteria)
        corners2_d = cv2.cornerSubPix(gray_d, corners_d, (11,11), (-1,-1), criteria)

        objpoints.append(objp)
        imgpoints_g.append(corners2_g)
        imgpoints_d.append(corners2_d)

# Calibration individuelle des caméras
ret_g, K_g, dist_g, _, _ = cv2.calibrateCamera(objpoints, imgpoints_g, gray_g.shape[::-1], None, None)
ret_d, K_d, dist_d, _, _ = cv2.calibrateCamera(objpoints, imgpoints_d, gray_d.shape[::-1], None, None)

# Calibration stéréo (extrinsèque)
flags = cv2.CALIB_FIX_INTRINSIC
ret_stereo, K_g, dist_g, K_d, dist_d, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_g, imgpoints_d, K_g, dist_g, K_d, dist_d,
    gray_g.shape[::-1], criteria=criteria, flags=flags)

# Affichage résultats
print("\n--- Calibration individuelle Gauche ---")
print("Matrice Intrinsèque Gauche :\n", K_g)
print("Distorsion Gauche :\n", dist_g)

print("\n--- Calibration individuelle Droite ---")
print("Matrice Intrinsèque Droite :\n", K_d)
print("Distorsion Droite :\n", dist_d)

print("\n--- Calibration Stéréo ---")
print("Rotation entre caméras (R) :\n", R)
print("Translation entre caméras (T) :\n", T)
print("Erreur de reprojection moyenne (stéréo):", ret_stereo)

# Sauvegarde des résultats
fs = cv2.FileStorage('calibration_stereo.yml', cv2.FILE_STORAGE_WRITE)
fs.write("K_gauche", K_g)
fs.write("dist_gauche", dist_g)
fs.write("K_droite", K_d)
fs.write("dist_droite", dist_d)
fs.write("R", R)
fs.write("T", T)
fs.release()
