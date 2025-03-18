import cv2
import numpy as np
import os
import glob

# Dossier de sauvegarde
if not os.path.exists('images_calibration'):
    os.makedirs('images_calibration')

# Définition du damier
nb_lignes = 7
nb_colonnes = 9
taille_case = 15  # en mm

# Préparation des points réels (coordonnées 3D)
objp = np.zeros((nb_lignes * nb_colonnes, 3), np.float32)
objp[:, :2] = np.mgrid[0:nb_colonnes, 0:nb_lignes].T.reshape(-1, 2) * taille_case

# Connexion à la caméra
cap = cv2.VideoCapture(0)  # Change à 0 si nécessaire

if not cap.isOpened():
    print("Erreur : la caméra ne peut pas être ouverte.")
    exit()

compteur_images = 0
nombre_images_voulues = 10

print("Appuie sur 's' pour sauvegarder l'image lorsque les coins sont détectés.")
print("Appuie sur 'q' pour quitter.")

while compteur_images < nombre_images_voulues:
    ret, frame = cap.read()
    if not ret:
        print("Erreur : impossible de lire le flux vidéo.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (nb_colonnes, nb_lignes), None)

    if ret:
        cv2.drawChessboardCorners(frame, (nb_colonnes, nb_lignes), corners, ret)
        cv2.putText(frame, "Damier detecte, pret a enregistrer !", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    else:
        cv2.putText(frame, "Damier non detecte", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    cv2.imshow('Acquisition images calibration', frame)
    key = cv2.waitKey(1)

    if key == ord('s') and ret:
        image_path = f'images_calibration/calibration_{compteur_images + 1}.jpg'
        cv2.imwrite(image_path, frame)
        compteur_images += 1
        print(f"✅ Image sauvegardée : {image_path} ({compteur_images}/{nombre_images_voulues})")

    elif key == ord('q'):
        print("Interruption par l'utilisateur.")
        break

cap.release()
cv2.destroyAllWindows()

# ===== Calibration Automatique =====

points_objet, points_image = [], []

images = glob.glob('images_calibration/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (nb_colonnes, nb_lignes), None)

    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_precis = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

        points_objet.append(objp)
        points_image.append(corners_precis)

        cv2.drawChessboardCorners(img, (nb_colonnes, nb_lignes), corners_precis, ret)
        cv2.imshow('Verification detection coins', img)
        cv2.waitKey(200)

cv2.destroyAllWindows()

ret, matrice_intrinseque, distorsion, rvecs, tvecs = cv2.calibrateCamera(
    points_objet, points_image, gray.shape[::-1], None, None)

# Résultats détaillés
print("\n--- RESULTATS DE CALIBRATION ---")
print("\nMatrice intrinsèque:\n", matrice_intrinseque)
print("\nCoefficients de distorsion:\n", distorsion)
print("\nErreur de reprojection moyenne :", ret, "pixels")

# Sauvegarde
fs = cv2.FileStorage("calibration_camera.yml", cv2.FILE_STORAGE_WRITE)
fs.write("matrice_intrinseque", matrice_intrinseque)
fs.write("distorsion", distorsion)
fs.release()

# Correction d'image exemple
img_test = cv2.imread(images[0])
h, w = img_test.shape[:2]
nouvelle_matrice, roi = cv2.getOptimalNewCameraMatrix(matrice_intrinseque, distorsion, (w,h), 1, (w,h))

img_corrigee = cv2.undistort(img_test, matrice_intrinseque, distorsion, None, nouvelle_matrice)

cv2.imshow("Image originale", img_test)
cv2.imshow("Image corrigee", img_corrigee)
cv2.waitKey(0)
cv2.destroyAllWindows()