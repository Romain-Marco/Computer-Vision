import cv2
import os

# Crée les dossiers nécessaires
os.makedirs('images_calibration/cam_gauche', exist_ok=True)
os.makedirs('images_calibration/cam_droite', exist_ok=True)

cap_gauche = cv2.VideoCapture(1)  # caméra gauche (change si besoin)
cap_droite = cv2.VideoCapture(0)  # caméra droite (change si besoin)

if not cap_gauche.isOpened() or not cap_droite.isOpened():
    print("Erreur : Vérifie les caméras.")
    exit()

compteur = 0
nombre_images = 10

print("Appuie sur 's' pour sauvegarder une paire d'images.")
print("Appuie sur 'q' pour quitter.")

while compteur < nombre_images:
    ret_g, frame_g = cap_gauche.read()
    ret_d, frame_d = cap_droite.read()

    if not ret_g or not ret_d:
        print("Erreur : Impossible de lire les flux vidéo.")
        break

    # Affichage simultané
    cv2.imshow('Camera Gauche', frame_g)
    cv2.imshow('Camera Droite', frame_d)

    key = cv2.waitKey(1)

    if key == ord('s'):
        cv2.imwrite(f'images_calibration/cam_gauche/gauche_{compteur+1}.jpg', frame_g)
        cv2.imwrite(f'images_calibration/cam_droite/droite_{compteur+1}.jpg', frame_d)
        print(f"✅ Paire sauvegardée n°{compteur+1}")
        compteur += 1

    if key == ord('q'):
        break

cap_gauche.release()
cap_droite.release()
cv2.destroyAllWindows()