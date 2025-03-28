import cv2
import os

# Nom du sous-dossier où enregistrer les images
output_folder = "captures"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Ouverture des deux caméras (indice 0 et 1)
cap_gauche = cv2.VideoCapture(1)
cap_droite = cv2.VideoCapture(2)

if not cap_gauche.isOpened() or not cap_droite.isOpened():
    print("Erreur lors de l'ouverture d'une des caméras.")
    exit()

capture_index = 1

while True:
    ret_g, frame_g = cap_gauche.read()
    ret_d, frame_d = cap_droite.read()

    if not ret_g or not ret_d:
        print("Erreur de lecture d'une des caméras.")
        break

    # Affichage des flux vidéo
    cv2.imshow("Camera Gauche", frame_g)
    cv2.imshow("Camera Droite", frame_d)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        # Construction des noms de fichiers
        filename_g = os.path.join(output_folder, f"gauche_{capture_index}.jpg")
        filename_d = os.path.join(output_folder, f"droite_{capture_index}.jpg")
        # Enregistrement des images
        cv2.imwrite(filename_g, frame_g)
        cv2.imwrite(filename_d, frame_d)
        print(f"Images enregistrées : {filename_g} et {filename_d}")
        capture_index += 1

    elif key == ord('q'):
        break

# Libération des ressources
cap_gauche.release()
cap_droite.release()
cv2.destroyAllWindows()
