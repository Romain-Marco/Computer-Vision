import cv2

# Ouverture des deux caméras
cap0 = cv2.VideoCapture(1)
cap1 = cv2.VideoCapture(2)

if not cap0.isOpened() or not cap1.isOpened():
    print("Erreur lors de l'ouverture d'une des caméras.")
    exit()

while True:
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()

    if not ret0 or not ret1:
        print("Erreur de lecture d'une image.")
        break

    # Affichage des images dans deux fenêtres
    cv2.imshow("Camera 0", frame0)
    cv2.imshow("Camera 1", frame1)

    key = cv2.waitKey(1) & 0xFF
    # Si on appuie sur 's', on sauvegarde les images
    if key == ord('s'):
        cv2.imwrite("camera0.jpg", frame0)
        cv2.imwrite("camera1.jpg", frame1)
        print("Images sauvegardées : camera0.jpg et camera1.jpg")
    # Appuyez sur 'q' pour quitter
    elif key == ord('q'):
        break

# Libération des ressources et fermeture des fenêtres
cap0.release()
cap1.release()
cv2.destroyAllWindows()
