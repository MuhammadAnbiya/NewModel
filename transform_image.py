import cv2
import os


def transform_image(image_path, output_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (48, 48)) #taille
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY) #ton grisâtre
    
    
    cv2.imwrite(output_path, img_gray) #enregistrement
    print(f"Image transformée et enregistrée sous {output_path}")


input_image_path = 'C:/Users/syrin/OneDrive/Bureau/face_expression_recognition/mon_image.jpeg' #A ADAPTER
output_image_path = 'C:/Users/syrin/OneDrive/Bureau/face_expression_recognition/mon_image_transformee.jpg'#A ADAPTER

transform_image(input_image_path, output_image_path)
