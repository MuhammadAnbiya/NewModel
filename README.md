# Face Expression Recognition

## Introduction

L'expression faciale est l'un des principaux vecteurs de la communication humaine. Ce projet repose sur l'utilisation d'un réseau de neurones convolutionnels (CNN) pour analyser des images de visages et en extraire l'émotion correspondante. L'objectif est de fournir une classification en temps réel des expressions faciales en s'appuyant sur un modèle entraîné avec un ensemble de données annotées.

## Fonctionnalités

- Détection des visages à l'aide d'OpenCV.
- Classification des expressions faciales en différentes catégories (joie, tristesse, colère, surprise, etc.).
- Prédiction en temps réel via une webcam ou sur des images statiques.
- Entraînement et évaluation d'un modèle de deep learning avec TensorFlow et Keras.

## Installation

### Prérequis

Avant d'exécuter ce projet, assurez-vous d'avoir installé les dépendances nécessaires. Si vous utilisez un environnement virtuel, vous pouvez le recréer avec la commande suivante :

```bash
python -m venv venv
source venv/bin/activate  # Sur macOS/Linux
venv\Scripts\activate  # Sur Windows
pip install -r requirements.txt
```

### Cloner le dépôt GitHub

```bash
git clone https://github.com/sy895/facial_emotion_recognition.git
cd facial_emotion_recognition
```

## Utilisation

### Lancer la détection en temps réel

```bash
python app.py
```

Ce script capture la vidéo depuis la webcam, détecte les visages et affiche l’émotion prédite en temps réel.

### Tester une image statique

Si vous souhaitez effectuer une prédiction sur une image unique, exécutez :

```bash
python emotion_recognition.py --image chemin/vers/image.jpg
```

### Entraîner le modèle

Si vous souhaitez réentraîner le modèle avec un nouveau jeu de données, exécutez :

```bash
python train_model.py
```

Assurez-vous que les données sont bien organisées dans des répertoires `train/` et `test/`, avec chaque classe représentée par un sous-dossier contenant les images correspondantes.

## Structure du projet

```
face_expression_recognition/
│── app.py                  # Script principal pour la détection en temps réel
│── train_model.py           # Entraînement du modèle
│── emotion_recognition.py   # Prédiction sur une image statique
│── requirements.txt         # Liste des dépendances nécessaires
│── model/                   # Modèle pré-entraîné
│── dataset/                 # Dossier contenant les images d'entraînement et de test
│── utils/                   # Fonctions auxiliaires pour le prétraitement des données
└── README.md                # Documentation du projet
```

## Dépendances

Les principales bibliothèques utilisées dans ce projet sont :

- TensorFlow
- OpenCV
- NumPy
- Matplotlib
- Pillow

Pour installer toutes les dépendances, utilisez :

```bash
pip install -r requirements.txt
```

## Améliorations possibles

- Optimisation du modèle avec d'autres architectures plus performantes.
- Ajout d’un mécanisme d’augmentation des données pour améliorer la robustesse du modèle.
- Prise en charge des micro-expressions pour affiner l’analyse des émotions.
- Intégration d’un mode interactif avec affichage des statistiques sur les prédictions.

## Licence

Ce projet est sous licence MIT. Vous êtes libre de l'utiliser et de le modifier selon vos besoins.

