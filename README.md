# Analyse d'Image

## Partie 1 :

```
python3 -m venv env_analyse_image
source env_analyse_image/bin/activate
pip install -r requirement.txt
jupyter-notebook
```

## Partie 2 :
### Models dynamique
```
python3 src/backgroundremovalatd.py
```
### Technique de soustraction d'image
```
python3 src/backRemovalImageSub.py
```

## Partie 3 :
```
python3 src/main.py
```
*Commandes optionnelle :*
```
--body <body_filename>
--bg <bg_filename>
```

Ou de façon plus modulaire, chaque partie toute seul (en respectant l'ordre de dépendance) :
```
python3 src/bg_remover.py
python3 src/image_editor.py
python3 src/seeds_sower.py
python3 src/seeds_reducer.py
python3 src/seeds_expander.py
```