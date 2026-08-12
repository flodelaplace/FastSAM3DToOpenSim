# Génération d'avatars — arguments à exposer dans l'app

Destinataire : Sacha (intégration app).
Contexte : quand l'utilisateur demande un avatar anonymisé à partir d'une vidéo,
l'app doit choisir quelques options. Ce document dit **quel bouton produit quel
argument**, et surtout **ce qu'il ne faut PAS exposer**.

Point d'entrée : `generate_avatars.py` (pipeline avatar seul, pas d'IK, pas
d'analytics — c'est le plus rapide). Le pipeline complet
`demo_video_opensim.py` prend les mêmes arguments de cadrage.

---

## 1. Ce qui est déjà automatique — ne rien demander à l'utilisateur

Ces traitements sont **actifs par défaut**. Ne pas créer de bouton pour eux.

| Traitement | Détail |
|---|---|
| Mise au sol + redressement caméra | `--floor_moge` est activé par défaut (RANSAC multi-frames sur l'estimation MoGe). Corrige le pitch/roll de la caméra et pose le sujet au sol. |
| Anti-glisse des pieds | Actif par défaut. Fige le XZ des pieds pendant les phases d'appui. |
| Markerset | 95 marqueurs (87 SOLE + 8 doigts). Rien à choisir. |
| Catalogue d'avatars | Les 10 avatars de `assets/avatars/` sont générés d'office, un GLB chacun. |

---

## 2. Les boutons à exposer

### Taille du sujet — **obligatoire**

```
--person_height 1.75
```

En **mètres**. C'est le seul paramètre vraiment indispensable : toute la mise à
l'échelle 3D en dépend. À demander systématiquement (champ numérique, défaut
1.75 si l'utilisateur ne sait pas).

### « Le sujet reste sur place ? »

```
--stationary
```

À cocher pour squat, CMJ, soulevé de terre, gainage, exercices sur tapis —
tout ce qui n'est pas un déplacement. Verrouille la translation XZ globale :
le sujet reste centré, pieds plantés.

**Ne pas cocher** pour la marche, la course, les fentes marchées, les sauts avec
déplacement : le sujet resterait collé sur place.

### « Position de départ »

Trois choix mutuellement exclusifs :

| Bouton | Argument | Quand |
|---|---|---|
| Debout (défaut) | *(aucun)* | Le sujet est debout, sol visible |
| Assis | `--floor_seated` | Sit-to-stand, exercices sur chaise |
| Au sol / allongé | `--floor` | Gainage, pont fessier, étirements au sol |

⚠️ `--floor` active aussi une correction « corps vertical » qui force l'axe
médio-pied→cou à la verticale. Elle **suppose le sujet debout** : à ne pas
utiliser pour rameur, sit-to-stand, Lasègue ou suspension.

### « Exercice avec appuis (squat, fente…) »

```
--contact_anchor
```

Recommandé pour les avatars d'exercice. Ré-ancre le pied en appui soutenu — sans
lui, le bassin peut rester figé pendant qu'un squat descend. Préserve les phases
de vol (course, saut).

---

## 3. Options de secours — quand le résultat est penché

L'estimation du sol échoue dans deux cas connus :

- **le sujet est trop près du bas du cadre** (pieds coupés ou au ras du bord) ;
- **aucun sol n'est visible** dans la vidéo.

Le symptôme est visible immédiatement : l'avatar penche en avant ou en arrière
sur toute la séquence. Trois recours, du plus simple au plus précis :

| Argument | Effet |
|---|---|
| `--no_floor_moge` | Coupe l'estimation MoGe. À utiliser quand il n'y a **pas de sol** dans la vidéo — mieux vaut aucune correction qu'une correction fausse. |
| `--lean_ref_frame <n>` | Mesure l'inclinaison sur la frame `n` (moyennée sur ±5) et la corrige. L'app peut proposer « choisis une image où la personne est droite ». |
| `--lean_angle <deg>` | Correction manuelle en degrés. Positif = redresse une inclinaison avant. **Écrase** MoGe et la correction automatique. |

Proposer ces options en « réglage avancé », pas dans le flux principal.

---

## 4. À ne PAS exposer

| Argument | Pourquoi |
|---|---|
| `--lock-vertical` | Ne redresse **pas** le sujet malgré son nom : il verrouille la translation verticale du bassin, et sert uniquement au bikefit home-trainer. Absent de `generate_avatars.py`. |
| `--no_anti_foot_skate` | Sert au débogage. L'anti-glisse doit rester actif. |
| `--markerset`, `--detector_model`, `--bbox_thr`, `--nms_thr`, `--inference_*`, `--fallback_*` | Réglages moteur. Valeurs de production figées, cf. exemple ci-dessous. |

---

## 5. Exemple complet

Fente statique, sujet de 1,75 m, debout, sur place :

```bash
python generate_avatars.py \
  --video_path /app/videos/ma_video.mp4 \
  --output_dir /outputs/mon_resultat \
  --person_height 1.75 \
  --contact_anchor \
  --inference_type body --markerset flodelaplace \
  --detector_model checkpoints/yolo/yolo11m-pose.engine \
  --bbox_thr 0.2 --nms_thr 0.9 --detect_then_infer --inference_batch_cap 4 \
  --fallback_lower_bbox 0.05 --fallback_nms 0.9 --fallback_iou_thresh 0.5
```

Tout ce qui suit `--inference_type` est de la configuration moteur : à recopier
tel quel.

### Sortie

```
markers_<nom>.trc                        TRC 95 marqueurs (mm, Y-up)
markers_<nom>_avatar_<template>.glb      1 GLB animé par avatar du catalogue
inference_meta.json                      métadonnées vidéo
```

---

## 6. Catalogue d'avatars — ce que le kiné voit et choisit

Un run produit **un fichier GLB par avatar**, nommé :

```
markers_<nom_video>_avatar_<avatar>.glb
```

Le suffixe après `_avatar_` est l'identité à présenter dans l'interface. Tous
jouent **exactement le même mouvement** — celui du patient filmé. Seule
l'apparence change. Le choix est donc purement une question de **à qui
l'exercice est destiné**.

| Fichier `..._avatar_X.glb` | À montrer à | Taille | Apparence |
|---|---|---|---|
| `male_child` | garçon | 1,24 m | enfant, t-shirt + bermuda |
| `female_child` | fille | 1,22 m | enfant, t-shirt + jean |
| `male_young` | homme adulte | 1,74 m | ~25 ans, t-shirt + bermuda |
| `female_young` | femme adulte | 1,59 m | ~30 ans, t-shirt + short |
| `male_old` | homme senior | 1,69 m | ~70 ans, cheveux gris, chemise + jean |
| `female_old` | femme senior | 1,57 m | ~70 ans, cheveux blancs, pull + pantalon |
| `male_young_large` | homme adulte en surpoids | 1,74 m | idem `male_young`, corpulence élevée |
| `female_young_large` | femme adulte en surpoids | 1,59 m | idem `female_young`, corpulence élevée |
| `male_old_large` | homme senior en surpoids | 1,69 m | idem `male_old`, corpulence élevée |
| `female_old_large` | femme senior en surpoids | 1,57 m | idem `female_old`, corpulence élevée |

**Règle de sélection à implémenter** : proposer l'avatar dont l'âge, le sexe et
la corpulence se rapprochent le plus du patient. Un patient de 70 ans se
reconnaît mieux dans `male_old` que dans `male_young`, et l'adhésion à
l'exercice en dépend.

⚠️ **Ne pas proposer un avatar d'enfant pour une vidéo d'adulte** (ni
l'inverse) : l'avatar est étiré vers la morphologie du sujet filmé, mais cet
étirement est **plafonné**. Un avatar enfant (tronc ~33 cm) sur un adulte
(tronc ~64 cm) sature la limite et sort avec des proportions fausses. Filtrer
le catalogue sur la classe d'âge du sujet filmé.

### Poids : ne pas tout générer

Chaque GLB pèse **~27 Mo** — le catalogue complet fait donc **~270 Mo par
exercice**. Le temps n'est pas le problème (1,4 s par avatar), le stockage si.

Deux façons de n'en produire qu'un :

```bash
--avatars male_old                 # un ou plusieurs noms du catalogue
--avatars male_old,female_old      # séparés par des virgules
```

Et surtout : **le TRC ne pèse que ~380 Ko**. On peut donc archiver le TRC seul
et rejouer le retargeting à la demande en ~1,4 s quand le kiné change d'avatar,
sans refaire l'inférence (qui, elle, prend des minutes de GPU). C'est
l'architecture recommandée pour l'app.

Les sources sont des specs JSON versionnées (`assets/avatars/specs/`) : une
variante se dérive en changeant une ligne, sans repasser par MakeHuman.

---

## 7. Déclenchement AWS par nom de fichier

Sur AWS, le lambda lit les options dans le **nom du fichier** déposé sur S3,
sous la forme `<nom>__h<cm>[_tokens].mp4` :

| Token | Argument |
|---|---|
| `h175` | `--person_height 1.75` |
| `st` | `--stationary` |
| `floor` | `--floor` |
| `com` | `--compute_com` |

Exemple : `ex03_fente__h175_floor.mp4`.

⚠️ Tous les arguments n'ont **pas** de token (`--contact_anchor`,
`--lean_ref_frame`, `--no_floor_moge` n'en ont pas aujourd'hui). Si l'app doit
les piloter via S3, il faut d'abord les ajouter à `aws/lambda_trigger.py`.
