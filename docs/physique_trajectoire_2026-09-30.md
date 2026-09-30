# Trajectoire physiquement cohérente : vols balistiques, raccords, bornes

Rédigé le 2026-09-30. Demande de Florian : « très important de gérer les contacts au sol,
l'anti-glissement, et quand il y a un mouvement et un saut, avoir un contrôle de la profondeur
plausible pour éviter que la personne recule alors qu'elle saute ; intégrer un peu de physique
du corps et de cohérence temporelle ».

Livrables :

- `sam_3d_body/export/physique_trajectoire.py` : prototype, fonctions pures numpy/scipy,
  **branché nulle part**.
- `tests/test_physique_trajectoire.py` : 15 tests sur signaux synthétiques
  (`python3 -m pytest tests/test_physique_trajectoire.py -q` : 15 passent, 0,4 s).
- `tools/eval_physique_trajectoire.py` : évaluation reproductible sur les sorties locales
  (`python3 tools/eval_physique_trajectoire.py --json sortie.json --dj <dossiers drop jump>`).
- Aucun fichier existant n'a été modifié. Le module se charge par son chemin (le paquet
  `sam_3d_body` tire torch, absent de l'hôte).

## 1. Le défaut, mesuré

La translation globale vient de `cam_t`, donc d'une **profondeur estimée image par image**.
Sur le CMJ `NUIT_jump` (30 i/s, vol de 0,59 s), le centre de masse **recule de 8,4 cm en
profondeur** pendant la montée, puis repart en avant : il s'écarte de 8,8 cm de la droite
décollage → réception, alors qu'aucune force horizontale n'agit sur un corps en l'air. Même
chose sur les deux drop jumps de production (4,9 et 6,5 cm en profondeur, jusqu'à 8,5 cm en latéral)
et sur le hop unipodal (9,2 cm en médiane, 17,3 cm au pire bond).

X est bien l'axe de profondeur : `CAMERA_TO_OPENSIM` envoie Z caméra sur X OpenSim (au
redressement de la verticale près, quelques degrés).

## 2. Ce qui existait déjà, et ce que l'étage ajoute

| étage | fichier | ce qu'il fait | limite pour le saut |
|---|---|---|---|
| lissage rigide 2,5 Hz | `coordinate_transform.lisser_trajectoire_rigide` | retire les sauts de `cam_t` | ne connaît ni appui ni vol ; coupé sur les modules « en place » (CMJ, drop jump) |
| anti-glissement | `coordinate_transform.anti_foot_skate_markers` | ancre le pied par appui (XZ) | ne dit rien du vol : l'ancre expire au décollage, la profondeur est libre en l'air |
| gardes balistiques | `synkro-analytics/.../jump.py::_detect_jumps_sole` | écartent les faux vols (durée > 0,9 s, montée non balistique) | côté analyse seulement, ne corrige pas la trajectoire |
| contacts v2 | `synkro-analytics/core/contacts_v2.py` | datation d'événements de marche/course | prototype, pas de notion de vol balistique |

Le nouvel étage est **complémentaire** : il passe **après** l'anti-glissement et ne touche,
par défaut, qu'au vol, aux raccords et aux bornes.

## 3. Méthode

### 3.1 Phases (`detecter_phases`)

- Hauteur du point de semelle `SOLE_*` le plus bas de chaque pied au-dessus du sol (sol
  global = 10e centile, ou enveloppe basse glissante pour une caméra portée).
- Deux voies vers le contact : **hystérésis** 3 cm à l'entrée / 5 cm à la sortie, **ou pied bas
  et immobile** (moins de 10 cm du sol, vitesse verticale < 0,25 m/s pendant 0,1 s). La seconde
  voie vient de `NUIT_hop` : à la réception du 2e bond, les semelles restent à 5-7 cm pendant
  0,3 s alors que le sujet s'enfonce, et la hauteur seule prolongeait le vol de 0,47 à 0,80 s.
- **Vol** = aucun pied en contact, 0,06-0,90 s. Au-delà de 0,90 s, « passage en hauteur » non
  corrigé (sujet debout sur la box d'un drop jump, sol mal posé).
- **Garde balistique** pour les vols ≥ 0,15 s : une parabole *libre* ajustée au centre de masse
  vertical doit impliquer une gravité entre g/2 et 2g. ⚠️ Le critère de `jump.py` (montée
  depuis le décollage entre ½ et 2 × gT²/8) suppose un vol symétrique ; sur un hop pour la
  distance la réception se fait 25 cm plus bas que le décollage et il écartait un vrai vol
  (6 cm de montée pour 16 « attendus »). À signaler à `jump.py` pour les hops.

### 3.2 Centre de masse (`centre_de_masse_approx`)

En vol, c'est le **centre de masse** qui suit une droite et une parabole, pas le bassin (bras
lancés, jambes groupées). Moyenne des centres segmentaires pondérée par les fractions de masse
de de Leva (1996), segments manquants retirés image par image, repli sur le bassin. Une
translation rigide ne change pas la position relative du centre de masse : l'approximation ne
se reporte que sur la forme imposée, jamais sur la posture.

### 3.3 Le décalage rigide (`trajectoire_physique`)

On cherche un vecteur s(t) par image, **le même pour tous les marqueurs**, par moindres carrés
creux, axe par axe :

| terme | rôle | réglage |
|---|---|---|
| rappel s → 0 | la correction reste petite, `cam_t` reste la référence | σ = 10 cm hors appui, **2 cm en appui** |
| lissage D²s | correction lisse (raccords C1) | σ = 5 m/s² |
| vol, X et Z : D²(c+s) = 0 | vitesse horizontale constante en l'air | σ = 0,05 m/s² (dur) |
| vol, Y : D²(c+s) = −g | parabole cohérente avec la durée du vol | idem |
| raccord : D²(c+s) au dernier appui et à la réception | la vitesse ne saute pas | σ = 8 m/s² |
| **s(réception) = s(décollage)** en X et Z | **distance de vol conservée** | 1 mm |
| s(t+1) − s(t) en appui | la correction ne fait pas patiner le pied posé | σ = 0,15 m/s |
| s_y = 0 en appui | le pied posé reste au sol | 3 mm |
| bornes : a ≤ a_max, v ≤ v_max (horizontal, échelle 0,1 s) | vitesses humaines | a_max 15 m/s² (25 « explosif »), v_max 10 m/s |
| option : variance du pied par appui | anti-glissement | **coupée** (§5.4) |

Les bornes, non linéaires, sont traitées par passes : là où la trajectoire corrigée dépasse, on
ajoute D²(c+s) → 0 à cette image avec un poids qui quadruple tant qu'elle dépasse (lissage de
Whittaker local), et v → v écrêtée pour la vitesse.

Garde-fous : vol en bord de clip ignoré (décollage ou réception hors champ) ; vol qui s'écarte
de plus de 25 cm de sa droite laissé tel quel (saut de suivi, pas erreur de profondeur) ; si
les bornes demandent plus de 50 cm, repli sur la correction des vols seule ; si même celle-ci
dépasse 50 cm, marqueurs inchangés. Coût : 0,01-0,03 s par clip de 100-500 images, < 0,1 s
pour 1 800 images.

Sortie : `(marqueurs corrigés, décalages (T, 3) en mètres, rapport)`. `appliquer_decalages`
rejoue les décalages sur `verts_world` / `kpts_world` / `jc_world` (listes avec `None`,
rééchantillonnage si les longueurs diffèrent).

### 3.4 Pistes essayées et abandonnées (mesurées)

| essai | effet mesuré | décision |
|---|---|---|
| bornes visant l'accélération **écrêtée** (même direction, norme à la borne) | sur un signal qui tremble, les cibles s'intègrent en dérive : `FINAL_basket` finissait 15 cm plus loin | remplacé par D² → 0 (une droite est dans le noyau de D², la tendance est intacte) |
| rappel uniforme 10 cm partout | le clip entier absorbait une part de la bosse du vol : 1,8 cm de décalage sur toute la station debout d'un saut synthétique | rappel 2 cm en appui |
| s(t+1) − s(t) en appui à 0,05 m/s | le décalage ne se relâche plus entre deux bonds, les corrections s'additionnent : hop 4,10 → 3,83 m de déplacement net | 0,15 m/s : 4,09 m |
| sans conservation de la distance de vol | la correction de forme fuit sur les extrémités : 2e bond du hop 114,6 → 109,5 cm sans bornes, **96,4 cm** avec | égalité dure s(réception) = s(décollage) : 114,1 cm |
| vols en bord de clip contraints | `DIAG_run` commence semelles 0,54 s « en l'air » (sol mal posé) : 60 cm de correction, garde-fou déclenché | ignorés par défaut |
| garde balistique « montée ≈ gT²/8 » | écarte les vols asymétriques (hop) | gravité d'une parabole libre |

## 4. Protocole

- **Sources.** (a) `_etapes_marqueurs.npz` (`SYNKRO_DUMP_ETAPES=1`, marqueurs *avant* lissage
  et anti-glissement) : on rejoue les étages existants avec les réglages de
  `demo_video_opensim.py` pour le module (colonne « existant »), puis l'étage physique ; (b) TRC
  pré-IK `markers_<nom>.trc`, déjà passés par les étages existants ; (c) les deux drop jumps
  de production du 2026-09-24 (`A`, 60 i/s ; `B`, 30 i/s), TRC pré-IK téléchargés en
  lecture seule.
- **Phases fixées** : détectées une fois sur l'entrée de l'étage, réutilisées avant et après
  (on juge la trajectoire, pas la détection).
- **Métriques.** Vol : écart maximal du centre de masse à la droite décollage → réception (X =
  profondeur, Z = latéral), RMS à la parabole g, montée, distance horizontale du vol, saut de
  vitesse au décollage. Appui : glissement = écart maximal du centroïde XZ du pied à sa médiane
  sur le **50 % central** de chaque appui. Global : déplacement net du bassin (médianes des
  premiers et derniers 10 %, au moins 0,3 s), vitesse et accélération horizontales du centre de
  masse à l'échelle 0,1 s (P99). Rigidité : écart maximal à une translation commune, variation de
  l'angle hanche-genou-cheville.
- ⚠️ **L'écart à la droite vaut 0 après correction par construction** : c'est tautologique. Les
  chiffres qui informent sont la valeur *avant* (taille de l'erreur) et les **coûts** —
  déplacement net, distance de vol, glissement, amplitude de la correction.
- ⚠️ Le glissement n'a de sens que sur les gestes à vol. En marche et en course, la détection
  par la hauteur fusionne des appuis traînés et l'on mesure le pas lui-même (20-30 cm) : ces
  lignes servent seulement à vérifier que l'étage ne change rien.
- Aucune de ces vidéos n'a de référence (plateforme, capture optoélectronique) : on mesure la
  cohérence physique et les coûts, pas l'exactitude.

## 5. Résultats

Réglage par défaut sauf mention « explosif » (a_max 25 m/s²). « corr h / v » = amplitude
maximale du décalage horizontal / vertical.

### 5.1 Sauts — le cas visé

| essai | vol | écart profondeur X | écart latéral Z | RMS parabole | montée CM (Bosco gT²/8) | distance du vol | glissement méd / P90 | dépl. net | corr h / v |
|---|---|---|---|---|---|---|---|---|---|
| `NUIT_jump` CMJ | 0,59 s | **8,8 → 0** (recul −8,4 → 0) | 3,3 → 0 | 1,3 → 0 cm | 50,4 → 46,9 (43,2) | 1,5 → 1,5 cm | 2,6/3,4 → 2,6/3,5 | 0,226 → 0,226 m | 8,9 / 3,7 cm |
| drop jump `A` | 0,48 s | **4,9 → 0** (avance +5,8 → +0,5) | 3,1 → 0 | 0,7 → 0 | 27,0 → 28,7 (28,7) | 8,1 → 8,1 | 1,5/9,0 → 1,4/8,8 | 0,170 → 0,170 | 5,4 / 2,7 |
| drop jump `B` | 0,47 s | **6,5 → 0** (+7,7 → +1,7) | **8,5 → 0** | 0,7 → 0 | 23,2 → 25,1 (26,7) | 3,0 → 3,0 | 1,4/6,7 → 1,4/6,9 | 0,172 → 0,172 | 10,0 / 2,4 |
| `NUIT_hop`, 4 bonds | 0,07-0,50 s | **9,2 → 0** (max 17,3) | 2,5 → 0 | 0,8 → 0 | — | 114,6 → 114,1 ; 151,3 → 151,2 | 5,7/14,2 → 5,0/17,1 (explosif **5,0/13,7**) | 4,102 → 4,087 (4,091) | 23,6 / 4,6 |

`MO_cmj` porte les mêmes marqueurs que `NUIT_jump` (résultats identiques). Le drop jump
`A` a un second vol court (0,08 s) avant le vol principal, corrigé aussi ; la station
sur la box est écartée (non balistique : « gravité » de 0,3 m/s²), celle de `B` comme
« passage en hauteur » (1,0 s).

Lecture :

- **La personne ne recule plus en sautant** : l'excursion de profondeur pendant le vol tombe
  de 5-8 cm à 0-2 cm (le reste est le déplacement réel décollage → réception, conservé).
- La **montée du centre de masse se rapproche de la hauteur de Bosco** sur les trois sauts
  (50,4 → 46,9 pour 43,2 ; 27,0 → 28,7 pour 28,7 ; 23,2 → 25,1 pour 26,7) : la verticale devient
  cohérente avec le temps de vol, que le module `d3.jump` mesure déjà par les semelles.
- **Coûts nuls** sur le déplacement net (≤ 1,5 cm) et sur la distance des bonds (≤ 0,5 cm).
- Hop : les bornes à 15 m/s² prennent la propulsion pour de la gigue et dégradent le glissement
  P90 (14,2 → 17,1 cm) ; à 25 m/s² il s'améliore (13,7). **Gestes explosifs : a_max 25.**

### 5.2 Départ sprint et course

| essai | vols corrigés | écart X avant | dépl. net | corr h / v | remarque |
|---|---|---|---|---|---|
| `NUIT_sprint` / `DIAG_sprint` | 2 | 0,1-0,4 cm | 3,469 → 3,468 m | 1,6 / 0,8 cm | rien à corriger |
| `DIAG_run` (course, repère monde, 8 m) | 7 | 2,0 (max 6,9) | 6,736 → 6,695 m (−0,6 %) | 33,7 / 3,0 | bornes actives sur 211 images, accélération P99 53 → 23 m/s² |
| `CAP_run_hh`, `FIX_run_hh`, `OSC_run_hh` (caméra portée) | 7-20 | 0,2-0,6 cm | inchangé | 0,5-3,4 / **2,2-5,6** | l'horizontal n'a rien à corriger ; la parabole, elle, bouge la verticale jusqu'à 5,6 cm |

En course, les vols durent 0,07-0,25 s : une erreur d'une image sur le décollage (17 ms à
60 i/s) change gT²/8 de 20 à 40 %, et la parabole imposée devient une source d'oscillation
verticale fausse. **Pas de parabole en course**, et pas d'intérêt horizontal mesurable.

### 5.3 Contrôles sans vol : marche, squat, basket

| essai | effet | dépl. net | corr max |
|---|---|---|---|
| `FIX_gait`, `NUIT_gait` | inchangé (bornes sur 2-6 images) | 0,108 → 0,108 ; 0,046 → 0,046 m | 0,2-0,6 cm |
| `DIAG_squat`, `DIAG_appsquat` | bornes seules, accélération P99 19-25 → 10 m/s² | 0,574 → 0,575 / 0,582 | 1,4-2,1 cm |
| `FINAL_basket` | aucune semelle ne quitte le sol de plus de 3,7 cm : pas de vol ; les bornes retirent un saut de suivi de départ (bassin +50 cm puis retour en 0,2 s) | 0,475 → 0,537 (explosif 0,501) | 33,6 cm (21,3) |
| `DIAG_basket` (sol ancien) | 1 vol court ; accélération P99 42 → 16, glissement P90 17,8 → 14,8 | 0,354 → 0,410 | 31,9 cm |
| `LIBRE_basket`, `SOL2_basket` (sols anciens, semelles à 70-90 cm) | garde-fous : bornes abandonnées, vol à 40 cm d'écart laissé tel quel | inchangé | — |

Le déplacement net de `FINAL_basket` bouge parce que le saut de suivi tombe **dans** la fenêtre
de référence du début : c'est le lissage d'un artefact, pas une dérive. ⚠️ Aucun basket local
récent n'a de vol exploitable (`SOL3_basket` est un passage interrompu, vidéo illisible) : le
geste libre n'est **pas validé** pour la partie vol.

### 5.4 L'ancrage des appuis (option `ancrer_appuis`) : pourquoi il reste coupé

Il améliore le glissement P90 sur les sauts (drop jump 9,0 → 6,1 cm, `FINAL_basket` 8,9 → 6,2,
`DIAG_basket` 17,8 → 7,8), mais il **reconstruit la trajectoire depuis les pieds** dès que les
appuis se chevauchent : déplacement net du squat suivi de marche 0,574 → **0,277 m**, marche
`FIX_gait` 0,108 → **0,014 m**. C'est exactement le constat du 2026-09-09 (`outputs/GAIT_VAR`,
+26 %). L'anti-glissement existant garde la main sur l'appui.

### 5.5 Rigidité : aucun angle ne change

Sur les 19 essais et toutes les variantes : écart maximal à une translation commune
**2 × 10⁻¹² mm**, variation de l'angle hanche-genou-cheville **≤ 8 × 10⁻¹³ °** (bruit du
flottant). Test unitaire dédié (`test_translation_rigide_aucun_angle_ne_change`).

## 6. Limites

1. **Pas de vérité terrain.** Les chiffres montrent la cohérence physique et l'absence de coût,
   pas l'exactitude. Une plateforme de force sur un CMJ et un drop jump trancherait (hauteur de
   Bosco, déplacement horizontal réel ≈ 0).
2. **La détection des vols dépend du sol.** Sol faux (anciens basket) → vols ratés ou faux vols ;
   les gardes les écartent mais ne les rattrapent pas. La descente d'une box n'est corrigée que
   si la station sur la box est reconnue comme appui (seuil de 10 cm) : non, en général.
3. **±1 image de flottement** : la correction verticale peut décaler d'une image le décollage ou
   la réception redétectés après coup (`NUIT_hop`, drop jump `A`). Le temps de vol publié
   par `d3.jump` doit rester lu **avant** l'étage, ou sur les semelles de la sortie finale en
   acceptant ±17-33 ms.
4. **Le raccord C1 n'est pas garanti** : la vitesse mesurée juste avant le décollage est parfois
   incompatible avec le vol (hop : 2 m/s d'écart). Le saut de vitesse diminue en général
   (hop 1,51 → 0,80 m/s, `DIAG_run` 1,06 → 0,56) mais augmente sur un drop jump (0,20 → 0,31).
5. **Caméra portée** : dans un repère qui accélère, « vitesse constante en vol » n'est
   qu'approchée. À exclure comme l'anti-glissement.
6. **Centre de masse approché** (de Leva, marqueurs cutanés) : erreur de quelques centimètres
   sur sa position relative, sans effet sur la posture mais avec un effet sur la forme imposée.
7. **Le TRC passe ensuite par l'IK.** Une translation commune de tous les marqueurs est
   exactement représentable par la translation du bassin (`pelvis_tx/ty/tz`) et le coût de l'IK
   y est invariant : elle devrait traverser l'IK intacte, contrairement au gel de marqueurs de pied
   (`docs/SUIVI_SOL_CONTACT.md`, 5,52 → 9,71 cm). **À vérifier sur un passage Docker** (angles
   du `.mot` identiques à 10⁻³ ° près, `pelvis_t*` décalés de s).

## 7. Plan d'intégration

### 7.1 Où l'appeler

Dans `demo_video_opensim.py`, **juste après le bloc anti-glissement** (fin de
`if _anti_skate_on and marker_names is not None:`, avant le commentaire `--feet_anchor`), sur
`markers_array` (TRC final, avant export et IK), avec les mêmes exclusions que l'anti-glissement :

```python
    _phys_shifts = None
    # a_max par geste : en place 15 m/s2, explosif 25 m/s2 (hop : 15 degrade le
    # glissement P90 14,2 -> 17,1 cm, 25 l'ameliore a 13,7).
    _PHYS = {"d3.jump": 15.0, "d3.drop_jump": 15.0, "d3.single_leg_hop": 25.0}
    _phys_on = (getattr(args, "physique_trajectoire", False)
                and marker_names is not None
                and args.module in _PHYS
                and not _is_treadmill and not getattr(args, "handheld", False))
    if _phys_on:
        from sam_3d_body.export.physique_trajectoire import trajectoire_physique
        markers_array, _phys_shifts, _rap_phys = trajectoire_physique(
            markers_array, marker_names, fps=out_fps, a_max_ms2=_PHYS[args.module])
        print(f"  [physique] vols corriges {_rap_phys.get('vols_corriges')} | correction "
              f"max {_rap_phys.get('correction_max_cm', 0):.1f} cm (translation, aucun angle "
              f"modifie) {_rap_phys['avertissements'] or ''}")
```

### 7.2 Pour quels modules

| geste | activer | réglage | motif |
|---|---|---|---|
| `d3.jump` (CMJ) | **oui** | défaut | 8,8 cm de recul en profondeur retirés, coût nul |
| `d3.drop_jump` | **oui** | défaut | 4,9-8,5 cm retirés, box correctement écartée |
| `d3.single_leg_hop` (jetons `slh`) | **oui** | `a_max_ms2=25` | 9-17 cm retirés, distance des bonds conservée à 0,5 cm |
| geste libre (`module is None`) | plus tard | `a_max_ms2=25` | aucun basket local avec vol propre pour valider |
| `d3.sprint_start` | non (inutile) | — | 0,4 cm à corriger |
| `d3.running`, `d3.gait` | **non** | — | rien à gagner à l'horizontale ; parabole nuisible sur des vols de 0,1-0,2 s |
| squat, STS, SLS | non | — | pas de vol ; les bornes n'y changent que 1-2 cm |
| cyclisme, tapis, caméra portée | **jamais** | — | pas de sol fixe / repère mobile |

### 7.3 Sous quel interrupteur

1. D'abord **opt-in** : `--physique_trajectoire` (store_true) et jeton de nom de fichier `phys`
   dans `aws/lambda_trigger.py` (convention `reference_filename_convention`).
2. Après un passage Docker local de validation (§7.5), **défaut** sur les trois modules du
   tableau, avec `--no_physique_trajectoire` pour couper (même schéma que `--no_traj_smooth`).

### 7.4 Rejouer les décalages sur le mesh

Les décalages portent **trois** composantes (la parabole agit sur Y), là où la boucle actuelle
ne rejoue que XZ :

```python
        # après la boucle `for _sh_xz in (_traj_shifts, _antiskate_shifts):`
        if _phys_shifts is not None:
            from sam_3d_body.export.physique_trajectoire import appliquer_decalages
            verts_world = appliquer_decalages(verts_world, _phys_shifts)
            kpts_world = appliquer_decalages(kpts_world, _phys_shifts)
            jc_world = appliquer_decalages(jc_world, _phys_shifts)
```

`appliquer_decalages` accepte les listes avec `None` et rééchantillonne si le mesh n'a pas le
même nombre d'images (règle de `_lateral_shifts`). Même ordre que l'application sur les
marqueurs : lissage → anti-glissement → physique.

### 7.5 Validation avant défaut (Docker local)

1. Rebuild local, passage de `CMJYT.mp4` (`d3.jump`), d'un drop jump et de `single_leg_hop.mp4`
   (`d3.single_leg_hop`) avec et sans `--physique_trajectoire`, `SYNKRO_DUMP_ETAPES=1`.
2. Vérifier : angles du `.mot` identiques (≤ 10⁻³ °) et `pelvis_t*` décalés de s (§6.7) ; sorties
   `d3.jump` / `d3.drop_jump` (hauteur, temps de vol, RSI) inchangées à une image près ; LSI du hop
   inchangé ; GLB : mesh et squelette alignés pendant le vol.
3. Consigner le constat dans `docs/SUIVI_SOL_CONTACT.md` (statut `À VÉRIFIER` → `CONFIRMÉ`)
   avant de passer au défaut, puis seulement pousser l'image ECR.

## 8. Recommandation

**Brancher, en opt-in puis par défaut, sur `d3.jump`, `d3.drop_jump` et `d3.single_leg_hop`**
(a_max 25 pour le hop). C'est exactement le cas signalé — la personne qui recule en sautant — et
il est corrigé (5-17 cm de dérive de profondeur retirés par vol) sans coût mesurable : angles
intacts par construction, déplacement net et distance des bonds conservés à 1,5 cm près,
glissement inchangé.

**Ne pas brancher** sur la marche, la course (caméra portée ou non), le squat, le lever de
chaise, le cyclisme : rien à gagner, et la parabole serait nuisible en course. Le geste libre
(basket) attend un passage récent avec de vrais vols ; l'ancrage des appuis (`ancrer_appuis`)
reste coupé, l'anti-glissement existant garde la main sur l'appui.
