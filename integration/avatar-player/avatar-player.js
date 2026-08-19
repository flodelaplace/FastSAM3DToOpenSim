// Lecteur d'avatars Synkro — assemble un corps et un geste charges separement.
//
// Principe : les 10 corps sont telecharges UNE fois et gardes en cache
// (~40 Mo au total). Par exercice on ne charge que l'animation, 0,3 a 0,5 Mo.
// Changer d'avatar ne retelecharge donc rien.
//
// Deux pieges que ce module encapsule :
//
//   1. Un maillage skinne ne se clone PAS avec .clone() : les clones
//      partageraient le meme squelette et bougeraient ensemble. Il faut
//      SkeletonUtils.clone(), sinon deux avatars affiches cote a cote sont
//      indissociables.
//
//   2. Un clip ne vaut que pour SON gabarit. Le retargeting depend de la
//      morphologie (recalage au sol de -38 a -56 cm selon l'avatar), donc
//      l'animation de `male_young` appliquee a `female_child` donne un
//      resultat faux — sans qu'aucune erreur ne soit levee.

import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { clone as cloneSkinned } from 'three/examples/jsm/utils/SkeletonUtils.js';

export class AvatarPlayer {
  /**
   * @param {string} baseUrl racine des assets, ex. 'https://cdn.../public'
   */
  constructor(baseUrl, loader = new GLTFLoader()) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.loader = loader;
    this._corps = new Map();   // avatarId -> gltf du template (charge une fois)
    this._clips = new Map();   // `${exo}/${avatarId}` -> AnimationClip
  }

  /** Template d'avatar, telecharge au plus une fois par id. */
  async corps(avatarId) {
    if (!this._corps.has(avatarId)) {
      this._corps.set(avatarId,
        this.loader.loadAsync(`${this.baseUrl}/_avatars/${avatarId}.glb`));
    }
    return this._corps.get(avatarId);
  }

  /** Animation d'un exercice pour un gabarit donne. */
  async clip(exerciceId, avatarId) {
    const cle = `${exerciceId}/${avatarId}`;
    if (!this._clips.has(cle)) {
      this._clips.set(cle,
        this.loader.loadAsync(`${this.baseUrl}/${exerciceId}/${avatarId}.glb`)
          .then((g) => g.animations[0]));
    }
    return this._clips.get(cle);
  }

  /**
   * Instance prete a afficher : { scene, mixer, duree }.
   * `scene` est a ajouter au graphe, `mixer.update(delta)` a appeler dans la
   * boucle de rendu.
   */
  async monter(exerciceId, avatarId) {
    const [gltf, anim] = await Promise.all([
      this.corps(avatarId),
      this.clip(exerciceId, avatarId),
    ]);

    // Clone squelette-safe : chaque instance a son propre squelette.
    const scene = cloneSkinned(gltf.scene);
    const mixer = new THREE.AnimationMixer(scene);
    mixer.clipAction(anim).play();
    mixer.update(0);

    return { scene, mixer, duree: anim.duration };
  }

  /** Precharge les corps en tache de fond, apres le premier affichage. */
  async prechargerCorps(ids) {
    await Promise.all(ids.map((id) => this.corps(id)));
  }
}

export const AVATARS = [
  'male_young', 'female_young',
  'male_young_large', 'female_young_large',
  'male_old', 'female_old',
  'male_old_large', 'female_old_large',
  'male_child', 'female_child',
];
