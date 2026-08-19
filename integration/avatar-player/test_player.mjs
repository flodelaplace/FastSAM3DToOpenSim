// Teste AvatarPlayer sur une copie locale de l'arborescence publique.
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { readFileSync } from 'node:fs';
import { AvatarPlayer, AVATARS } from './avatar-player.js';

// Loader de test : lit sur disque au lieu du reseau. L'app garde le vrai.
const loaderDisque = {
  _l: new GLTFLoader(),
  loadAsync(chemin) {
    const buf = readFileSync(chemin);
    const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
    return new Promise((res, rej) => this._l.parse(ab, '', res, rej));
  },
};

const player = new AvatarPlayer('./stage', loaderDisque);
const bougés = (scene) => {
  const os = [];
  scene.traverse((o) => { if (o.isBone) os.push(o); });
  return os;
};
const positions = (scene, os) => {
  scene.updateMatrixWorld(true);
  return os.map((b) => b.getWorldPosition(new THREE.Vector3()));
};

let echecs = 0;
const verifier = (nom, cond) => {
  console.log(`  ${cond ? '\x1b[32m✓\x1b[0m' : '\x1b[31m✗\x1b[0m'} ${nom}`);
  if (!cond) echecs++;
};

// --- 1. montage simple ------------------------------------------------------
console.log('\n1. Montage d\'un exercice');
const a = await player.monter('ex09_bird_dog_standard', 'male_young');
const osA = bougés(a.scene);
const p0 = positions(a.scene, osA);
a.mixer.update(a.duree * 0.5);
const p1 = positions(a.scene, osA);
const deplaces = p0.filter((p, i) => p.distanceTo(p1[i]) > 1e-4).length;
verifier(`${osA.length} os, ${deplaces} animes, duree ${a.duree.toFixed(1)} s`,
  deplaces > 100);

// --- 2. le corps n'est telecharge qu'une fois -------------------------------
console.log('\n2. Cache du corps');
let appels = 0;
const compteur = { loadAsync: (c) => { appels++; return loaderDisque.loadAsync(c); } };
const p2 = new AvatarPlayer('./stage', compteur);
await p2.monter('ex09_bird_dog_standard', 'male_young');
const apres1 = appels;
await p2.monter('ex03_fente_statique_standard', 'male_young');   // meme corps
const apres2 = appels;
verifier(`1er montage : ${apres1} requetes (corps + clip)`, apres1 === 2);
verifier(`2e exercice, meme corps : +${apres2 - apres1} requete (clip seul)`,
  apres2 - apres1 === 1);

// --- 3. deux instances ne partagent PAS leur squelette ----------------------
console.log('\n3. Independance des instances (SkeletonUtils.clone)');
const i1 = await player.monter('ex09_bird_dog_standard', 'female_young');
const i2 = await player.monter('ex03_fente_statique_standard', 'female_young');
const os1 = bougés(i1.scene), os2 = bougés(i2.scene);
const avant2 = positions(i2.scene, os2);
i1.mixer.update(i1.duree * 0.5);          // on n'avance QUE la premiere
const apresA = positions(i1.scene, os1);
const apresB = positions(i2.scene, os2);
const bougeA = avant2.filter((p, i) => p.distanceTo(apresA[i]) > 1e-4).length;
const bougeB = avant2.filter((p, i) => p.distanceTo(apresB[i]) > 1e-4).length;
verifier(`instance 1 avancee : ${bougeA} os deplaces`, bougeA > 100);
verifier(`instance 2 intacte : ${bougeB} os deplaces`, bougeB === 0);

// --- 4. les 10 gabarits montent tous ----------------------------------------
console.log('\n4. Les 10 gabarits');
let ok = 0;
for (const id of AVATARS) {
  try {
    const inst = await player.monter('ex09_bird_dog_standard', id);
    const os = bougés(inst.scene);
    const q0 = positions(inst.scene, os);
    inst.mixer.update(inst.duree * 0.5);
    const q1 = positions(inst.scene, os);
    if (q0.filter((p, i) => p.distanceTo(q1[i]) > 1e-4).length > 100) ok++;
  } catch (e) { console.log(`    ${id} : ${e.message}`); }
}
verifier(`${ok} / ${AVATARS.length} gabarits animes`, ok === AVATARS.length);

console.log(echecs === 0
  ? '\n\x1b[32mTOUS LES TESTS PASSENT\x1b[0m'
  : `\n\x1b[31m${echecs} ECHEC(S)\x1b[0m`);
process.exit(echecs === 0 ? 0 : 1);
