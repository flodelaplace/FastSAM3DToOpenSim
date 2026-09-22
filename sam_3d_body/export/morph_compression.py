"""Compression des GLB animes par morph targets, sans extension glTF.

Le mesh exporte porte une forme (morph target) PAR IMAGE : 359 formes de
18 439 sommets pesent 79 Mo sur les 82 Mo du fichier, soit 97 %. Le poids croit
donc lineairement avec la duree de la video.

Or ces formes sont massivement redondantes : un corps humain en mouvement vit
dans un sous-espace de faible dimension. On remplace donc les N formes par k
formes principales (decomposition par matrice de Gram) et des poids par image.
Mesure sur 6 s de course : 359 formes -> 91, 82 Mo -> 22 Mo, pour 0,047 mm
d'ecart maximal par sommet sur un mouvement de 1,3 m.

POURQUOI PAS gltfpack : il applique KHR_mesh_quantization, qui casse les morph
targets dans le viewer, et EXT_meshopt_compression, que Blender ne sait pas
lire sans greffon. Le fichier produit ici ne declare AUCUNE extension.

DEUX PIEGES, tous deux rencontres et corriges :

1. float64 obligatoire. En float32 la matrice de Gram ecrase les petites
   valeurs propres et fabrique une base fausse : 7,8 mm d'erreur au lieu
   de 0,002 mm sur exactement les memes donnees.

2. Les poids doivent tenir dans [0, 1]. Les coefficients d'une base principale
   sont signes et sans borne (mesures : -58 a +34) ; Blender borne les valeurs
   de cles de forme a [0, 1] et ramene le reste a zero, ce qui FIGE l'enveloppe
   pendant que le squelette continue de bouger. On replie donc l'echelle dans
   les formes et le decalage constant dans le maillage de base. La transformee
   devient affine : les poids d'animation doivent subir le meme recadrage,
   sinon l'image aux poids nuls ne rend plus la bonne pose.

Un ecart moyen faible ne prouve RIEN ici : la premiere version cassee affichait
0,0077 mm de moyenne avec une enveloppe immobile. C'est pourquoi la
verification compare aussi l'AMPLITUDE du mouvement.
"""
import json
import os
import struct

import numpy as np

GLB_MAGIC, JSON_CHUNK, BIN_CHUNK = 0x46546C67, 0x4E4F534A, 0x004E4942
_DTYPE = {5126: '<f4', 5123: '<u2', 5121: 'u1', 5125: '<u4', 5122: '<i2',
          5120: 'i1'}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _lire(path):
    b = open(path, 'rb').read()
    off, js, bn = 12, None, b''
    while off < len(b):
        ln, ty = struct.unpack_from('<II', b, off)
        off += 8
        if ty == JSON_CHUNK:
            js = json.loads(b[off:off + ln])
        elif ty == BIN_CHUNK:
            bn = b[off:off + ln]
        off += ln
    return js, bytearray(bn)


def _acc(js, bn, i):
    a = js['accessors'][i]
    bv = js['bufferViews'][a['bufferView']]
    o = bv.get('byteOffset', 0) + a.get('byteOffset', 0)
    n = _NCOMP[a['type']]
    arr = np.frombuffer(bytes(bn), dtype=np.dtype(_DTYPE[a['componentType']]),
                        count=a['count'] * n, offset=o)
    return arr.reshape(-1, n).astype(np.float32)


def _ajouter(js, bn, data, type_):
    raw = np.ascontiguousarray(data, dtype='<f4').tobytes()
    off = len(bn)
    bn.extend(raw)
    bn.extend(b'\x00' * ((4 - len(raw) % 4) % 4))
    js['bufferViews'].append({'buffer': 0, 'byteOffset': off,
                              'byteLength': len(raw)})
    a = {'bufferView': len(js['bufferViews']) - 1, 'componentType': 5126,
         'count': len(data), 'type': type_}
    if type_ == 'VEC3':          # POSITION exige min/max
        a['min'] = data.min(0).tolist()
        a['max'] = data.max(0).tolist()
    js['accessors'].append(a)
    return len(js['accessors']) - 1


def _ecrire(path, js, bn):
    jb = json.dumps(js, separators=(',', ':')).encode()
    jb += b' ' * ((4 - len(jb) % 4) % 4)
    bb = bytes(bn) + b'\x00' * ((4 - len(bn) % 4) % 4)
    out = struct.pack('<III', GLB_MAGIC, 2, 12 + 8 + len(jb) + 8 + len(bb))
    out += struct.pack('<II', len(jb), JSON_CHUNK) + jb
    out += struct.pack('<II', len(bb), BIN_CHUNK) + bb
    open(path, 'wb').write(out)


def _ramasse_miettes(js, bn):
    """Supprime accessors et bufferViews devenus orphelins.

    Sans ca les anciennes formes survivent dans le binaire et le fichier
    GROSSIT au lieu de maigrir (mesure : 82 Mo -> 108 Mo).
    """
    vus = set()
    for m in js.get('meshes', []):
        for p in m.get('primitives', []):
            vus.update(p.get('attributes', {}).values())
            if 'indices' in p:
                vus.add(p['indices'])
            for t in p.get('targets', []):
                vus.update(t.values())
    for an in js.get('animations', []):
        for s in an.get('samplers', []):
            vus.add(s['input'])
            vus.add(s['output'])
    for sk in js.get('skins', []):
        if 'inverseBindMatrices' in sk:
            vus.add(sk['inverseBindMatrices'])

    garde = sorted(vus)
    remap = {v: i for i, v in enumerate(garde)}
    js['accessors'] = [js['accessors'][i] for i in garde]
    for m in js.get('meshes', []):
        for p in m.get('primitives', []):
            p['attributes'] = {k: remap[v] for k, v in p.get('attributes', {}).items()}
            if 'indices' in p:
                p['indices'] = remap[p['indices']]
            p['targets'] = [{k: remap[v] for k, v in t.items()}
                            for t in p.get('targets', [])]
    for an in js.get('animations', []):
        for s in an.get('samplers', []):
            s['input'] = remap[s['input']]
            s['output'] = remap[s['output']]
    for sk in js.get('skins', []):
        if 'inverseBindMatrices' in sk:
            sk['inverseBindMatrices'] = remap[sk['inverseBindMatrices']]

    utilises = sorted({a['bufferView'] for a in js['accessors']
                       if 'bufferView' in a})
    remap_bv = {v: i for i, v in enumerate(utilises)}
    neuf = bytearray()
    bvs = []
    for v in utilises:
        bv = js['bufferViews'][v]
        o, L = bv.get('byteOffset', 0), bv['byteLength']
        deb = len(neuf)
        neuf.extend(bn[o:o + L])
        neuf.extend(b'\x00' * ((4 - len(neuf) % 4) % 4))
        d = {'buffer': 0, 'byteOffset': deb, 'byteLength': L}
        for cle in ('byteStride', 'target'):
            if cle in bv:
                d[cle] = bv[cle]
        bvs.append(d)
    for a in js['accessors']:
        if 'bufferView' in a:
            a['bufferView'] = remap_bv[a['bufferView']]
    js['bufferViews'] = bvs
    js['buffers'] = [{'byteLength': len(neuf)}]
    return neuf


def compresser_en_place(path, tol_mm=0.05, k_max=160, verbeux=True):
    """Reduit la base de formes d'un GLB morph, en remplacant le fichier.

    tol_mm borne l'ecart MAXIMAL par sommet, pas la moyenne. En cas d'echec
    ou de gain nul, le fichier d'origine est conserve intact.

    Renvoie (mo_avant, mo_apres, k, err_max_mm) ou None si rien n'a ete fait.
    """
    try:
        avant = os.path.getsize(path)
        js, bn = _lire(path)
        prim = js['meshes'][0]['primitives'][0]
        cibles = prim.get('targets', [])
        n = len(cibles)
        if n < 8:                       # rien a gagner
            return None
        nv = js['accessors'][prim['attributes']['POSITION']]['count']

        D = np.empty((n, nv * 3), dtype=np.float64)   # float64 : cf. en-tete
        for i, t in enumerate(cibles):
            D[i] = _acc(js, bn, t['POSITION']).ravel()

        G = D @ D.T
        val, vec = np.linalg.eigh(G)
        vec = vec[:, np.argsort(val)[::-1]]

        def base(kk):
            B = vec[:, :kk].T @ D
            nb = np.linalg.norm(B, axis=1, keepdims=True)
            nb[nb == 0] = 1
            B = B / nb
            return B, D @ B.T

        def err(B, W):
            return np.linalg.norm((D - W @ B).reshape(n, nv, 3), axis=2).max() * 1000

        lo, hi = 2, min(k_max, n)
        B, W = base(hi)
        if err(B, W) > tol_mm:
            k = hi
        else:
            while lo < hi:
                mid = (lo + hi) // 2
                Bm, Wm = base(mid)
                if err(Bm, Wm) <= tol_mm:
                    hi = mid
                else:
                    lo = mid + 1
            k = hi
            B, W = base(k)

        # Recadrage affine des poids dans [0, 1] (cf. piege 2 en en-tete)
        W_lin = W.copy()
        wmin = W.min(axis=0)
        etendue = W.max(axis=0) - wmin
        plat = etendue < 1e-12
        sure = np.where(plat, 1.0, etendue)
        decalage = (wmin @ B).reshape(nv, 3)
        B = B * sure[:, None]
        W = (W - wmin) / sure
        W[:, plat] = 0.0

        dist = np.linalg.norm((D - (W @ B + decalage.ravel())).reshape(n, nv, 3),
                              axis=2) * 1000
        err_max = float(dist.max())

        base_pos = _acc(js, bn, prim['attributes']['POSITION']) + decalage
        prim['attributes']['POSITION'] = _ajouter(js, bn,
                                                  base_pos.astype(np.float32), 'VEC3')
        prim['targets'] = [{'POSITION': _ajouter(js, bn,
                                                 B[j].reshape(nv, 3).astype(np.float32),
                                                 'VEC3')} for j in range(k)]
        if 'extras' in js['meshes'][0]:
            js['meshes'][0]['extras'].pop('targetNames', None)
        js['meshes'][0]['weights'] = [0.0] * k

        for an in js.get('animations', []):
            for ch in an['channels']:
                if ch['target']['path'] != 'weights':
                    continue
                s = an['samplers'][ch['sampler']]
                anciens = _acc(js, bn, s['output']).reshape(-1, n)
                coef = anciens.astype(np.float64) @ W_lin
                neufs = ((coef - wmin) / sure).astype(np.float32)
                neufs[:, plat] = 0.0
                s['output'] = _ajouter(js, bn, neufs.ravel().reshape(-1, 1),
                                       'SCALAR')
                s['interpolation'] = 'LINEAR'

        bn = _ramasse_miettes(js, bn)
        tmp = path + '.pca.tmp'
        _ecrire(tmp, js, bn)
        apres = os.path.getsize(tmp)
        if apres >= avant:              # aucun gain : on garde l'original
            os.remove(tmp)
            return None
        os.replace(tmp, path)
        if verbeux:
            print(f"  [morph] {n} formes -> {k} | {avant/1e6:.1f} Mo -> "
                  f"{apres/1e6:.1f} Mo ({100*apres/avant:.0f} %) | "
                  f"ecart max {err_max:.3f} mm")
        return avant / 1e6, apres / 1e6, k, err_max
    except Exception as exc:            # jamais au prix du fichier
        tmp = path + '.pca.tmp'
        if os.path.isfile(tmp):
            os.remove(tmp)
        print(f"  [morph] compression ignoree ({type(exc).__name__}: {exc})")
        return None


def compresser_meshopt(path, bits_position=16, verbeux=True):
    """Compression meshopt (EXT_meshopt_compression) du GLB, en place.

    A appliquer APRES `compresser_en_place` : la reduction de base divise le
    nombre de formes, meshopt compresse ensuite chacune (et les pistes
    d'animation). Mesure sur la course in situ de Ced (2026-09-22) :
    28,0 -> 3,6 Mo. Ecart apres decodage, sommets apparies au plus proche
    voisin (meshopt REORDONNE les sommets, les comparer par indice n'a pas de
    sens) : corps 0,05 mm en moyenne et 0,13 mm au pire sur les 115 formes,
    translations 0,5 mm, rotations 0,005 deg. A 14 bits (defaut de l'outil)
    l'ecart monte a 0,53 mm pour 0,2 Mo de gain : on garde 16.

    Le fichier declare EXT_meshopt_compression et KHR_mesh_quantization comme
    REQUISES : le lecteur doit savoir decoder meshopt. La visionneuse de l'app
    le sait (confirme 2026-09-22) ; Blender a besoin d'un importeur recent.
    Retire en 2026-08 pour ces raisons, remis a la demande de l'app.
    `SYNKRO_MESHOPT=0` le coupe.

    Renvoie (mo_avant, mo_apres) ou None ; en cas d'echec le fichier d'origine
    est conserve intact.
    """
    import shutil
    import subprocess
    if os.environ.get("SYNKRO_MESHOPT", "1") == "0":
        if verbeux:
            print("  [meshopt] coupe (SYNKRO_MESHOPT=0)")
        return None
    if shutil.which("gltf-transform") is None:
        if verbeux:
            print("  [meshopt] gltf-transform absent — fichier laisse tel quel")
        return None
    racine, ext = os.path.splitext(path)
    tmp = racine + ".meshopt" + ext
    try:
        avant = os.path.getsize(path)
        res = subprocess.run(
            ["gltf-transform", "meshopt", path, tmp, "--level", "high",
             "--quantize-position", str(int(bits_position))],
            capture_output=True, text=True, timeout=600)
        if res.returncode != 0 or not os.path.isfile(tmp):
            if verbeux:
                print(f"  [meshopt] echec (code {res.returncode}) : "
                      f"{(res.stderr or res.stdout)[:300]}")
            if os.path.isfile(tmp):
                os.remove(tmp)
            return None
        apres = os.path.getsize(tmp)
        os.replace(tmp, path)
        if verbeux:
            print(f"  [meshopt] {avant/1e6:.1f} Mo -> {apres/1e6:.1f} Mo "
                  f"(positions {bits_position} bits)")
        return avant / 1e6, apres / 1e6
    except Exception as e:                                  # pragma: no cover
        if verbeux:
            print(f"  [meshopt] exception {type(e).__name__}: {e}")
        if os.path.isfile(tmp):
            os.remove(tmp)
        return None
