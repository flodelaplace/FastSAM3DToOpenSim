"""Génère un avatar MakeHuman (MPFB2) à partir d'une spec JSON, en headless.

Pourquoi ce script existe
-------------------------
Les 2 premiers avatars de production ont été faits à la main dans MakeHuman et
leurs sources `.mhm` ont été PERDUES — impossible d'en dériver une variante
(vieillir le sujet, changer la tenue) sans tout refaire. Ici la spec JSON EST
la source : elle est versionnée avec le repo, donc un avatar est toujours
reproductible et dérivable.

Le rig produit est le rig MakeHuman "default" (163 os), le seul que
`sam_3d_body/export/avatar_retarget.py` sache piloter. Attention : ce dernier
ignore SILENCIEUSEMENT un os qu'il ne trouve pas (avatar figé sans erreur), d'où
le script de validation `scripts/check_avatar_glb.py` à passer systématiquement.

Usage
-----
    blender --background --python scripts/blender_make_avatar.py -- <spec.json> <out.glb>
"""

import json
import os
import sys

import addon_utils
import bpy

MPFB = "bl_ext.blender_org.mpfb"

# Types d'asset MPFB, cf. objectservice._BODY_PART_TYPES
_BODY_PARTS = ("eyes", "eyelashes", "eyebrows", "tongue", "teeth", "hair")


def _mhclo_uuid(mhclo_path):
    with open(mhclo_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.lower().startswith("uuid "):
                return line.split(None, 1)[1].strip()
    return None


def _resolve(data_dir, rel):
    path = os.path.join(data_dir, rel)
    if not os.path.exists(path):
        raise FileNotFoundError(f"asset introuvable : {path}")
    return path


def _hem_top_z(top):
    """Altitude du point le plus haut de l'ourlet BAS du vêtement du haut.

    Les hauts MakeHuman ont plusieurs bords ouverts (col, deux poignets, ourlet).
    On les sépare en composantes connexes d'arêtes de bord et on retient celle
    qui contient le point le plus bas : c'est l'ourlet.
    """
    import bmesh

    bm = bmesh.new()
    bm.from_mesh(top.data)
    mw = top.matrix_world
    z = {v.index: (mw @ v.co).z for v in bm.verts}

    adj = {}
    for e in bm.edges:
        if len(e.link_faces) == 1:                     # arête de bord
            a, b = e.verts[0].index, e.verts[1].index
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
    if not adj:
        bm.free()
        return None

    seen, loops = set(), []
    for start in adj:
        if start in seen:
            continue
        stack, comp = [start], []
        seen.add(start)
        while stack:
            v = stack.pop()
            comp.append(v)
            for w in adj[v]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        loops.append(comp)
    bm.free()

    hem = min(loops, key=lambda c: min(z[v] for v in c))
    return max(z[v] for v in hem)


def _trim_waistband(top, lower):
    """Supprime la partie du bas de vêtement qui dépasse au-dessus de l'ourlet.

    Les ceintures MakeHuman sont modélisées dans le mesh du pantalon, à un rayon
    plus grand que le haut : elles transpercent donc le pull au lieu de passer
    dessous. Aucun matériau séparé ne permet de les isoler, et aucun haut du
    catalogue n'est assez long pour les couvrir.

    On coupe le pantalon à l'altitude du POINT LE PLUS HAUT de l'ourlet. Comme
    l'ourlet est partout à cette altitude ou en dessous, le pull recouvre la
    coupe sur tout le tour : pas de trou, et la ceinture disparaît.
    """
    import bmesh

    if top is None or lower is None:
        return
    cut = _hem_top_z(top)
    if cut is None:
        print("[avatar]   ceinture : ourlet introuvable, découpe ignorée")
        return

    bm = bmesh.new()
    bm.from_mesh(lower.data)
    mw = lower.matrix_world
    doomed = [v for v in bm.verts if (mw @ v.co).z > cut]
    if doomed:
        bmesh.ops.delete(bm, geom=doomed, context="VERTS")
        bm.to_mesh(lower.data)
        lower.data.update()
    bm.free()
    print(f"[avatar]   ceinture : {lower.name} coupé à z={cut:.3f} "
          f"({len(doomed)} verts retirés)")


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    spec_path, out_path = argv[0], argv[1]

    with open(spec_path) as f:
        spec = json.load(f)

    addon_utils.enable(MPFB, default_set=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)

    from bl_ext.blender_org.mpfb.services.humanservice import HumanService
    from bl_ext.blender_org.mpfb.services.locationservice import LocationService

    data_dir = LocationService.get_user_data()

    basemesh = HumanService.create_human(macro_detail_dict=spec["macro"])
    print(f"[avatar] {spec['name']} — taille {basemesh.dimensions[2]:.4f} m")

    # ⚠️ Le rig AVANT les vêtements. `add_builtin_rig` ne gréé que le corps ;
    # c'est `add_mhclo_asset` qui lie chaque asset à une armature DÉJÀ présente
    # (set_up_rigging / interpolate_weights). En riggant après, les vêtements,
    # cheveux et yeux sortaient du glTF avec `skin: null` : le corps suivait
    # l'animation, tout le reste restait figé en pose de bind — l'avatar se
    # disloquait dès qu'il bougeait, alors que le rendu statique semblait parfait.
    armature = HumanService.add_builtin_rig(basemesh, spec.get("rig", "default"))
    nbones = len(armature.data.bones) if armature else 0
    print(f"[avatar]   rig      : {spec.get('rig', 'default')} ({nbones} os)")

    # Peau AVANT les assets : set_character_skin retouche le matériau du corps.
    if spec.get("skin"):
        HumanService.set_character_skin(_resolve(data_dir, spec["skin"]), basemesh)
        print(f"[avatar]   peau     : {spec['skin']}")

    for part in _BODY_PARTS:
        rel = spec.get(part)
        if not rel:
            continue
        mhclo = _resolve(data_dir, rel)
        # Un asset n'a qu'une géométrie mais peut avoir plusieurs matériaux
        # (ex : short02 existe en gris, brun, roux). MPFB les sélectionne par
        # l'UUID déclaré dans le .mhclo, pas par un chemin — on va donc le lire.
        alt = None
        mat_rel = spec.get(f"{part}_material")
        if mat_rel:
            uuid = _mhclo_uuid(mhclo)
            if not uuid:
                raise ValueError(f"{rel} n'a pas d'uuid : matériau alternatif impossible")
            alt = {uuid: mat_rel}
        HumanService.add_mhclo_asset(
            _resolve(data_dir, rel), basemesh, asset_type=part.capitalize(),
            alternative_materials=alt)
        print(f"[avatar]   {part:<9}: {rel}"
              + (f"  [matériau {mat_rel}]" if mat_rel else ""))

    clothes_objs = []
    for rel in spec.get("clothes", []):
        before_objs = set(bpy.context.scene.objects)
        HumanService.add_mhclo_asset(
            _resolve(data_dir, rel), basemesh, asset_type="Clothes")
        new = [o for o in set(bpy.context.scene.objects) - before_objs
               if o.type == "MESH"]
        clothes_objs.append(new[0] if new else None)
        print(f"[avatar]   vêtement : {rel}")

    if spec.get("trim_waistband") and len(clothes_objs) >= 2:
        _trim_waistband(clothes_objs[0], clothes_objs[1])

    # MPFB masque la géométrie "helper" du corps (aide au drapé des vêtements,
    # cubes de repère articulaire) avec un modificateur MASK. L'export glTF
    # n'applique PAS les modificateurs : sans ça, ces helpers partent dans le
    # GLB et s'affichent comme une robe conique par-dessus les vêtements.
    # On applique donc le masque pour de bon avant d'exporter.
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = basemesh
    basemesh.select_set(True)
    before = len(basemesh.data.vertices)
    # Les targets MPFB (âge, genre, corpulence…) vivent comme shape keys, et
    # Blender refuse d'appliquer un modificateur tant qu'il en reste. On fige
    # donc le mélange courant dans le mesh — c'est de toute façon ce qu'on veut
    # exporter : une morphologie figée, pas un modèle encore paramétrable.
    if basemesh.data.shape_keys:
        bpy.ops.object.shape_key_remove(all=True, apply_mix=True)
    for mod in [m for m in basemesh.modifiers if m.type == "MASK"]:
        bpy.ops.object.modifier_apply(modifier=mod.name)
    print(f"[avatar]   helpers  : {before} → {len(basemesh.data.vertices)} verts")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(
        filepath=out_path, export_format="GLB", use_selection=True,
        export_yup=True, export_skins=True)
    print(f"[avatar] écrit : {out_path}")


main()
