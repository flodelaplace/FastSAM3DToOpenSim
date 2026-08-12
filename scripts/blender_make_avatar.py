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

    for rel in spec.get("clothes", []):
        HumanService.add_mhclo_asset(
            _resolve(data_dir, rel), basemesh, asset_type="Clothes")
        print(f"[avatar]   vêtement : {rel}")

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
