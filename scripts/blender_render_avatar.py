"""Rend un aperçu PNG (face + profil) d'un GLB avatar, en headless.

Sert de contrôle visuel avant de déposer un avatar dans assets/avatars/ :
la validation géométrique (check_avatar_glb.py) ne dit rien sur l'allure —
un vêtement peut être structurellement valide et visuellement raté.

Moteur WORKBENCH : pas de GPU requis (WSL), rendu en quelques secondes, et
suffisant pour juger silhouette et tenue.

Usage :
    blender --background --python scripts/blender_render_avatar.py -- <in.glb> <out.png>
"""

import math
import sys

import bpy


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    glb_path, out_path = argv[0], argv[1]

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=glb_path)

    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("aucun mesh importé")

    # Boîte englobante de tous les meshes, en coordonnées monde.
    xs, ys, zs = [], [], []
    for o in meshes:
        for corner in o.bound_box:
            w = o.matrix_world @ type(o.location)(corner)
            xs.append(w.x); ys.append(w.y); zs.append(w.z)
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    cz = (min(zs) + max(zs)) / 2
    height = max(zs) - min(zs)

    # WORKBENCH = rapide et sans GPU, mais il rend la cornée transparente comme
    # un disque opaque : les yeux ressortent blancs et sans iris. Pour juger le
    # visage il faut CYCLES, qui respecte la transparence (plus lent, CPU).
    engine = argv[2].upper() if len(argv) > 2 else "WORKBENCH"
    scene = bpy.context.scene
    if engine == "CYCLES":
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = 48
        light_data = bpy.data.lights.new("Key", type="AREA")
        light_data.energy = 800
        light_data.size = 3.0
        light = bpy.data.objects.new("Key", light_data)
        light.location = (cx - 1.5, cy - 2.5, cz + 1.5)
        light.rotation_euler = (math.radians(65), 0, math.radians(-30))
        scene.collection.objects.link(light)
    else:
        scene.render.engine = "BLENDER_WORKBENCH"
        scene.display.shading.light = "STUDIO"
        scene.display.shading.color_type = "TEXTURE"
    scene.render.film_transparent = False
    scene.world = bpy.data.worlds.new("W")
    scene.world.color = (0.15, 0.15, 0.17)
    scene.render.resolution_x = 700
    scene.render.resolution_y = 1000
    scene.render.image_settings.file_format = "PNG"

    cam_data = bpy.data.cameras.new("Cam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = height * 1.15
    cam = bpy.data.objects.new("Cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam

    # Deux vues : face (-Y) et profil (+X), collées côte à côte par le caller
    # si besoin. Ici on rend la face, puis le profil sous <out>_side.png.
    head_z = max(zs) - height * 0.08
    views = [
        (out_path, (cx, cy - height * 2.0, cz), (math.pi / 2, 0, 0),
         height * 1.15),
        (out_path.replace(".png", "_side.png"),
         (cx + height * 2.0, cy, cz), (math.pi / 2, 0, math.pi / 2),
         height * 1.15),
        # Gros plan tête : l'âge se juge au visage (rides, affaissement), pas
        # à la silhouette — invisible sur un plan pied.
        (out_path.replace(".png", "_head.png"),
         (cx, cy - height * 2.0, head_z), (math.pi / 2, 0, 0),
         height * 0.22),
    ]
    for path, loc, rot, ortho in views:
        cam.location = loc
        cam.rotation_euler = rot
        cam_data.ortho_scale = ortho
        scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        print("RENDERED:", path)


main()
