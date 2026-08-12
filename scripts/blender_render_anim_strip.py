"""Rend une bande d'images d'un GLB ANIMÉ, pour juger un retarget d'un coup d'œil.

Un avatar peut passer toutes les validations géométriques et malgré tout bouger
faux (membre vrillé, pieds qui glissent, bassin figé). Seul le mouvement le dit.

Usage :
    blender --background --python scripts/blender_render_anim_strip.py -- <in.glb> <out_prefix> [n_frames]
"""

import math
import sys

import bpy


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    glb_path, out_prefix = argv[0], argv[1]
    n_frames = int(argv[2]) if len(argv) > 2 else 6

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=glb_path)

    scene = bpy.context.scene
    # NE PAS se fier à scene.frame_end : l'import glTF ne le règle pas, et une
    # scène Blender neuve vaut 1..250. On échantillonnerait alors très au-delà
    # de la fin de l'animation, où l'avatar tient sa dernière pose — ce qui
    # donne l'illusion d'un retarget figé.
    f0, f1 = None, None
    for action in bpy.data.actions:
        a0, a1 = action.frame_range
        f0 = a0 if f0 is None else min(f0, a0)
        f1 = a1 if f1 is None else max(f1, a1)
    if f0 is None:
        f0, f1 = scene.frame_start, scene.frame_end
    f0, f1 = int(round(f0)), int(round(f1))
    scene.frame_start, scene.frame_end = f0, f1
    print(f"FRAMES: {f0}..{f1}  ({len(bpy.data.actions)} action(s))")

    meshes = [o for o in scene.objects if o.type == "MESH"]
    xs, ys, zs = [], [], []
    for o in meshes:
        for c in o.bound_box:
            w = o.matrix_world @ type(o.location)(c)
            xs.append(w.x); ys.append(w.y); zs.append(w.z)
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    cz, height = (min(zs) + max(zs)) / 2, max(zs) - min(zs)

    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "TEXTURE"
    scene.world = bpy.data.worlds.new("W")
    scene.world.color = (0.15, 0.15, 0.17)
    scene.render.resolution_x = 460
    scene.render.resolution_y = 720
    scene.render.image_settings.file_format = "PNG"

    cam_data = bpy.data.cameras.new("Cam")
    cam_data.type = "ORTHO"
    # Cadrage large et FIXE : si la caméra suivait le sujet on ne verrait plus
    # ni le saut ni une éventuelle dérive au sol.
    cam_data.ortho_scale = height * 2.0
    cam = bpy.data.objects.new("Cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    cam.location = (cx, cy - height * 3.0, cz)
    cam.rotation_euler = (math.pi / 2, 0, 0)

    for i in range(n_frames):
        f = f0 + round(i * (f1 - f0) / max(n_frames - 1, 1))
        scene.frame_set(int(f))
        scene.render.filepath = f"{out_prefix}_{i:02d}.png"
        bpy.ops.render.render(write_still=True)
        print(f"RENDERED frame {f} → {scene.render.filepath}")


main()
