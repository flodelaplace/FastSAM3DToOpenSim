"""Session d'inference reutilisable : charge les modeles UNE fois.

Mesure a l'origine de ce module (job AWS Batch de 285 frames, 257 s au total) :
    37 s  sync des checkpoints S3
    18 s  imports Python
    73 s  chargement des modeles + torch.compile + warmup TensorRT
    58 s  boucle d'inference GPU
    64 s  queue CPU (shape-lock, IK OpenSim, analytics, exports)
     6 s  upload
Les 128 premieres secondes sont du re-chauffage repaye a CHAQUE video parce que
chaque video a son propre process. Une session persistante les supprime.

Le contrat : `run()` prend exactement la meme ligne de commande que le CLI, et
`demo_video_opensim.py` reste utilisable tel quel. On ne duplique aucun flag.
"""
import os

import numpy as np
import torch

import demo_video_opensim as pipeline


class InferenceSession:
    """Modeles charges une fois, N videos traitees ensuite.

    Usage :
        s = InferenceSession()
        s.run(["--video_path", "a.mp4", "--person_height", "1.80"])
        s.run(["--video_path", "b.mp4", "--module", "d3.squat"])
    """

    def __init__(self, detector="yolo_pose",
                 detector_model="./checkpoints/yolo/yolo11m-pose.engine",
                 local_checkpoint="./checkpoints/sam-3d-body-dinov3"):
        self.estimator = pipeline.setup_sam_3d_body(
            detector_name=detector,
            detector_model=detector_model,
            local_checkpoint_path=local_checkpoint,
        )
        self.visualizer = pipeline.SkeletonVisualizer(line_width=2, radius=5)
        self.visualizer.set_pose_meta(pipeline.mhr70_pose_info)
        self.jobs_done = 0

    # -- isolation entre deux videos -----------------------------------------
    def reset(self):
        """Remet a zero tout ce qui pourrait fuir d'une video a la suivante.

        C'est le coeur du sujet : une fuite ici ne fait pas planter, elle
        produit des resultats FAUX et silencieux, ce qui est bien pire.
        """
        # Le detecteur garde `_tracking_enabled` colle apres une video
        # --multi_person : sans ca, toutes les suivantes passeraient par
        # run_yolo_pose_tracked (filtrage BoT-SORT) au lieu de run_yolo_pose.
        det = getattr(self.estimator, "detector", None)
        if det is not None and hasattr(det, "disable_tracking"):
            det.disable_tracking()

        # NO_FLOOR_CLAMP est desormais reecrit a chaque video par main(), donc
        # cette ligne est une ceinture. On la garde : le cout est nul et elle
        # protege si quelqu'un rajoute un jour un autre chemin qui la pose.
        os.environ.pop("NO_FLOOR_CLAMP", None)

        # Aucun appel a manual_seed n'a ete trouve sur le chemin de traitement,
        # donc l'etat du RNG ne devrait pas influer. Le figer coute zero et rend
        # deux videos independantes de leur ordre de passage, ce qui est
        # exactement ce que le test de validation cherche a prouver.
        torch.manual_seed(0)
        np.random.seed(0)

    # -- traitement -----------------------------------------------------------
    def run(self, argv):
        """Traite une video. `argv` est la ligne de commande, comme au CLI."""
        self.reset()
        # Un Namespace neuf a chaque fois : main() mute args.output_dir, donc
        # reutiliser le meme objet ferait ecrire la 2e video dans le dossier
        # de la 1re.
        args = pipeline.build_parser().parse_args(argv)
        try:
            pipeline.main(args, estimator=self.estimator,
                          visualizer=self.visualizer)
        finally:
            self.jobs_done += 1
        return args.output_dir

    # -- surveillance ---------------------------------------------------------
    def gpu_memory(self):
        """(alloue, reserve) en Mo — a logger apres chaque video.

        Si la courbe est plate sur une vingtaine de videos, la question de la
        fuite memoire est close ; sinon on sait qu'il faut un redemarrage
        periodique du worker.
        """
        if not torch.cuda.is_available():
            return (0.0, 0.0)
        return (torch.cuda.memory_allocated() / 1e6,
                torch.cuda.memory_reserved() / 1e6)
