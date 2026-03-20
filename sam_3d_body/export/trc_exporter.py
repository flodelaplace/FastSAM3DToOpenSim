"""
TRC file exporter for OpenSim.
Ported from https://github.com/AitorIriondo/SAM3D-OpenSim
"""
from pathlib import Path
from typing import List
import numpy as np


class TRCExporter:
    def __init__(self, fps: float = 30.0, units: str = "m"):
        self.fps = fps
        self.units = units

    def export(
        self,
        markers: np.ndarray,
        marker_names: List[str],
        output_path: str,
        start_frame: int = 1,
    ) -> str:
        """
        Write an OpenSim TRC file.

        Args:
            markers: (N_frames, N_markers, 3) array
            marker_names: list of marker label strings
            output_path: output .trc file path
            start_frame: frame number for the first row
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        num_frames, num_markers = markers.shape[0], markers.shape[1]

        lines = []
        lines.append(f"PathFileType\t4\t(X/Y/Z)\t{output_path.name}")
        lines.append("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames")
        lines.append(f"{self.fps:.6f}\t{self.fps:.6f}\t{num_frames}\t{num_markers}\t{self.units}\t{self.fps:.6f}\t{start_frame}\t{num_frames}")

        header = "Frame#\tTime"
        for name in marker_names:
            header += f"\t{name}\t\t"
        lines.append(header.rstrip("\t"))

        coord_header = "\t"
        for i in range(num_markers):
            coord_header += f"\tX{i+1}\tY{i+1}\tZ{i+1}"
        lines.append(coord_header)
        lines.append("")

        scale = 1000.0 if self.units == "mm" else 1.0

        for frame_idx in range(num_frames):
            frame_num = start_frame + frame_idx
            t = frame_idx / self.fps
            row = f"{frame_num}\t{t:.6f}"
            for j in range(num_markers):
                x, y, z = markers[frame_idx, j] * scale
                row += f"\t{x:.6f}\t{y:.6f}\t{z:.6f}"
            lines.append(row)

        output_path.write_text("\n".join(lines) + "\n")
        return str(output_path)
