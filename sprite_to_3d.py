"""Converte um sprite pixel art (PNG) em um modelo 3D voxelizado (.glb/.stl).

Uso:
    venv/bin/python sprite_to_3d.py mario-sprite.png --out output/mario --resolution 48 --depth 2
"""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage


def load_foreground(image_path: Path, sat_tol: int = 14, light_thresh: int = 180, alpha_thresh: int = 128):
    """Carrega a imagem e separa fundo de primeiro plano.

    Se a imagem tiver canal alpha com transparência real, usa o alpha (funciona
    com qualquer cor de fundo). Caso contrário, cai para detectar a região
    cinza-clara conectada às bordas (cobre o xadrez de transparência já achatado).
    """
    img_rgba = Image.open(image_path).convert("RGBA")
    arr_rgba = np.array(img_rgba)
    arr = arr_rgba[:, :, :3].astype(int)
    alpha = arr_rgba[:, :, 3]

    has_alpha = bool((alpha < 250).any())
    if has_alpha:
        foreground_mask = alpha >= alpha_thresh
        return arr, foreground_mask

    channel_range = arr.max(axis=2) - arr.min(axis=2)
    lightness = arr.mean(axis=2)
    bg_candidate = (channel_range <= sat_tol) & (lightness >= light_thresh)

    labels, _ = ndimage.label(bg_candidate)
    border_labels = set(labels[0, :]) | set(labels[-1, :]) | set(labels[:, 0]) | set(labels[:, -1])
    border_labels.discard(0)

    background_mask = np.isin(labels, list(border_labels))
    foreground_mask = ~background_mask
    return arr, foreground_mask


def downsample_to_grid(arr, foreground_mask, target_long_side: int, min_coverage: float = 0.4):
    """Reduz a arte para uma grade de pixel art (voxels), com voto de maioria por célula."""
    h, w, _ = arr.shape
    scale = target_long_side / max(h, w)
    grid_w = max(1, round(w * scale))
    grid_h = max(1, round(h * scale))

    col_edges = np.linspace(0, w, grid_w + 1).round().astype(int)
    row_edges = np.linspace(0, h, grid_h + 1).round().astype(int)

    grid_color = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    grid_fg = np.zeros((grid_h, grid_w), dtype=bool)

    for r in range(grid_h):
        y0, y1 = row_edges[r], max(row_edges[r] + 1, row_edges[r + 1])
        for c in range(grid_w):
            x0, x1 = col_edges[c], max(col_edges[c] + 1, col_edges[c + 1])
            cell_fg = foreground_mask[y0:y1, x0:x1]
            coverage = cell_fg.mean()
            if coverage >= min_coverage:
                cell_colors = arr[y0:y1, x0:x1][cell_fg]
                grid_color[r, c] = np.median(cell_colors, axis=0)
                grid_fg[r, c] = True

    return grid_fg, grid_color


def voxel_cells(grid_fg, grid_color):
    """Lista (x, y, r, g, b) de cada voxel de primeiro plano, Y já invertido
    (linha 0 da imagem = topo do modelo)."""
    grid_h, _ = grid_fg.shape
    rows, cols = np.nonzero(grid_fg)
    return [
        (int(c), int(grid_h - 1 - r), *[int(v) for v in grid_color[r, c]])
        for r, c in zip(rows, cols)
    ]


def build_voxel_mesh(grid_fg, grid_color, depth: int):
    """Extrude cada célula de primeiro plano em uma coluna de cubos coloridos."""
    grid_h, grid_w = grid_fg.shape
    boxes = []
    for r in range(grid_h):
        for c in range(grid_w):
            if not grid_fg[r, c]:
                continue
            color = grid_color[r, c]
            x = c
            y = grid_h - 1 - r  # inverte Y: linha 0 da imagem fica no topo do modelo
            for z in range(depth):
                box = trimesh.creation.box(extents=(1, 1, 1))
                box.apply_translation((x + 0.5, y + 0.5, z + 0.5))
                box.visual.face_colors = np.append(color, 255)
                boxes.append(box)

    if not boxes:
        raise ValueError("Nenhum voxel de primeiro plano encontrado — verifique a detecção de fundo.")

    return trimesh.util.concatenate(boxes)


def save_preview(grid_fg, grid_color, mesh, out_path: Path):
    """Salva um PNG com a silhueta 2D reconstruída e o modelo 3D em ângulo, para conferência visual."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 6))

    ax2d = fig.add_subplot(1, 2, 1)
    rgba = np.dstack([grid_color, grid_fg.astype(np.uint8) * 255])
    ax2d.imshow(rgba)
    ax2d.set_title("Grade de voxels (2D)")
    ax2d.axis("off")

    ax3d = fig.add_subplot(1, 2, 2, projection="3d")
    faces_colors = mesh.visual.face_colors[:, :3] / 255.0
    # matplotlib desenha o eixo Z sempre na vertical da tela; nosso Y (altura da
    # sprite) que deve ficar em pé, então trocamos Y<->Z só para a visualização.
    plot_vertices = mesh.vertices[:, [0, 2, 1]]
    ax3d.add_collection3d(
        __import__("mpl_toolkits.mplot3d.art3d", fromlist=["Poly3DCollection"]).Poly3DCollection(
            plot_vertices[mesh.faces], facecolor=np.repeat(faces_colors, 1, axis=0), edgecolor=None
        )
    )
    bounds = mesh.bounds
    x_range, y_range, z_range = bounds[1] - bounds[0]
    ax3d.set_xlim(bounds[0][0], bounds[1][0])
    ax3d.set_ylim(bounds[0][2], bounds[1][2])
    ax3d.set_zlim(bounds[0][1], bounds[1][1])
    ax3d.set_box_aspect((x_range, z_range, y_range))
    ax3d.view_init(elev=15, azim=-60)
    ax3d.axis("off")
    ax3d.set_title("Modelo 3D voxelizado")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_voxel_json(grid_fg, grid_color, depth: int, out_path: Path):
    """Exporta a grade de voxels em JSON compacto (usado pelo visualizador web)."""
    grid_h, grid_w = grid_fg.shape
    cells = [list(cell) for cell in voxel_cells(grid_fg, grid_color)]
    data = {"width": int(grid_w), "height": int(grid_h), "depth": int(depth), "cells": cells}
    out_path.write_text(json.dumps(data, separators=(",", ":")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Caminho do sprite PNG de entrada")
    parser.add_argument("--out", type=Path, default=Path("output/model"), help="Prefixo dos arquivos de saída")
    parser.add_argument("--resolution", type=int, default=48, help="Resolução (em voxels) do lado maior")
    parser.add_argument("--depth", type=int, default=2, help="Profundidade em voxels da extrusão")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    arr, foreground_mask = load_foreground(args.image)
    grid_fg, grid_color = downsample_to_grid(arr, foreground_mask, args.resolution)
    mesh = build_voxel_mesh(grid_fg, grid_color, args.depth)

    glb_path = args.out.with_suffix(".glb")
    stl_path = args.out.with_suffix(".stl")
    preview_path = args.out.with_suffix(".preview.png")
    json_path = args.out.with_suffix(".voxels.json")

    mesh.export(glb_path)
    mesh.export(stl_path)
    save_preview(grid_fg, grid_color, mesh, preview_path)
    save_voxel_json(grid_fg, grid_color, args.depth, json_path)

    print(f"Voxels: {grid_fg.sum()} (grade {grid_fg.shape[1]}x{grid_fg.shape[0]}, profundidade {args.depth})")
    print(f"Triangulos: {len(mesh.faces)}")
    print(f"Exportado: {glb_path}, {stl_path}")
    print(f"Preview: {preview_path}")
    print(f"JSON (visualizador web): {json_path}")


if __name__ == "__main__":
    main()
