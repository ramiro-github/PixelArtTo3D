"""Interface desktop (Tkinter): escolha uma imagem, o modelo 3D voxelizado
aparece na janela, girável com o mouse (renderizado via OpenGL, sem travar).

Uso:
    venv/bin/python voxelize_gui.py
"""

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from PIL import Image, ImageTk
from tkinterdnd2 import DND_FILES, TkinterDnD

from gl_viewer import VoxelGLFrame
from sprite_to_3d import build_voxel_mesh, downsample_to_grid, load_foreground, voxel_cells

DEFAULT_IMAGE = Path(__file__).parent / "mario-sprite.png"


class VoxelizeApp:
    def __init__(self, root):
        self.root = root
        root.title("Voxelize")
        root.geometry("1040x760")

        self.arr = None
        self.foreground_mask = None
        self.grid_fg = None
        self.grid_color = None
        self.cells = []
        self._debounce_id = None

        self._build_widgets()
        if DEFAULT_IMAGE.exists():
            self._load_image(DEFAULT_IMAGE)

    def _build_widgets(self):
        bar = ttk.Frame(self.root, padding=8)
        bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(bar, text="Escolher imagem...", command=self._pick_image).pack(side=tk.LEFT)

        self.thumb_label = ttk.Label(bar)
        self.thumb_label.pack(side=tk.LEFT, padx=(10, 8))

        ttk.Label(bar, text="ou arraste uma imagem sobre a janela", foreground="#888").pack(
            side=tk.LEFT, padx=(0, 14)
        )

        ttk.Label(bar, text="Resolução").pack(side=tk.LEFT, padx=(6, 4))
        self.resolution = tk.IntVar(value=48)
        res_scale = ttk.Scale(
            bar, from_=12, to=96, orient=tk.HORIZONTAL, length=140,
            variable=self.resolution, command=lambda _v: self._on_resolution_change(),
        )
        res_scale.pack(side=tk.LEFT)
        self.res_value_label = ttk.Label(bar, text="48", width=3)
        self.res_value_label.pack(side=tk.LEFT, padx=(2, 14))

        ttk.Label(bar, text="Profundidade").pack(side=tk.LEFT, padx=(6, 4))
        self.depth = tk.IntVar(value=2)
        depth_scale = ttk.Scale(
            bar, from_=1, to=8, orient=tk.HORIZONTAL, length=100,
            variable=self.depth, command=lambda _v: self._on_depth_change(),
        )
        depth_scale.pack(side=tk.LEFT)
        self.depth_value_label = ttk.Label(bar, text="2", width=2)
        self.depth_value_label.pack(side=tk.LEFT, padx=(2, 14))

        ttk.Button(bar, text="Salvar modelo...", command=self._save_model).pack(side=tk.LEFT)

        self.gl_view = VoxelGLFrame(self.root, width=700, height=700)
        self.gl_view.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.status = tk.StringVar(value="Escolha uma imagem para começar.")
        ttk.Label(self.root, textvariable=self.status, padding=(8, 4), anchor="w").pack(
            side=tk.BOTTOM, fill=tk.X
        )

        # Registrar o widget OpenGL (ou qualquer filho) como alvo de drop corrompe
        # a renderização GLX — só a janela raiz é seguro, e ainda pega drops em
        # qualquer ponto da janela.
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind("<<Drop>>", self._on_drop)

    def _pick_image(self):
        path = filedialog.askopenfilename(
            title="Escolher sprite",
            filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp *.gif"), ("Todos os arquivos", "*.*")],
        )
        if path:
            self._load_image(Path(path))

    def _on_drop(self, event):
        paths = self.root.tk.splitlist(event.data)
        if paths:
            self._load_image(Path(paths[0]))

    def _load_image(self, path: Path):
        self.status.set(f"Processando {path.name}...")
        self.root.update_idletasks()
        try:
            self.arr, self.foreground_mask = load_foreground(path)
        except Exception as exc:
            self.status.set(f"Erro ao abrir {path.name}: {exc}")
            return

        self.current_name = path.name
        self._update_thumbnail(path)
        self._regenerate_grid()

    def _update_thumbnail(self, path: Path):
        thumb = Image.open(path).convert("RGB")
        thumb.thumbnail((40, 40))
        photo = ImageTk.PhotoImage(thumb)
        self.thumb_label.configure(image=photo)
        self.thumb_label.image = photo

    def _on_resolution_change(self):
        self.res_value_label.configure(text=str(self.resolution.get()))
        if self.arr is None:
            return
        if self._debounce_id is not None:
            self.root.after_cancel(self._debounce_id)
        self._debounce_id = self.root.after(150, self._regenerate_grid)

    def _on_depth_change(self):
        depth = self.depth.get()
        self.depth_value_label.configure(text=str(depth))
        if self.cells:
            self.gl_view.set_depth(depth)
            self._update_status()

    def _regenerate_grid(self):
        self._debounce_id = None
        if self.arr is None:
            return
        self.grid_fg, self.grid_color = downsample_to_grid(
            self.arr, self.foreground_mask, self.resolution.get()
        )
        if not self.grid_fg.any():
            self.status.set("Nenhum voxel detectado — tente outra imagem ou aumente a resolução.")
            return
        self.cells = voxel_cells(self.grid_fg, self.grid_color)
        self.gl_view.set_voxels(self.cells, self.depth.get(), reset_camera=True)
        self._update_status()

    def _update_status(self):
        grid_h, grid_w = self.grid_fg.shape
        depth = self.depth.get()
        self.status.set(
            f"{self.current_name}  —  grade {grid_w}x{grid_h}  |  "
            f"{len(self.cells)} voxels  |  profundidade {depth}  |  "
            f"{len(self.cells) * 12} triângulos"
        )

    def _save_model(self):
        if not self.cells:
            return
        path = filedialog.asksaveasfilename(
            title="Salvar modelo",
            defaultextension=".glb",
            filetypes=[("glTF binário", "*.glb"), ("STL", "*.stl")],
        )
        if not path:
            return
        path = Path(path)
        mesh = build_voxel_mesh(self.grid_fg, self.grid_color, self.depth.get())
        mesh.export(path)
        if path.suffix.lower() == ".glb":
            mesh.export(path.with_suffix(".stl"))
        else:
            mesh.export(path.with_suffix(".glb"))
        self.status.set(f"Salvo: {path.stem}.glb e {path.stem}.stl")


if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = VoxelizeApp(root)
    root.mainloop()
