"""Widget Tkinter com um viewport OpenGL: renderiza os voxels como cubos
instanciados na GPU, com órbita por arraste do mouse (sem o travamento do
render via matplotlib, que redesenha cada polígono em software)."""

import math
import os

os.environ.setdefault("PYOPENGL_PLATFORM", "glx")

import numpy as np
from OpenGL import GL
from OpenGL.GL import shaders
from pyopengltk import OpenGLFrame

VERTEX_SRC = """
#version 330
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec2 aOffset;
layout(location=3) in vec3 aColor;
uniform mat4 uViewProj;
uniform float uDepth;
out vec3 vNormal;
out vec3 vColor;
void main() {
    vec3 pos = vec3(aPos.xy + aOffset, aPos.z * uDepth);
    gl_Position = uViewProj * vec4(pos, 1.0);
    vNormal = aNormal;
    vColor = aColor;
}
"""

FRAGMENT_SRC = """
#version 330
in vec3 vNormal;
in vec3 vColor;
out vec4 outColor;
void main() {
    vec3 lightDir = normalize(vec3(0.45, 0.85, 0.6));
    float diffuse = max(dot(vNormal, lightDir), 0.0);
    float light = 0.55 + 0.5 * diffuse;
    outColor = vec4(vColor * light, 1.0);
}
"""

# cubo unitário (0..1), normais por face para sombreamento flat
_CUBE_VERTS = np.array([
    1,0,0, 1,1,0, 1,1,1,  1,0,0, 1,1,1, 1,0,1,
    0,0,1, 0,1,1, 0,1,0,  0,0,1, 0,1,0, 0,0,0,
    0,1,0, 0,1,1, 1,1,1,  0,1,0, 1,1,1, 1,1,0,
    0,0,1, 0,0,0, 1,0,0,  0,0,1, 1,0,0, 1,0,1,
    0,0,1, 1,0,1, 1,1,1,  0,0,1, 1,1,1, 0,1,1,
    1,0,0, 0,0,0, 0,1,0,  1,0,0, 0,1,0, 1,1,0,
], dtype=np.float32)

_CUBE_NORMALS = np.array([
    1,0,0, 1,0,0, 1,0,0,  1,0,0, 1,0,0, 1,0,0,
    -1,0,0, -1,0,0, -1,0,0,  -1,0,0, -1,0,0, -1,0,0,
    0,1,0, 0,1,0, 0,1,0,  0,1,0, 0,1,0, 0,1,0,
    0,-1,0, 0,-1,0, 0,-1,0,  0,-1,0, 0,-1,0, 0,-1,0,
    0,0,1, 0,0,1, 0,0,1,  0,0,1, 0,0,1, 0,0,1,
    0,0,-1, 0,0,-1, 0,0,-1,  0,0,-1, 0,0,-1, 0,0,-1,
], dtype=np.float32)


def _perspective(fovy, aspect, near, far):
    f = 1.0 / math.tan(fovy / 2.0)
    nf = 1.0 / (near - far)
    return np.array([
        f / aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, (far + near) * nf, -1,
        0, 0, 2 * far * near * nf, 0,
    ], dtype=np.float32)


def _look_at(eye, target, up):
    zx, zy, zz = eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]
    zl = math.sqrt(zx * zx + zy * zy + zz * zz)
    zx, zy, zz = zx / zl, zy / zl, zz / zl
    xx, xy, xz = up[1] * zz - up[2] * zy, up[2] * zx - up[0] * zz, up[0] * zy - up[1] * zx
    xl = math.sqrt(xx * xx + xy * xy + xz * xz)
    xx, xy, xz = xx / xl, xy / xl, xz / xl
    yx, yy, yz = zy * xz - zz * xy, zz * xx - zx * xz, zx * xy - zy * xx
    return np.array([
        xx, yx, zx, 0,
        xy, yy, zy, 0,
        xz, yz, zz, 0,
        -(xx * eye[0] + xy * eye[1] + xz * eye[2]),
        -(yx * eye[0] + yy * eye[1] + yz * eye[2]),
        -(zx * eye[0] + zy * eye[1] + zz * eye[2]), 1,
    ], dtype=np.float32)


def _multiply(a, b):
    out = np.zeros(16, dtype=np.float32)
    for col in range(4):
        for row in range(4):
            s = 0.0
            for k in range(4):
                s += a[k * 4 + row] * b[col * 4 + k]
            out[col * 4 + row] = s
    return out


class VoxelGLFrame(OpenGLFrame):
    """Frame OpenGL com órbita (arraste = girar, scroll = zoom)."""

    def initgl(self):
        if getattr(self, "_ready", False):
            return
        self._ready = True

        GL.glEnable(GL.GL_DEPTH_TEST)
        self.program = shaders.compileProgram(
            shaders.compileShader(VERTEX_SRC, GL.GL_VERTEX_SHADER),
            shaders.compileShader(FRAGMENT_SRC, GL.GL_FRAGMENT_SHADER),
        )
        self.u_view_proj = GL.glGetUniformLocation(self.program, "uViewProj")
        self.u_depth = GL.glGetUniformLocation(self.program, "uDepth")

        self.vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)

        self._static_buffer(0, _CUBE_VERTS, 3)
        self._static_buffer(1, _CUBE_NORMALS, 3)

        self.offset_buf = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.offset_buf)
        GL.glEnableVertexAttribArray(2)
        GL.glVertexAttribPointer(2, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glVertexAttribDivisor(2, 1)

        self.color_buf = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.color_buf)
        GL.glEnableVertexAttribArray(3)
        GL.glVertexAttribPointer(3, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glVertexAttribDivisor(3, 1)

        GL.glBindVertexArray(0)

        self.cell_count = 0
        self.depth = 2.0
        self.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.radius = 40.0
        self.yaw = -0.55
        self.pitch = 0.28
        self.auto_rotate = True
        self._idle_after_id = None
        self._dragging = False
        self._last_xy = (0, 0)
        self.bg_color = (0.06, 0.063, 0.08)

        self.animate = 16  # ~60fps redraw loop
        self._bind_controls()

    def _static_buffer(self, loc, data, size):
        buf = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buf)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(loc)
        GL.glVertexAttribPointer(loc, size, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

    def _bind_controls(self):
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<MouseWheel>", self._on_wheel)
        self.bind("<Button-4>", self._on_wheel)
        self.bind("<Button-5>", self._on_wheel)

    def _wake(self):
        self.auto_rotate = False
        if self._idle_after_id is not None:
            self.after_cancel(self._idle_after_id)
        self._idle_after_id = self.after(2600, self._resume_auto_rotate)

    def _resume_auto_rotate(self):
        self.auto_rotate = True
        self._idle_after_id = None

    def _on_press(self, event):
        self._dragging = True
        self._last_xy = (event.x, event.y)
        self._wake()

    def _on_release(self, _event):
        self._dragging = False

    def _on_drag(self, event):
        if not self._dragging:
            return
        lx, ly = self._last_xy
        dx, dy = event.x - lx, event.y - ly
        self._last_xy = (event.x, event.y)
        self.yaw += dx * 0.008
        self.pitch = max(-1.3, min(1.3, self.pitch + dy * 0.008))

    def _on_wheel(self, event):
        if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0:
            self.radius = max(6.0, self.radius - 2.0)
        else:
            self.radius = min(300.0, self.radius + 2.0)
        self._wake()

    def set_voxels(self, cells, depth, reset_camera=False):
        """cells: lista de (x, y, r, g, b) com r,g,b em 0..255."""
        n = len(cells)
        if n:
            arr = np.asarray(cells, dtype=np.float32)
            offsets = np.ascontiguousarray(arr[:, 0:2])
            colors = np.ascontiguousarray(arr[:, 2:5] / 255.0)
            max_x = offsets[:, 0].max() + 1
            max_y = offsets[:, 1].max() + 1
        else:
            offsets = np.zeros((0, 2), dtype=np.float32)
            colors = np.zeros((0, 3), dtype=np.float32)
            max_x = max_y = 1

        self.tkMakeCurrent()
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.offset_buf)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, offsets.nbytes, offsets, GL.GL_DYNAMIC_DRAW)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.color_buf)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, colors.nbytes, colors, GL.GL_DYNAMIC_DRAW)

        self.cell_count = n
        self.depth = float(depth)
        if reset_camera:
            self.target = np.array([max_x / 2, max_y / 2, self.depth / 2], dtype=np.float32)
            self.radius = max(max_x, max_y) * 1.35
            self.yaw, self.pitch = -0.55, 0.28
        else:
            self.target[2] = self.depth / 2

    def set_depth(self, depth):
        self.depth = float(depth)
        self.target[2] = self.depth / 2

    def redraw(self):
        GL.glClearColor(*self.bg_color, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        if self.cell_count == 0 or self.height <= 0:
            return

        if self.auto_rotate:
            self.yaw += 0.0022 * (self.animate / 16.0)

        eye = [
            self.target[0] + self.radius * math.cos(self.pitch) * math.sin(self.yaw),
            self.target[1] + self.radius * math.sin(self.pitch),
            self.target[2] + self.radius * math.cos(self.pitch) * math.cos(self.yaw),
        ]
        aspect = self.width / max(1, self.height)
        proj = _perspective(math.pi / 4.2, aspect, 0.1, 500.0)
        view = _look_at(eye, self.target, (0.0, 1.0, 0.0))
        view_proj = _multiply(proj, view)

        GL.glUseProgram(self.program)
        GL.glUniformMatrix4fv(self.u_view_proj, 1, GL.GL_FALSE, view_proj)
        GL.glUniform1f(self.u_depth, self.depth)
        GL.glBindVertexArray(self.vao)
        GL.glDrawArraysInstanced(GL.GL_TRIANGLES, 0, 36, self.cell_count)
