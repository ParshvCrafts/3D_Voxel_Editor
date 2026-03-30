"""
Renderer Module for JARVIS Voxel Editor
========================================
ModernGL-based renderer with holographic visual effects.
"""

import moderngl
import numpy as np
import math
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pyrr import Matrix44, Vector3
from config import CONFIG
from voxel_engine import VoxelEngine, Voxel, VoxelState


@dataclass
class Camera:
    """3D camera with perspective projection."""
    position: np.ndarray
    target: np.ndarray
    up: np.ndarray
    fov: float
    aspect: float
    near: float
    far: float

    # Animation
    breathing_offset: float = 0.0

    def get_view_matrix(self) -> Matrix44:
        """Calculate view matrix."""
        # Apply breathing effect
        breathing_pos = self.position.copy()
        if CONFIG.camera.BREATHING_ENABLED:
            breathing_pos[1] += self.breathing_offset

        return Matrix44.look_at(
            Vector3(breathing_pos),
            Vector3(self.target),
            Vector3(self.up)
        )

    def get_projection_matrix(self) -> Matrix44:
        """Calculate projection matrix."""
        return Matrix44.perspective_projection(
            self.fov, self.aspect, self.near, self.far
        )

    def update_breathing(self, time: float):
        """Update camera breathing effect."""
        self.breathing_offset = (
            math.sin(time * CONFIG.camera.BREATHING_SPEED) *
            CONFIG.camera.BREATHING_AMPLITUDE
        )


class Renderer:
    """
    OpenGL renderer for the voxel editor.

    Features:
    - Instanced voxel rendering
    - Wireframe overlay
    - Grid floor
    - Bloom post-processing
    - HUD elements (cursor, loading circle, radial menu)
    """

    def __init__(self, ctx: moderngl.Context, width: int, height: int):
        """Initialize the renderer."""
        self.ctx = ctx
        self.width = width
        self.height = height

        # Camera setup
        cam_cfg = CONFIG.camera
        self.camera = Camera(
            position=np.array(cam_cfg.INITIAL_POSITION, dtype=np.float32),
            target=np.array(cam_cfg.LOOK_AT, dtype=np.float32),
            up=np.array(cam_cfg.UP_VECTOR, dtype=np.float32),
            fov=cam_cfg.FOV,
            aspect=width / height,
            near=cam_cfg.NEAR_PLANE,
            far=cam_cfg.FAR_PLANE
        )

        # Time tracking
        self.time = 0.0

        # Load shaders
        self._load_shaders()

        # Create geometry
        self._create_cube_geometry()
        self._create_wireframe_geometry()
        self._create_grid_geometry()
        self._create_fullscreen_quad()

        # Create framebuffers for post-processing
        self._create_framebuffers()

        # Instance buffers (will be updated per frame)
        self.max_instances = CONFIG.voxel.MAX_VOXELS
        self._create_instance_buffers()

    def _load_shaders(self):
        """Load and compile all shaders."""
        shader_dir = os.path.join(os.path.dirname(__file__), 'shaders')

        def load_shader(name: str) -> str:
            with open(os.path.join(shader_dir, name), 'r') as f:
                return f.read()

        # Voxel shader (faces with fresnel)
        self.voxel_program = self.ctx.program(
            vertex_shader=load_shader('voxel.vert'),
            fragment_shader=load_shader('voxel.frag')
        )

        # Wireframe shader
        self.wireframe_program = self.ctx.program(
            vertex_shader=load_shader('wireframe.vert'),
            fragment_shader=load_shader('wireframe.frag')
        )

        # Grid shader
        self.grid_program = self.ctx.program(
            vertex_shader=load_shader('grid.vert'),
            fragment_shader=load_shader('grid.frag')
        )

        # Post-processing shaders
        fullscreen_vert = load_shader('fullscreen.vert')

        self.bloom_extract_program = self.ctx.program(
            vertex_shader=fullscreen_vert,
            fragment_shader=load_shader('bloom_extract.frag')
        )

        self.blur_program = self.ctx.program(
            vertex_shader=fullscreen_vert,
            fragment_shader=load_shader('blur.frag')
        )

        self.bloom_combine_program = self.ctx.program(
            vertex_shader=fullscreen_vert,
            fragment_shader=load_shader('bloom_combine.frag')
        )

    def _create_cube_geometry(self):
        """Create cube vertex data for instanced rendering."""
        # Cube vertices with positions and normals
        # Each face has 2 triangles (6 vertices)
        half = CONFIG.voxel.GRID_SIZE / 2.0

        vertices = np.array([
            # Front face (+Z)
            -half, -half,  half,  0,  0,  1,
             half, -half,  half,  0,  0,  1,
             half,  half,  half,  0,  0,  1,
            -half, -half,  half,  0,  0,  1,
             half,  half,  half,  0,  0,  1,
            -half,  half,  half,  0,  0,  1,
            # Back face (-Z)
             half, -half, -half,  0,  0, -1,
            -half, -half, -half,  0,  0, -1,
            -half,  half, -half,  0,  0, -1,
             half, -half, -half,  0,  0, -1,
            -half,  half, -half,  0,  0, -1,
             half,  half, -half,  0,  0, -1,
            # Top face (+Y)
            -half,  half,  half,  0,  1,  0,
             half,  half,  half,  0,  1,  0,
             half,  half, -half,  0,  1,  0,
            -half,  half,  half,  0,  1,  0,
             half,  half, -half,  0,  1,  0,
            -half,  half, -half,  0,  1,  0,
            # Bottom face (-Y)
            -half, -half, -half,  0, -1,  0,
             half, -half, -half,  0, -1,  0,
             half, -half,  half,  0, -1,  0,
            -half, -half, -half,  0, -1,  0,
             half, -half,  half,  0, -1,  0,
            -half, -half,  half,  0, -1,  0,
            # Right face (+X)
             half, -half,  half,  1,  0,  0,
             half, -half, -half,  1,  0,  0,
             half,  half, -half,  1,  0,  0,
             half, -half,  half,  1,  0,  0,
             half,  half, -half,  1,  0,  0,
             half,  half,  half,  1,  0,  0,
            # Left face (-X)
            -half, -half, -half, -1,  0,  0,
            -half, -half,  half, -1,  0,  0,
            -half,  half,  half, -1,  0,  0,
            -half, -half, -half, -1,  0,  0,
            -half,  half,  half, -1,  0,  0,
            -half,  half, -half, -1,  0,  0,
        ], dtype=np.float32)

        self.cube_vbo = self.ctx.buffer(vertices.tobytes())
        self.cube_vertex_count = 36

    def _create_wireframe_geometry(self):
        """Create cube wireframe line geometry."""
        half = CONFIG.voxel.GRID_SIZE / 2.0

        # 12 edges, each edge has 2 vertices
        lines = np.array([
            # Bottom face
            -half, -half, -half,  half, -half, -half,
             half, -half, -half,  half, -half,  half,
             half, -half,  half, -half, -half,  half,
            -half, -half,  half, -half, -half, -half,
            # Top face
            -half,  half, -half,  half,  half, -half,
             half,  half, -half,  half,  half,  half,
             half,  half,  half, -half,  half,  half,
            -half,  half,  half, -half,  half, -half,
            # Vertical edges
            -half, -half, -half, -half,  half, -half,
             half, -half, -half,  half,  half, -half,
             half, -half,  half,  half,  half,  half,
            -half, -half,  half, -half,  half,  half,
        ], dtype=np.float32)

        self.wireframe_vbo = self.ctx.buffer(lines.tobytes())
        self.wireframe_vertex_count = 24

    def _create_grid_geometry(self):
        """Create grid floor geometry."""
        size = CONFIG.voxel.INITIAL_GRID_EXTENT * 5  # Large grid

        vertices = np.array([
            -size, 0, -size,
             size, 0, -size,
             size, 0,  size,
            -size, 0, -size,
             size, 0,  size,
            -size, 0,  size,
        ], dtype=np.float32)

        self.grid_vbo = self.ctx.buffer(vertices.tobytes())
        self.grid_vao = self.ctx.vertex_array(
            self.grid_program,
            [(self.grid_vbo, '3f', 'in_position')]
        )

    def _create_fullscreen_quad(self):
        """Create fullscreen quad for post-processing."""
        vertices = np.array([
            -1, -1, 0, 0,
             1, -1, 1, 0,
             1,  1, 1, 1,
            -1, -1, 0, 0,
             1,  1, 1, 1,
            -1,  1, 0, 1,
        ], dtype=np.float32)

        self.fullscreen_vbo = self.ctx.buffer(vertices.tobytes())

        # Create VAOs for post-processing programs
        self.bloom_extract_vao = self.ctx.vertex_array(
            self.bloom_extract_program,
            [(self.fullscreen_vbo, '2f 2f', 'in_position', 'in_uv')]
        )
        self.blur_vao = self.ctx.vertex_array(
            self.blur_program,
            [(self.fullscreen_vbo, '2f 2f', 'in_position', 'in_uv')]
        )
        self.bloom_combine_vao = self.ctx.vertex_array(
            self.bloom_combine_program,
            [(self.fullscreen_vbo, '2f 2f', 'in_position', 'in_uv')]
        )

    def _create_framebuffers(self):
        """Create framebuffers for post-processing."""
        # Main scene framebuffer
        self.scene_texture = self.ctx.texture((self.width, self.height), 4)
        self.scene_depth = self.ctx.depth_texture((self.width, self.height))
        self.scene_fbo = self.ctx.framebuffer(
            color_attachments=[self.scene_texture],
            depth_attachment=self.scene_depth
        )

        # Bloom extraction framebuffer (half resolution for performance)
        bloom_width = self.width // 2
        bloom_height = self.height // 2

        self.bloom_texture = self.ctx.texture((bloom_width, bloom_height), 4)
        self.bloom_fbo = self.ctx.framebuffer(color_attachments=[self.bloom_texture])

        # Blur ping-pong framebuffers
        self.blur_texture_a = self.ctx.texture((bloom_width, bloom_height), 4)
        self.blur_fbo_a = self.ctx.framebuffer(color_attachments=[self.blur_texture_a])

        self.blur_texture_b = self.ctx.texture((bloom_width, bloom_height), 4)
        self.blur_fbo_b = self.ctx.framebuffer(color_attachments=[self.blur_texture_b])

    def _create_instance_buffers(self):
        """Create buffers for instanced rendering."""
        # Instance data: position (3), color (3), rotation (3), selected (1) = 10 floats per instance
        instance_data = np.zeros((self.max_instances, 10), dtype=np.float32)
        self.instance_vbo = self.ctx.buffer(instance_data.tobytes(), dynamic=True)
        self.instance_count = 0

        # Keep preview rendering isolated so it never mutates live voxel instance data.
        preview_instance = np.zeros((1, 10), dtype=np.float32)
        self.preview_instance_vbo = self.ctx.buffer(preview_instance.tobytes(), dynamic=True)

        # Create VAO for voxel faces
        self.voxel_vao = self.ctx.vertex_array(
            self.voxel_program,
            [
                (self.cube_vbo, '3f 3f', 'in_position', 'in_normal'),
                (self.instance_vbo, '3f 3f 3f 1f /i', 'instance_position', 'instance_color',
                 'instance_rotation', 'instance_selected'),
            ]
        )

        # Create VAO for wireframe
        self.wireframe_vao = self.ctx.vertex_array(
            self.wireframe_program,
            [
                (self.wireframe_vbo, '3f', 'in_position'),
                (self.instance_vbo, '3f 3f 3f 4x /i', 'instance_position', 'instance_color',
                 'instance_rotation'),
            ]
        )

        self.preview_wireframe_vao = self.ctx.vertex_array(
            self.wireframe_program,
            [
                (self.wireframe_vbo, '3f', 'in_position'),
                (self.preview_instance_vbo, '3f 3f 3f 4x /i', 'instance_position', 'instance_color',
                 'instance_rotation'),
            ]
        )

    def update_instance_data(self, voxel_engine: VoxelEngine):
        """Update instance buffer with current voxel data.
        
        PHASE 9.3: Group transform (grab/rotate) is now applied via model matrix
        in the shader, NOT by modifying instance data. This is the YouTube approach.
        """
        voxels = list(voxel_engine.voxels.values())
        count = min(len(voxels), self.max_instances)

        if count == 0:
            self.instance_count = 0
            return

        # Build instance data array - use raw voxel positions/rotations
        # Group transform is applied via u_model uniform in shader
        instance_data = np.zeros((count, 10), dtype=np.float32)

        for i, voxel in enumerate(voxels[:count]):
            instance_data[i, 0:3] = voxel.position
            instance_data[i, 3:6] = voxel.color
            instance_data[i, 6:9] = voxel.rotation
            instance_data[i, 9] = 1.0 if voxel.is_selected else 0.0

        # Upload to GPU
        self.instance_vbo.write(instance_data.tobytes())
        self.instance_count = count

    def _get_preview_wireframe_color(self, color: Tuple[float, float, float],
                                     glow_intensity: float) -> Tuple[float, float, float]:
        """Clamp preview luminance so the wireframe stays visible without tinting the scene bloom."""
        preview_color = np.array(color, dtype=np.float32)

        if not CONFIG.render.BLOOM_ENABLED:
            return tuple(np.clip(preview_color, 0.0, 1.0))

        luminance = float(np.dot(preview_color, np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)))
        if luminance <= 0.0:
            return tuple(np.clip(preview_color, 0.0, 1.0))

        # wireframe.frag boosts brightness by ~1.5x, so keep preview luminance comfortably below bloom.
        max_input_luminance = (CONFIG.render.BLOOM_THRESHOLD * 0.9) / (1.5 * glow_intensity)
        if luminance > max_input_luminance:
            preview_color *= max_input_luminance / luminance

        return tuple(np.clip(preview_color, 0.0, 1.0))

    def render(self, voxel_engine: VoxelEngine, delta_time: float,
               ar_mode: bool = False, voxel_opacity: float = 1.0,
               grid_opacity: float = 1.0,
               preview_position: Tuple[float, float, float] = None,
               preview_color: Tuple[float, float, float] = None):
        """Render a frame.

        Args:
            voxel_engine: The voxel engine with scene data
            delta_time: Time since last frame
            ar_mode: If True, don't clear to background color (Phase 4 AR)
            voxel_opacity: Opacity of voxels (0.0-1.0, for AR holographic effect)
            grid_opacity: Opacity of the grid floor
            preview_position: Position for wireframe preview cube (None = no preview)
            preview_color: Color for preview cube (RGB tuple)
        """
        self.time += delta_time

        # Update camera
        self.camera.update_breathing(self.time)

        # Update instance data
        self.update_instance_data(voxel_engine)

        # Get matrices
        view = self.camera.get_view_matrix()
        projection = self.camera.get_projection_matrix()

        # --- Pass 1: Render scene to framebuffer ---
        self.scene_fbo.use()

        if ar_mode:
            # AR Mode: Clear to TRANSPARENT so webcam shows through
            # CRITICAL: Must clear color to (0,0,0,0) not just depth
            # This allows alpha blending to properly composite over webcam
            self.ctx.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        else:
            # Traditional mode: Clear to background color
            bg = CONFIG.colors.BACKGROUND
            self.ctx.clear(bg[0], bg[1], bg[2], 1.0)

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        # Render grid - SKIP in AR mode to prevent flickering
        if not ar_mode:
            self._render_grid(view, projection, grid_opacity)

        # PHASE 9.3: Get model matrix for group transform (grab/rotate)
        model_matrix = voxel_engine.get_model_matrix()

        # Render voxels
        if self.instance_count > 0:
            self._render_voxels(view, projection, voxel_opacity, model_matrix)

        # PHASE 10: Render preview cube in same pass (no flickering)
        if preview_position is not None and preview_color is not None:
            self._render_preview_cube(view, projection, preview_position, preview_color)

        # --- Pass 2: Bloom post-processing ---
        if CONFIG.render.BLOOM_ENABLED:
            self._render_bloom()

        # --- Pass 3: Final composite to screen ---
        self._render_final_composite(ar_mode)

    def _render_grid(self, view: Matrix44, projection: Matrix44, opacity: float = 1.0):
        """Render the grid floor."""
        self.grid_program['u_projection'].write(projection.astype('f4').tobytes())
        self.grid_program['u_view'].write(view.astype('f4').tobytes())
        self.grid_program['u_camera_pos'].value = tuple(self.camera.position)
        self.grid_program['u_grid_color'].value = CONFIG.colors.CYAN
        self.grid_program['u_grid_spacing'].value = CONFIG.voxel.GRID_SIZE
        self.grid_program['u_fade_distance'].value = CONFIG.render.GRID_FADE_DISTANCE
        self.grid_program['u_time'].value = self.time
        # Set grid opacity for AR mode transparency (Phase 4)
        if 'u_opacity' in self.grid_program:
            self.grid_program['u_opacity'].value = opacity

        self.grid_vao.render(moderngl.TRIANGLES)

    def _render_voxels(self, view: Matrix44, projection: Matrix44, opacity: float = 1.0,
                       model_matrix: np.ndarray = None):
        """Render voxel faces and wireframes."""
        # Common uniforms
        proj_bytes = projection.astype('f4').tobytes()
        view_bytes = view.astype('f4').tobytes()
        
        # PHASE 9.3: Get model matrix for group transform (grab/rotate)
        if model_matrix is None:
            model_matrix = np.eye(4, dtype=np.float32)
        model_bytes = model_matrix.astype('f4').tobytes()

        # Render faces
        self.voxel_program['u_projection'].write(proj_bytes)
        self.voxel_program['u_view'].write(view_bytes)
        self.voxel_program['u_model'].write(model_bytes)  # PHASE 9.3: Group transform
        self.voxel_program['u_camera_pos'].value = tuple(self.camera.position)
        self.voxel_program['u_time'].value = self.time
        self.voxel_program['u_fresnel_power'].value = CONFIG.render.FRESNEL_POWER
        self.voxel_program['u_fresnel_bias'].value = CONFIG.render.FRESNEL_BIAS
        self.voxel_program['u_glow_intensity'].value = 1.0
        # Set voxel opacity for AR mode holographic effect (Phase 4)
        if 'u_opacity' in self.voxel_program:
            self.voxel_program['u_opacity'].value = opacity

        self.voxel_vao.render(moderngl.TRIANGLES, instances=self.instance_count)

        # Render wireframes on top
        self.wireframe_program['u_projection'].write(proj_bytes)
        self.wireframe_program['u_view'].write(view_bytes)
        self.wireframe_program['u_model'].write(model_bytes)  # PHASE 9.3: Group transform
        self.wireframe_program['u_time'].value = self.time
        self.wireframe_program['u_glow_intensity'].value = 1.5
        # Wireframe opacity (slightly brighter for AR effect)
        if 'u_opacity' in self.wireframe_program:
            self.wireframe_program['u_opacity'].value = min(1.0, opacity * 1.2)

        self.ctx.line_width = CONFIG.render.GRID_LINE_WIDTH
        self.wireframe_vao.render(moderngl.LINES, instances=self.instance_count)

    def _render_preview_cube(self, view: Matrix44, projection: Matrix44,
                             position: Tuple[float, float, float],
                             color: Tuple[float, float, float]):
        """Render a wireframe preview cube at the given position.
        
        PHASE 10: Simplified wireframe preview rendered in same pass as voxels.
        This eliminates flickering by avoiding separate render passes.
        
        Args:
            view: View matrix
            projection: Projection matrix
            position: World position for the cube center
            color: RGB color tuple for the wireframe
        """
        preview_glow_intensity = 0.85
        preview_color = self._get_preview_wireframe_color(color, preview_glow_intensity)

        proj_bytes = projection.astype('f4').tobytes()
        view_bytes = view.astype('f4').tobytes()

        # Create single-instance data for preview cube
        preview_data = np.zeros((1, 10), dtype=np.float32)
        preview_data[0, 0:3] = position  # Position
        preview_data[0, 3:6] = preview_color
        preview_data[0, 6:9] = (0, 0, 0) # No rotation
        preview_data[0, 9] = 0.0         # Not selected

        self.preview_instance_vbo.write(preview_data.tobytes())

        # Set wireframe uniforms
        self.wireframe_program['u_projection'].write(proj_bytes)
        self.wireframe_program['u_view'].write(view_bytes)
        self.wireframe_program['u_model'].write(np.eye(4, dtype=np.float32).tobytes())
        self.wireframe_program['u_time'].value = self.time
        self.wireframe_program['u_glow_intensity'].value = preview_glow_intensity
        if 'u_opacity' in self.wireframe_program:
            self.wireframe_program['u_opacity'].value = 0.9

        # Render single wireframe cube without touching the live voxel instance buffer.
        self.ctx.line_width = max(
            CONFIG.render.GRID_LINE_WIDTH,
            CONFIG.render.PREVIEW_WIREFRAME_WIDTH
        )
        self.preview_wireframe_vao.render(moderngl.LINES, instances=1)
        self.ctx.line_width = CONFIG.render.GRID_LINE_WIDTH

    def _render_bloom(self):
        """Apply bloom post-processing."""
        self.ctx.disable(moderngl.DEPTH_TEST)
        render_cfg = CONFIG.render

        # Extract bright pixels
        self.bloom_fbo.use()
        self.scene_texture.use(0)
        self.bloom_extract_program['u_texture'].value = 0
        self.bloom_extract_program['u_threshold'].value = render_cfg.BLOOM_THRESHOLD
        self.bloom_extract_vao.render(moderngl.TRIANGLES)

        # Gaussian blur passes
        for i in range(render_cfg.BLOOM_BLUR_PASSES):
            # Horizontal blur
            self.blur_fbo_a.use()
            if i == 0:
                self.bloom_texture.use(0)
            else:
                self.blur_texture_b.use(0)
            self.blur_program['u_texture'].value = 0
            self.blur_program['u_direction'].value = (1.0, 0.0)
            self.blur_program['u_resolution'].value = (self.width // 2, self.height // 2)
            self.blur_vao.render(moderngl.TRIANGLES)

            # Vertical blur
            self.blur_fbo_b.use()
            self.blur_texture_a.use(0)
            self.blur_program['u_texture'].value = 0
            self.blur_program['u_direction'].value = (0.0, 1.0)
            self.blur_program['u_resolution'].value = (self.width // 2, self.height // 2)
            self.blur_vao.render(moderngl.TRIANGLES)

    def _render_final_composite(self, ar_mode: bool = False):
        """Render final composite to screen."""
        self.ctx.screen.use()

        if ar_mode:
            # AR Mode: Don't clear - preserve webcam background
            # Note: Webcam was rendered to screen before render() was called
            pass
        else:
            self.ctx.clear(0, 0, 0, 1)

        self.scene_texture.use(0)
        self.blur_texture_b.use(1)

        self.bloom_combine_program['u_scene'].value = 0
        self.bloom_combine_program['u_bloom'].value = 1
        self.bloom_combine_program['u_bloom_intensity'].value = CONFIG.render.BLOOM_INTENSITY
        self.bloom_combine_program['u_time'].value = self.time
        self.bloom_combine_program['u_scanlines_enabled'].value = CONFIG.render.SCANLINE_ENABLED
        self.bloom_combine_program['u_scanline_density'].value = CONFIG.render.SCANLINE_DENSITY

        # In AR mode, use additive blending for holographic effect
        if ar_mode:
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        self.bloom_combine_vao.render(moderngl.TRIANGLES)

        if ar_mode:
            self.ctx.disable(moderngl.BLEND)

    def render_cursor(self, world_pos: Tuple[float, float, float],
                     state: str = "idle", progress: float = 0.0):
        """
        Render the 3D cursor at the given position.
        This is called separately after the main render.
        """
        # TODO: Implement 3D cursor rendering
        pass

    def render_loading_circle(self, screen_pos: Tuple[int, int],
                             progress: float, color: Tuple[float, float, float]):
        """Render loading circle progress indicator."""
        # TODO: Implement loading circle
        pass

    def render_radial_menu(self, center_pos: Tuple[float, float, float],
                          selected_index: int):
        """Render radial color selection menu."""
        # TODO: Implement radial menu
        pass

    def resize(self, width: int, height: int):
        """Handle window resize."""
        self.width = width
        self.height = height
        self.camera.aspect = width / height

        # Recreate framebuffers
        self._create_framebuffers()

    def cleanup(self):
        """Release GPU resources."""
        # ModernGL handles cleanup automatically through context
        pass


if __name__ == "__main__":
    print("Renderer module - requires ModernGL context to run")
    print("Run through main.py for full application")
