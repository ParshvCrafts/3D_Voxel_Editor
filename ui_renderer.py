"""
UI Renderer Module for JARVIS Voxel Editor
===========================================
Handles rendering of:
- 3D animated cursor
- Radial color menu
- Particle system
- HUD overlays (webcam preview, help text)
- Loading circles
"""

import moderngl
import numpy as np
import math
import os
from typing import Tuple, List, Optional
from dataclasses import dataclass, field
from pyrr import Matrix44, Vector3
from config import CONFIG


@dataclass
class Particle:
    """A single ambient particle."""
    position: np.ndarray
    velocity: np.ndarray
    size: float
    alpha: float
    life: float
    max_life: float


class ParticleSystem:
    """Ambient particle system for holographic atmosphere."""

    def __init__(self, count: int = 100, bounds: Tuple[float, float, float] = (20, 15, 20)):
        self.count = count
        self.bounds = bounds
        self.particles: List[Particle] = []
        self._init_particles()

    def _init_particles(self):
        """Initialize particles with random positions."""
        for _ in range(self.count):
            self.particles.append(self._create_particle())

    def _create_particle(self) -> Particle:
        """Create a new particle with random properties."""
        return Particle(
            position=np.array([
                (np.random.random() - 0.5) * 2 * self.bounds[0],
                np.random.random() * self.bounds[1],
                (np.random.random() - 0.5) * 2 * self.bounds[2]
            ], dtype=np.float32),
            velocity=np.array([
                (np.random.random() - 0.5) * 0.5,
                np.random.random() * 0.3 + 0.1,
                (np.random.random() - 0.5) * 0.5
            ], dtype=np.float32),
            size=np.random.random() * 0.1 + 0.02,
            alpha=np.random.random() * 0.5 + 0.2,
            life=np.random.random() * 10,
            max_life=np.random.random() * 5 + 5
        )

    def update(self, delta_time: float):
        """Update all particles."""
        for i, p in enumerate(self.particles):
            # Update position
            p.position += p.velocity * delta_time

            # Update life
            p.life += delta_time

            # Respawn if out of bounds or dead
            if (p.life > p.max_life or
                abs(p.position[0]) > self.bounds[0] or
                p.position[1] > self.bounds[1] or
                abs(p.position[2]) > self.bounds[2]):
                self.particles[i] = self._create_particle()
                self.particles[i].position[1] = -self.bounds[1] * 0.5

    def get_instance_data(self) -> np.ndarray:
        """Get particle data for instanced rendering."""
        data = np.zeros((self.count, 5), dtype=np.float32)
        for i, p in enumerate(self.particles):
            data[i, 0:3] = p.position
            data[i, 3] = p.size
            # Fade alpha based on life
            life_factor = 1.0 - (p.life / p.max_life)
            data[i, 4] = p.alpha * life_factor
        return data


class UIRenderer:
    """
    Renderer for all UI elements.

    Handles:
    - 3D cursor with rotating rings
    - Radial color selection menu
    - Ambient particle system
    - HUD elements (webcam preview, overlays)
    """

    def __init__(self, ctx: moderngl.Context, width: int, height: int):
        self.ctx = ctx
        self.width = width
        self.height = height
        self.time = 0.0

        # Load shaders
        self._load_shaders()

        # Create geometry
        self._create_cursor_geometry()
        self._create_particle_geometry()
        self._create_quad_geometry()

        # Particle system
        self.particle_system = ParticleSystem(
            count=CONFIG.render.PARTICLE_COUNT,
            bounds=(20, 15, 20)
        )

        # Particle instance buffer
        self._create_particle_buffer()

        # Webcam texture
        self.webcam_texture: Optional[moderngl.Texture] = None

        # State - Primary cursor (right hand)
        self.cursor_position = (0.0, 0.0, 0.0)
        self.cursor_color = CONFIG.colors.CYAN
        self.cursor_progress = 0.0

        # State - Left hand cursor (Phase 2)
        self.left_cursor_position = (0.0, 0.0, 0.0)
        self.left_cursor_color = (0.3, 0.5, 1.0)  # Blue for left hand
        self.left_cursor_visible = False
        self.left_cursor_progress = 0.0
        self.left_hand_confidence = 1.0

        # State - Right hand cursor
        self.right_cursor_visible = True
        self.right_hand_confidence = 1.0

        # Ghost preview block (Phase 2)
        self.ghost_block_position: Optional[Tuple[float, float, float]] = None
        self.ghost_block_color = CONFIG.colors.CYAN
        self.ghost_block_visible = False

        # Two-hand connection line
        self.show_hand_connection = False
        self.hand_connection_color = (0.0, 1.0, 1.0)

        # Radial menu
        self.show_radial_menu = False
        self.radial_menu_center = (0.0, 0.0, 0.0)
        self.selected_color_index = 0

        # Loading circle state
        self.loading_circle_position = (0.0, 0.0, 0.0)
        self.loading_circle_progress = 0.0
        self.loading_circle_color = CONFIG.colors.CYAN
        self.loading_circle_visible = False

        # Symmetry mode (Phase 2)
        self.symmetry_enabled = False
        self.symmetry_axis = 'x'  # 'x', 'y', 'z'
        self.axis_lock_indicator = None
        self.sketch_voxels = []

    def _load_shaders(self):
        """Load UI shaders."""
        shader_dir = os.path.join(os.path.dirname(__file__), 'shaders')

        def load_shader(name: str) -> str:
            with open(os.path.join(shader_dir, name), 'r') as f:
                return f.read()

        # Cursor shader
        self.cursor_program = self.ctx.program(
            vertex_shader=load_shader('cursor.vert'),
            fragment_shader=load_shader('cursor.frag')
        )

        # Particle shader
        self.particle_program = self.ctx.program(
            vertex_shader=load_shader('particle.vert'),
            fragment_shader=load_shader('particle.frag')
        )

        # Radial menu shader
        self.radial_menu_program = self.ctx.program(
            vertex_shader=load_shader('radial_menu.vert'),
            fragment_shader=load_shader('radial_menu.frag')
        )

        # HUD shader
        self.hud_program = self.ctx.program(
            vertex_shader=load_shader('hud.vert'),
            fragment_shader=load_shader('hud.frag')
        )
        
        # Simple line shader for wireframe ghost block
        self.simple_line_program = self.ctx.program(
            vertex_shader=load_shader('simple_line.vert'),
            fragment_shader=load_shader('simple_line.frag')
        )

    def _create_cursor_geometry(self):
        """Create cursor quad geometry."""
        # Billboard quad with UVs
        vertices = np.array([
            # position (x, y, z), uv (u, v)
            -0.5, -0.5, 0.0, 0.0, 0.0,
             0.5, -0.5, 0.0, 1.0, 0.0,
             0.5,  0.5, 0.0, 1.0, 1.0,
            -0.5, -0.5, 0.0, 0.0, 0.0,
             0.5,  0.5, 0.0, 1.0, 1.0,
            -0.5,  0.5, 0.0, 0.0, 1.0,
        ], dtype=np.float32)

        self.cursor_vbo = self.ctx.buffer(vertices.tobytes())
        self.cursor_vao = self.ctx.vertex_array(
            self.cursor_program,
            [(self.cursor_vbo, '3f 2f', 'in_position', 'in_uv')]
        )
        
        # Create wireframe cube for ghost block preview (Phase 7)
        self._create_wireframe_cube()
    
    def _create_wireframe_cube(self):
        """Create wireframe cube geometry for ghost block preview.
        
        YouTube reference: THREE.BoxGeometry with wireframe: true
        Creates 12 edges of a unit cube centered at origin.
        """
        # 8 vertices of a unit cube centered at origin
        s = 0.5  # Half size
        v = [
            (-s, -s, -s),  # 0: back-bottom-left
            ( s, -s, -s),  # 1: back-bottom-right
            ( s,  s, -s),  # 2: back-top-right
            (-s,  s, -s),  # 3: back-top-left
            (-s, -s,  s),  # 4: front-bottom-left
            ( s, -s,  s),  # 5: front-bottom-right
            ( s,  s,  s),  # 6: front-top-right
            (-s,  s,  s),  # 7: front-top-left
        ]
        
        # 12 edges of the cube (pairs of vertex indices)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Back face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Front face
            (0, 4), (1, 5), (2, 6), (3, 7),  # Connecting edges
        ]
        
        # Build line vertices (each edge = 2 vertices)
        line_vertices = []
        for e in edges:
            line_vertices.extend(v[e[0]])
            line_vertices.extend([0.0, 0.0])  # Dummy UV
            line_vertices.extend(v[e[1]])
            line_vertices.extend([0.0, 0.0])  # Dummy UV
        
        vertices = np.array(line_vertices, dtype=np.float32)
        self.wireframe_vbo = self.ctx.buffer(vertices.tobytes())
        # Use cursor_program for wireframe rendering (same shader, different geometry)
        self.wireframe_vao = self.ctx.vertex_array(
            self.cursor_program,
            [(self.wireframe_vbo, '3f 2f', 'in_position', 'in_uv')]
        )

    def _create_particle_geometry(self):
        """Create particle billboard quad."""
        # Simple quad centered at origin
        vertices = np.array([
            -0.5, -0.5, 0.0,
             0.5, -0.5, 0.0,
             0.5,  0.5, 0.0,
            -0.5, -0.5, 0.0,
             0.5,  0.5, 0.0,
            -0.5,  0.5, 0.0,
        ], dtype=np.float32)

        self.particle_vbo = self.ctx.buffer(vertices.tobytes())

    def _create_particle_buffer(self):
        """Create particle instance buffer."""
        # Instance data: position (3), size (1), alpha (1) = 5 floats
        instance_data = np.zeros((CONFIG.render.PARTICLE_COUNT, 5), dtype=np.float32)
        self.particle_instance_vbo = self.ctx.buffer(instance_data.tobytes(), dynamic=True)

        self.particle_vao = self.ctx.vertex_array(
            self.particle_program,
            [
                (self.particle_vbo, '3f', 'in_position'),
                (self.particle_instance_vbo, '3f 1f 1f /i', 'instance_position',
                 'instance_size', 'instance_alpha'),
            ]
        )

    def _create_quad_geometry(self):
        """Create generic quad for radial menu and HUD."""
        vertices = np.array([
            # position (x, y, z), uv (u, v)
            -1.0, -1.0, 0.0, 0.0, 0.0,
             1.0, -1.0, 0.0, 1.0, 0.0,
             1.0,  1.0, 0.0, 1.0, 1.0,
            -1.0, -1.0, 0.0, 0.0, 0.0,
             1.0,  1.0, 0.0, 1.0, 1.0,
            -1.0,  1.0, 0.0, 0.0, 1.0,
        ], dtype=np.float32)

        self.quad_vbo = self.ctx.buffer(vertices.tobytes())

        # Radial menu VAO
        self.radial_menu_vao = self.ctx.vertex_array(
            self.radial_menu_program,
            [(self.quad_vbo, '3f 2f', 'in_position', 'in_uv')]
        )

        # HUD VAO (2D positions)
        hud_vertices = np.array([
            0.0, 0.0, 0.0, 1.0,
            1.0, 0.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
            1.0, 1.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
        ], dtype=np.float32)

        self.hud_vbo = self.ctx.buffer(hud_vertices.tobytes())
        self.hud_vao = self.ctx.vertex_array(
            self.hud_program,
            [(self.hud_vbo, '2f 2f', 'in_position', 'in_uv')]
        )

    def update(self, delta_time: float):
        """Update UI elements."""
        self.time += delta_time
        self.particle_system.update(delta_time)

    def set_cursor(self, position: Tuple[float, float, float],
                  color: Tuple[float, float, float] = None,
                  progress: float = 0.0):
        """Set right hand cursor position and state."""
        self.cursor_position = position
        if color:
            self.cursor_color = color
        self.cursor_progress = progress
        self.right_cursor_visible = True

    def set_left_cursor(self, position: Tuple[float, float, float],
                       color: Tuple[float, float, float] = None,
                       progress: float = 0.0,
                       confidence: float = 1.0):
        """Set left hand cursor position and state (Phase 2)."""
        self.left_cursor_position = position
        if color:
            self.left_cursor_color = color
        self.left_cursor_progress = progress
        self.left_cursor_visible = True
        self.left_hand_confidence = confidence

    def hide_left_cursor(self):
        """Hide the left hand cursor."""
        self.left_cursor_visible = False

    def set_hand_confidences(self, left: float = 1.0, right: float = 1.0):
        """Set tracking confidence for both hands (0-1)."""
        self.left_hand_confidence = left
        self.right_hand_confidence = right

    def show_ghost_block(self, position: Tuple[float, float, float],
                        color: Tuple[float, float, float] = None):
        """Show ghost preview block at snapped grid position (Phase 2)."""
        self.ghost_block_position = position
        if color:
            self.ghost_block_color = color
        self.ghost_block_visible = True

    def hide_ghost_block(self):
        """Hide the ghost preview block."""
        self.ghost_block_visible = False

    def show_loading_circle(self, position: Tuple[float, float, float],
                           progress: float, color: Tuple[float, float, float] = None):
        """Show loading/confirmation circle at position."""
        self.loading_circle_position = position
        self.loading_circle_progress = progress
        if color:
            self.loading_circle_color = color
        self.loading_circle_visible = True

    def hide_loading_circle(self):
        """Hide the loading circle."""
        self.loading_circle_visible = False

    def show_hand_connection_line(self, show: bool = True):
        """Show/hide connection line between two hands."""
        self.show_hand_connection = show

    def toggle_symmetry(self, axis: str = None) -> bool:
        """Toggle symmetry mode. Returns new state."""
        if axis:
            self.symmetry_axis = axis
        self.symmetry_enabled = not self.symmetry_enabled
        return self.symmetry_enabled

    def set_symmetry(self, enabled: bool, axis: str = 'x'):
        """Set symmetry mode state."""
        self.symmetry_enabled = enabled
        self.symmetry_axis = axis

    def get_symmetry_position(self, position: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Get mirrored position based on current symmetry axis."""
        if not self.symmetry_enabled:
            return position

        x, y, z = position
        if self.symmetry_axis == 'x':
            return (-x, y, z)
        elif self.symmetry_axis == 'y':
            return (x, -y, z)
        elif self.symmetry_axis == 'z':
            return (x, y, -z)
        return position

    # ============ YouTube-style Sketch Visualization (Phase 6 fix) ============
    
    def add_sketch_voxel(self, position: Tuple[float, float, float], color: Tuple[float, float, float]):
        """Add a voxel to the sketch preview (wireframe visualization).
        
        YouTube reference: addSketchVoxel() creates wireframe preview cubes
        that are committed to solid voxels when pinch is released.
        """
        # Store sketch voxel for rendering
        if not hasattr(self, 'sketch_voxels'):
            self.sketch_voxels = []
        
        # Avoid duplicates
        for sv in self.sketch_voxels:
            if sv['position'] == position:
                return
        
        self.sketch_voxels.append({
            'position': position,
            'color': color
        })
    
    def clear_sketch_voxels(self):
        """Clear all sketch voxels (called when build is committed or cancelled)."""
        if hasattr(self, 'sketch_voxels'):
            self.sketch_voxels.clear()
    
    def show_axis_lock(self, axis: str):
        """Show visual indicator for axis lock during continuous building.
        
        Args:
            axis: 'x' or 'y' - the locked axis
        """
        # Store axis lock state for HUD rendering
        self.axis_lock_indicator = axis
    
    def hide_axis_lock(self):
        """Hide axis lock indicator."""
        self.axis_lock_indicator = None

    def show_color_menu(self, center: Tuple[float, float, float], selected: int = 0):
        """Show the radial color menu."""
        self.show_radial_menu = True
        self.radial_menu_center = center
        self.selected_color_index = selected

    def hide_color_menu(self):
        """Hide the radial color menu."""
        self.show_radial_menu = False

    def update_webcam_texture(self, frame: np.ndarray):
        """Update webcam texture from OpenCV frame."""
        if frame is None:
            return

        # Convert BGR to RGB and flip vertically for OpenGL coordinate system
        # OpenGL has origin at bottom-left, OpenCV has origin at top-left
        import cv2
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.flip(frame_rgb, 0)  # Flip vertically for OpenGL
        height, width = frame_rgb.shape[:2]

        if self.webcam_texture is None:
            self.webcam_texture = self.ctx.texture((width, height), 3)
            self.webcam_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

        self.webcam_texture.write(frame_rgb.tobytes())

    def render_cursor(self, view: Matrix44, projection: Matrix44, camera_pos: np.ndarray):
        """Render the 3D cursor."""
        # Create model matrix (billboard facing camera)
        cursor_size = CONFIG.render.CURSOR_SIZE

        # Calculate billboard orientation
        to_camera = camera_pos - np.array(self.cursor_position)
        to_camera = to_camera / np.linalg.norm(to_camera)

        # Create rotation to face camera (simplified billboard)
        model = Matrix44.from_translation(Vector3(self.cursor_position))
        model = model * Matrix44.from_scale(Vector3([cursor_size, cursor_size, cursor_size]))

        # Set uniforms
        self.cursor_program['u_projection'].write(projection.astype('f4').tobytes())
        self.cursor_program['u_view'].write(view.astype('f4').tobytes())
        self.cursor_program['u_model'].write(model.astype('f4').tobytes())
        self.cursor_program['u_color'].value = self.cursor_color
        self.cursor_program['u_time'].value = self.time
        self.cursor_program['u_pulse'].value = CONFIG.render.CURSOR_PULSE_SPEED
        self.cursor_program['u_progress'].value = self.cursor_progress

        # Render
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        self.cursor_vao.render(moderngl.TRIANGLES)

    def render_particles(self, view: Matrix44, projection: Matrix44):
        """Render ambient particles."""
        # Update instance buffer
        instance_data = self.particle_system.get_instance_data()
        self.particle_instance_vbo.write(instance_data.tobytes())

        # Set uniforms
        self.particle_program['u_projection'].write(projection.astype('f4').tobytes())
        self.particle_program['u_view'].write(view.astype('f4').tobytes())
        self.particle_program['u_color'].value = CONFIG.colors.CYAN
        self.particle_program['u_time'].value = self.time

        # Render
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        self.particle_vao.render(moderngl.TRIANGLES, instances=len(self.particle_system.particles))

    def render_radial_menu(self, view: Matrix44, projection: Matrix44):
        """Render the radial color selection menu."""
        if not self.show_radial_menu:
            return

        palette = CONFIG.colors.get_palette()

        try:
            # Set uniforms
            self.radial_menu_program['u_projection'].write(projection.astype('f4').tobytes())
            self.radial_menu_program['u_view'].write(view.astype('f4').tobytes())
            self.radial_menu_program['u_center'].value = self.radial_menu_center
            self.radial_menu_program['u_radius'].value = CONFIG.ui.RADIAL_MENU_RADIUS
            self.radial_menu_program['u_time'].value = self.time
            self.radial_menu_program['u_num_colors'].value = len(palette)
            self.radial_menu_program['u_selected_index'].value = self.selected_color_index

            # Set color array - use try/except for each element in case shader optimized some out
            for i, color in enumerate(palette[:8]):
                try:
                    self.radial_menu_program[f'u_colors[{i}]'].value = color
                except KeyError:
                    pass  # Uniform optimized out by shader compiler

            # Render
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            self.radial_menu_vao.render(moderngl.TRIANGLES)
        except KeyError as e:
            # Shader uniform not found - skip rendering
            pass

    def render_webcam_preview(self, x: int, y: int, width: int, height: int):
        """Render webcam preview in corner."""
        if self.webcam_texture is None:
            return

        # Set uniforms
        self.hud_program['u_position'].value = (x, y)
        self.hud_program['u_size'].value = (width, height)
        self.hud_program['u_screen_size'].value = (self.width, self.height)
        self.hud_program['u_alpha'].value = 0.9
        self.hud_program['u_border_color'].value = (*CONFIG.colors.CYAN, 1.0)
        self.hud_program['u_border_width'].value = 2.0
        # DISABLED: Shader mirroring - frame is pre-mirrored with cv2.flip() in hand_tracker.py
        self.hud_program['u_mirror_x'].value = 0.0  # No shader mirror

        # Bind texture
        self.webcam_texture.use(0)
        self.hud_program['u_texture'].value = 0

        # Render
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.hud_vao.render(moderngl.TRIANGLES)

    def render_webcam_fullscreen(self, dim_factor: float = 0.0, mirror: bool = True):
        """Render webcam as fullscreen AR background (Phase 4/5).

        Args:
            dim_factor: How much to dim the webcam (0.0 - 1.0)
            mirror: IGNORED - frame is pre-mirrored in hand_tracker.py
        """
        if self.webcam_texture is None:
            return

        texture_width = self.webcam_texture.width
        texture_height = self.webcam_texture.height
        texture_aspect = texture_width / texture_height
        screen_aspect = self.width / self.height

        # Preserve the full camera frame without stretching or cropping.
        if texture_aspect > screen_aspect:
            draw_width = float(self.width)
            draw_height = draw_width / texture_aspect
            draw_x = 0.0
            draw_y = (self.height - draw_height) * 0.5
        else:
            draw_height = float(self.height)
            draw_width = draw_height * texture_aspect
            draw_x = (self.width - draw_width) * 0.5
            draw_y = 0.0

        self.hud_program['u_position'].value = (draw_x, draw_y)
        self.hud_program['u_size'].value = (draw_width, draw_height)
        self.hud_program['u_screen_size'].value = (self.width, self.height)
        # Apply dimming for holographic effect if requested
        self.hud_program['u_alpha'].value = 1.0 - dim_factor
        self.hud_program['u_border_color'].value = (0.0, 0.0, 0.0, 0.0)  # No border
        self.hud_program['u_border_width'].value = 0.0
        # DISABLED: Shader mirroring - frame is pre-mirrored with cv2.flip() in hand_tracker.py
        # This ensures text drawn on the frame is readable (not mirrored by shader)
        self.hud_program['u_mirror_x'].value = 0.0  # No shader mirror

        # Bind texture
        self.webcam_texture.use(0)
        self.hud_program['u_texture'].value = 0

        # Clear first so letterboxed/pillarboxed regions are stable and don't show stale pixels.
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)

        # Render without blending for background
        self.ctx.disable(moderngl.BLEND)
        self.hud_vao.render(moderngl.TRIANGLES)

    def render_left_cursor(self, view: Matrix44, projection: Matrix44, camera_pos: np.ndarray):
        """Render the left hand 3D cursor (Phase 2)."""
        if not self.left_cursor_visible:
            return

        cursor_size = CONFIG.render.CURSOR_SIZE * 0.9  # Slightly smaller than right

        # Apply confidence to size and alpha
        effective_size = cursor_size * (0.5 + 0.5 * self.left_hand_confidence)

        model = Matrix44.from_translation(Vector3(self.left_cursor_position))
        model = model * Matrix44.from_scale(Vector3([effective_size, effective_size, effective_size]))

        # Set uniforms
        self.cursor_program['u_projection'].write(projection.astype('f4').tobytes())
        self.cursor_program['u_view'].write(view.astype('f4').tobytes())
        self.cursor_program['u_model'].write(model.astype('f4').tobytes())
        self.cursor_program['u_color'].value = self.left_cursor_color
        self.cursor_program['u_time'].value = self.time
        self.cursor_program['u_pulse'].value = CONFIG.render.CURSOR_PULSE_SPEED
        self.cursor_program['u_progress'].value = self.left_cursor_progress

        # Render
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        self.cursor_vao.render(moderngl.TRIANGLES)

    def render_ghost_block(self, view: Matrix44, projection: Matrix44):
        """Render the ghost preview block as WIREFRAME cube (YouTube style).
        
        YouTube reference: THREE.BoxGeometry with wireframe: true, opacity: 0.5
        Renders 12 edges of a cube for clear visibility.
        """
        if not self.ghost_block_visible or self.ghost_block_position is None:
            return
        
        # Ghost block size matches grid
        ghost_size = CONFIG.voxel.GRID_SIZE

        model = Matrix44.from_translation(Vector3(self.ghost_block_position))
        model = model * Matrix44.from_scale(Vector3([ghost_size, ghost_size, ghost_size]))

        # Use cursor_program for wireframe rendering
        self.cursor_program['u_projection'].write(projection.astype('f4').tobytes())
        self.cursor_program['u_view'].write(view.astype('f4').tobytes())
        self.cursor_program['u_model'].write(model.astype('f4').tobytes())
        self.cursor_program['u_color'].value = self.ghost_block_color
        self.cursor_program['u_time'].value = self.time
        self.cursor_program['u_pulse'].value = 1.0
        self.cursor_program['u_progress'].value = 0.0

        # Render WIREFRAME cube using GL_LINES
        # PHASE 9 FIX: Proper rendering for AR mode compatibility
        # 1. Disable depth test so wireframe renders on top
        # 2. Use standard alpha blending (not additive) for AR mode
        # 3. Disable depth write to prevent z-fighting
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        # Use standard alpha blending for AR mode compatibility
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        # Set line width for visibility (if supported)
        try:
            self.ctx.line_width = max(
                CONFIG.render.GRID_LINE_WIDTH,
                CONFIG.render.PREVIEW_WIREFRAME_WIDTH
            )
        except:
            pass  # Line width may not be supported on all systems
        self.wireframe_vao.render(moderngl.LINES)
        try:
            self.ctx.line_width = CONFIG.render.GRID_LINE_WIDTH
        except:
            pass
        # Re-enable depth test
        self.ctx.enable(moderngl.DEPTH_TEST)

        # Render symmetry mirror ghost if enabled
        if self.symmetry_enabled:
            mirror_pos = self.get_symmetry_position(self.ghost_block_position)
            model = Matrix44.from_translation(Vector3(mirror_pos))
            model = model * Matrix44.from_scale(Vector3([ghost_size, ghost_size, ghost_size]))

            # Dimmer color for mirror ghost
            mirror_color = (self.ghost_block_color[0] * 0.5, self.ghost_block_color[1] * 0.5, self.ghost_block_color[2] * 0.5)
            self.cursor_program['u_model'].write(model.astype('f4').tobytes())
            self.cursor_program['u_color'].value = mirror_color
            self.wireframe_vao.render(moderngl.LINES)

    def render_loading_circle(self, view: Matrix44, projection: Matrix44, camera_pos: np.ndarray):
        """Render the loading/confirmation circle (Phase 2)."""
        if not self.loading_circle_visible or self.loading_circle_progress <= 0:
            return

        # Use cursor shader with progress for loading circle visualization
        circle_size = CONFIG.render.CURSOR_SIZE * 1.5

        model = Matrix44.from_translation(Vector3(self.loading_circle_position))
        model = model * Matrix44.from_scale(Vector3([circle_size, circle_size, circle_size]))

        # Color based on progress
        progress = self.loading_circle_progress
        if progress > 0.9:
            # Near complete - bright pulse
            intensity = 1.0 + 0.5 * math.sin(self.time * 20.0)
            color = (
                self.loading_circle_color[0] * intensity,
                self.loading_circle_color[1] * intensity,
                self.loading_circle_color[2] * intensity
            )
        else:
            color = self.loading_circle_color

        self.cursor_program['u_projection'].write(projection.astype('f4').tobytes())
        self.cursor_program['u_view'].write(view.astype('f4').tobytes())
        self.cursor_program['u_model'].write(model.astype('f4').tobytes())
        self.cursor_program['u_color'].value = color
        self.cursor_program['u_time'].value = self.time
        self.cursor_program['u_pulse'].value = CONFIG.render.CURSOR_PULSE_SPEED
        self.cursor_program['u_progress'].value = progress

        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        self.cursor_vao.render(moderngl.TRIANGLES)

    def render(self, view: Matrix44, projection: Matrix44, camera_pos: np.ndarray,
              show_webcam: bool = True, show_particles: bool = True,
              ar_mode: bool = False, ar_dim: float = 0.0):
        """Render all UI elements.

        Args:
            view: View matrix
            projection: Projection matrix
            camera_pos: Camera position for billboarding
            show_webcam: Whether to show webcam preview (ignored in AR mode)
            show_particles: Whether to render ambient particles
            ar_mode: If True, render webcam as fullscreen background (Phase 4)
            ar_dim: How much to dim the AR background (0.0 - 1.0)
        """
        # Particles (behind everything)
        if show_particles:
            self.render_particles(view, projection)

        # Ghost block preview (Phase 2)
        self.render_ghost_block(view, projection)

        # Radial menu (if active)
        self.render_radial_menu(view, projection)

        # Left hand cursor (Phase 2)
        self.render_left_cursor(view, projection, camera_pos)

        # Right hand cursor (primary)
        if self.right_cursor_visible:
            self.render_cursor(view, projection, camera_pos)

        # Loading circle (Phase 2)
        self.render_loading_circle(view, projection, camera_pos)

        # Webcam display
        if self.webcam_texture:
            if ar_mode:
                # AR Mode: Webcam is rendered as background by main render loop
                # This method is called AFTER the 3D scene, so we don't render background here
                pass
            elif show_webcam:
                # Traditional mode: Small preview in corner
                preview_height = 180
                preview_width = int(preview_height * self.webcam_texture.width / self.webcam_texture.height)
                self.render_webcam_preview(10, self.height - preview_height - 10,
                                           preview_width, preview_height)

    def render_ar_background(self, dim_factor: float = 0.0):
        """Render webcam as AR background (call BEFORE 3D scene)."""
        if self.webcam_texture:
            self.render_webcam_fullscreen(dim_factor)

    def resize(self, width: int, height: int):
        """Handle window resize."""
        self.width = width
        self.height = height


class DebugOverlay:
    """Debug overlay showing internal state (Phase 3)."""

    def __init__(self):
        self.visible = False
        self.data: dict = {}

    def toggle(self):
        """Toggle debug overlay visibility."""
        self.visible = not self.visible
        return self.visible

    def update(self, data: dict):
        """Update debug data."""
        self.data = data

    def get_lines(self) -> list:
        """Get debug info as list of lines for display."""
        if not self.visible:
            return []

        lines = [
            "=== DEBUG INFO ===",
            f"Mode: {self.data.get('mode', 'N/A')}",
            f"Gesture State: {self.data.get('gesture_state', 'IDLE')}",
            f"Progress: {self.data.get('gesture_progress', 0.0):.2f}",
            "",
            "--- Hand Data ---",
            f"Pinch Distance: {self.data.get('pinch_distance', 0.0):.1f}px",
            f"Fingers Up: {self.data.get('fingers_up', [0,0,0,0,0])}",
            f"Is Pinching: {self.data.get('is_pinching', False)}",
            f"Edge Triggered: {self.data.get('edge_triggered', False)}",
            "",
            "--- Cursor ---",
            f"World Pos: ({self.data.get('cursor_x', 0):.2f}, {self.data.get('cursor_y', 0):.2f}, {self.data.get('cursor_z', 0):.2f})",
            f"Grid Pos: {self.data.get('grid_pos', (0,0,0))}",
            "",
            "--- Performance ---",
            f"FPS: {self.data.get('fps', 0):.1f}",
            f"Voxel Count: {self.data.get('voxel_count', 0)}",
            f"Visible Voxels: {self.data.get('visible_voxels', 0)}",
            "",
            "--- State ---",
            f"Symmetry: {self.data.get('symmetry', 'OFF')}",
            f"Scatter State: {self.data.get('scatter_state', 'normal')}",
            f"AR Mode: {self.data.get('ar_mode', 'OFF')}",
        ]
        return lines


class HelpOverlay:
    """Help overlay showing gesture guide."""

    HELP_TEXT = """
    JARVIS VOXEL EDITOR - GESTURE GUIDE

    SINGLE HAND GESTURES:
      Pinch (index + thumb)     → Place block (hold 0.5s)
      Peace sign → Fist         → Delete block (hold 1.0s)
      Pinch + drag              → Extend/extrude blocks
      Open palm (hold)          → Open color menu
      Fast hand spread          → Scatter blocks!
      Fist (hold 2s)            → Recombine blocks

    TWO-HAND GESTURES (Phase 2):
      Left any + Right pinch    → Place block (more precise)
      Left point + Right pinch  → Delete at pointed location
      Left palm + Right fist    → Pan camera (drag fist)
      Both palms rotate         → Rotate selection
      Both pinch + spread       → Zoom camera in/out

    EFFECTS:
      Thumbs up (hold)          → Recombine scattered blocks
      Thumbs down + fast move   → Scatter with force

    KEYBOARD:
      H - Toggle this help
      W - Toggle webcam preview
      P - Toggle particles
      M - Toggle symmetry mode
      A - Toggle AR mode (webcam background)
      D - Toggle debug overlay
      S - Save scene
      L - Load scene
      E - Export to OBJ
      C - Clear all blocks
      R - Reset camera
      1-8 - Quick color select
      F1-F5 - Editor modes
      ESC - Quit
    """

    def __init__(self):
        self.visible = False

    def toggle(self):
        self.visible = not self.visible

    def get_text(self) -> str:
        return self.HELP_TEXT if self.visible else ""


if __name__ == "__main__":
    print("UI Renderer Module")
    print("Contains: 3D Cursor, Particle System, Radial Menu, HUD")
    print("Run through main.py for full application")
