from __future__ import absolute_import

import ctypes
import functools
import math
import random
import time
import pdb
import numpy as np
import mathutils

import OpenGL.GL as gl
import OpenGL.GL.shaders as shaders
import OpenGL.GLU as glu
import OpenGL.GLUT as glut
from OpenGL.arrays import vbo

import pyopencl as cl; mf = cl.mem_flags
import pyopencl.cltypes as cltypes
import pyopencl.array as clarray
from pyopencl.tools import get_gl_sharing_context_properties

from Transform import Transform

import os
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"

def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer

    
# Global vars
controller      = None
camera          = None
screen_width    = 640
screen_height   = 480

# Shader and shader argument IDs
shader  = None
x_id    = None
y_id    = None
z_id    = None
modelview_matrix_id  = None
projection_matrix_id = None
color_id             = None

# Drawing/simulation configuration
draw_axes   = True
draw_grid   = True
draw_stars  = True
pause       = True
auto_rotate = True

# Mouse button enum
MOUSE_BUTTON_LEFT       = 0
MOUSE_BUTTON_MIDDLE     = 1
MOUSE_BUTTON_RIGHT      = 2
MOUSE_BUTTON_WHEEL_UP   = 3
MOUSE_BUTTON_WHEEL_DOWN = 4


# Shaders
vertex_shader = open("vertex.c", "r").read()
fragment_shader = open("fragment.c", "r").read()
 

def lerp(a, b, t):
    t = max(min(t, 1.0), 0.0)
    return a + (b - a) * t

class Galaxy():
    '''
    A class representing a galaxy, consisting of a center of mass and
    a collection of orbiting particles.
    '''

    def __init__(self, position, mass=1, body_count=-1, color=(1, 1, 1), ecc=1.0):
        self.transform  = Transform(pos=position, rot=-1)
        self.velocity   = 0 * mathutils.Vector(-0.25 * Universe.dt * self.position + (Universe.dt / 15) * Transform.random_vector())
        self.mass       = mass
        self.body_count = body_count
        self.color      = np.array(color, dtype=np.float32)
        self.ecc        = ecc

        if self.body_count < 0:
            self.body_count = 150000

        # buffer stuff
        self.body_positions             = np.ndarray((self.body_count, 4), dtype=np.float32)
        self.body_velocities            = np.ndarray((self.body_count, 4), dtype=np.float32)
        self.body_colors                = np.ndarray((self.body_count, 4), dtype=np.float32)
        self.body_positions_vbo         = vbo.VBO(data=self.body_positions, usage=gl.GL_DYNAMIC_DRAW, target=gl.GL_ARRAY_BUFFER)
        self.body_velocities_vbo        = vbo.VBO(data=self.body_velocities, usage=gl.GL_DYNAMIC_DRAW, target=gl.GL_ARRAY_BUFFER)
        self.body_colors_vbo            = vbo.VBO(data=self.body_colors, usage=gl.GL_DYNAMIC_DRAW, target=gl.GL_ARRAY_BUFFER)
        self.body_positions_cl_buffer   = None
        self.body_velocities_cl_buffer  = None
        self.body_colors_cl_buffer      = None

        self.body_positions_vbo.bind()
        self.body_velocities_vbo.bind()
        self.body_colors_vbo.bind()

        self.vertex_array               = gl.glGenVertexArrays(1)

    @property
    def position(self): return self.transform.position
    @position.setter
    def position(self, new_position): self.transform.position = new_position

    @property
    def rotation(self): return self.transform.rotation

    # Orientation Vectors
    @property
    def up(self):       return self.transform.up
    @property 
    def forward(self):  return self.transform.forward
    @property
    def right(self):    return self.transform.right

    def _generate_bodies(self):
        r1 = 0.5
        r2 = 2.5
        sigma = 0.25
        mu = (r1 + r2) / 2
        eccentricity = 0.0
        
        cl.enqueue_acquire_gl_objects(cl_queue, [self.body_positions_cl_buffer, self.body_velocities_cl_buffer, self.body_colors_cl_buffer])
        kernel_init(cl_queue,
                    (self.body_count,),
                    None,
                    self.body_positions_cl_buffer,
                    self.body_velocities_cl_buffer,
                    self.body_colors_cl_buffer,
                    np.uint(self.body_count),
                    np.array([self.position.x, self.position.y, self.position.z, self.mass], dtype=np.float32),
                    np.array([self.velocity.x, self.velocity.y, self.velocity.z, 0], dtype=np.float32),
                    np.array([self.rotation.x, self.rotation.y, self.rotation.z, self.rotation.w], dtype=np.float32),
                    np.float32(Universe.G),
                    np.float32(Universe.dt),
                    np.float32(mu),
                    np.float32(sigma),
                    np.float32(eccentricity))
        cl.enqueue_release_gl_objects(cl_queue, [self.body_positions_cl_buffer, self.body_velocities_cl_buffer, self.body_colors_cl_buffer])
        cl_queue.finish()

    def cl_init(self):
        t0 = time.time()
        self.body_positions_cl_buffer   = cl.GLBuffer(cl_context, mf.READ_WRITE, int(self.body_positions_vbo))
        self.body_velocities_cl_buffer  = cl.GLBuffer(cl_context, mf.READ_WRITE, int(self.body_velocities_vbo))
        self.body_colors_cl_buffer      = cl.GLBuffer(cl_context, mf.READ_WRITE, int(self.body_colors_vbo))
        self._generate_bodies()

    def cleanup(self):
        self.body_positions_vbo.delete()
        self.body_velocities_vbo.delete()
        self.body_colors_vbo.delete()

    def _gl_pack(self):
        '''Transfer body data to gl buffer'''
        gl.glBindVertexArray(self.vertex_array)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.body_colors_vbo)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.body_positions_vbo)
        
        position = gl.glGetAttribLocation(shader, b'position')
        gl.glEnableVertexAttribArray(position)
        gl.glEnableVertexAttribArray(color_id)
        
        gl.glVertexAttribPointer(position, 4, gl.GL_FLOAT, False, 0, None)
        gl.glVertexAttribPointer(color_id, 4, gl.GL_FLOAT, False, 0, None)
        
        #gl.glBufferData(gl.GL_ARRAY_BUFFER, self.body_count * 4 * 4, self.body_positions, gl.GL_DYNAMIC_DRAW)

        # unbind stuff
        
        #gl.glDisableVertexAttribArray(position)
        #gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

    def _gl_unpack(self):
        position = gl.glGetAttribLocation(shader, b'position')
        gl.glDisableVertexAttribArray(position)
        gl.glDisableVertexAttribArray(color_id)

    def draw(self):
        
        if draw_stars:
            self._gl_pack()
            #gl.glUniform4f(fragment_color, self.color[0], self.color[1], self.color[2], 0.5)
            gl.glBindVertexArray(self.vertex_array)
            gl.glDrawArrays(gl.GL_POINTS, 0, self.body_count)
            gl.glBindVertexArray(0)
            self._gl_unpack()

    def draw_axes2(self):
        if draw_axes:
            gl.glBegin(gl.GL_LINES)
            x, y, z = self.position
            d = 0.5
            u = self.up * d
            r = self.right * d
            f = self.forward * d
            gl.glColor3f(1, 0, 0); gl.glVertex3f(x, y, z); gl.glVertex3f(x + r.x, y + r.y, z + r.z)
            gl.glColor3f(0, 1, 0); gl.glVertex3f(x, y, z); gl.glVertex3f(x + u.x, y + u.y, z + u.z)
            gl.glColor3f(0, 0, 1); gl.glVertex3f(x, y, z); gl.glVertex3f(x + f.x, y + f.y, z + f.z)
            gl.glEnd()

class Universe():

    G = 10
    dt = 0.1

    def __init__(self, galaxies):
        self.galaxies = galaxies
        self.galaxy_count = len(self.galaxies)
        self.body_count = self.galaxies[0].body_count

    def others(self, n):
        for i in range(0, n):
            yield i
        for i in range(n+1, self.galaxy_count):
            yield i

    def cl_init(self):
        for galaxy in self.galaxies:
            galaxy.cl_init()

    def cleanup(self):
        for galaxy in self.galaxies:
            galaxy.cleanup()

    def step(self, delta_time):

        if pause:
            return

        centers = np.ndarray((self.galaxy_count, 4), dtype=np.float32)
        for i in range(self.galaxy_count):
            centers[i][:3] = self.galaxies[i].position
            centers[i][3]  = self.galaxies[i].mass

        centers_buffer = clarray.to_device(cl_queue, centers)

        gl.glFlush()
        gl.glFinish()

        for i, galaxy in enumerate(self.galaxies):
            cl.enqueue_acquire_gl_objects(cl_queue, [galaxy.body_positions_cl_buffer, galaxy.body_velocities_cl_buffer])
            kernel_step(cl_queue,
                        (galaxy.body_count,),
                        None,
                        galaxy.body_positions_cl_buffer,
                        galaxy.body_velocities_cl_buffer,
                        centers_buffer.data,
                        np.uint(galaxy.body_count),
                        np.uint(self.galaxy_count),
                        np.float32(self.dt * delta_time),
                        np.float32(self.G))
            cl.enqueue_release_gl_objects(cl_queue, [galaxy.body_positions_cl_buffer, galaxy.body_velocities_cl_buffer])
            cl_queue.finish()


        centers = [mathutils.Vector((galaxy.position.x,
                                     galaxy.position.y,
                                     galaxy.position.z,
                                     galaxy.mass)) for galaxy in self.galaxies]

        for i in range(self.galaxy_count):
            this_galaxy = self.galaxies[i]
            f = mathutils.Vector((0, 0, 0))
            #f = np.zeros((4,), dtype=np.float32)
            for j in self.others(i):
                delta_pos = mathutils.Vector(centers[j] - centers[i]).xyz
                length = max(1.0, delta_pos.length_squared)
                f += delta_pos.normalized() * self.G * centers[i][3] * centers[j][3] / delta_pos.length_squared
            this_galaxy.velocity += f * delta_time * self.dt
            this_galaxy.position += this_galaxy.velocity * delta_time * self.dt

    def draw(self):
        #gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        modelview_matrix  = gl.glGetDoublev(gl.GL_MODELVIEW_MATRIX)
        projection_matrix = gl.glGetDoublev(gl.GL_PROJECTION_MATRIX)
        gl.glUseProgram(shader)
        gl.glUniformMatrix4fv(projection_matrix_id, 1, False, projection_matrix)
        gl.glUniformMatrix4fv(modelview_matrix_id, 1, False, modelview_matrix)
        gl.glUniform1f(x_id, camera.position[0])
        gl.glUniform1f(y_id, camera.position[1])
        gl.glUniform1f(z_id, camera.position[2])
        for galaxy in self.galaxies:
            galaxy.draw()
        gl.glUseProgram(0)
        for galaxy in self.galaxies:
            galaxy.draw_axes2()
        

class Camera():
    '''Represents the camera in the scene'''

    OOTP = (0.5 / math.pi) # one over two pi
    LERP_RATE = 12

    def __init__(self, w, r, target=None):
        '''Initialize the camera
            - w:            rotation frequency
            - r:            distance from origin
            - lx, ly, lz:   look towards this position
        '''
        self.frequency = w
        self.r = r
        self.theta = 0
        self.phi = 30
        self.target_r = r
        self.target_theta = 0
        self.target_phi = 30
        self.theta_offset = 0
        self.phi_offset = 0

        self.lookAt = mathutils.Vector((0, 0, 0)) if not target else target
        self.up = Transform.UP

        self.rotate = True

    def step(self, delta_time):
        if auto_rotate:
            self.target_theta += delta_time * self.frequency * 2.0 * math.pi

        self.r      = lerp(self.r, self.target_r, self.LERP_RATE * delta_time)
        self.theta  = lerp(self.theta, self.target_theta, self.LERP_RATE * delta_time)
        self.phi    = lerp(self.phi, self.target_phi, self.LERP_RATE * delta_time)

    @property 
    def x(self): return self.r * math.sin(self.theta + self.theta_offset) * math.sin(self.phi + self.phi_offset)

    @property
    def y(self):
        return self.r * math.cos(self.phi + self.phi_offset)

    @property
    def z(self):
        return self.r * math.cos(self.theta + self.theta_offset) * math.sin(self.phi + self.phi_offset)

    @property
    def position(self):
        return [self.x, self.y, self.z]

    def mouse_drag(self, delta):
        self.set_offset(delta)

    def mouse_drag_end(self):
        self.apply_offset()

    def zoom_in(self):
        self.target_r /= 1.1

    def zoom_out(self):
        self.target_r *= 1.1

    def set_offset(self, delta):
        self.theta_offset   = -0.05 * float(delta[0]) * self.OOTP
        self.phi_offset     = 0.02 * float(delta[1]) * self.OOTP

    def apply_offset(self):
        self.theta          += self.theta_offset
        self.target_theta   += self.theta_offset
        self.phi            += self.phi_offset
        self.target_phi     += self.phi_offset
        self.theta_offset    = 0
        self.phi_offset      = 0


class Controller():
    '''
    The Controller maintains the simulation state and handles drawing
    operations and input.
    '''
    COLORS = [
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 1, 1),
        (0, 0, 1),
        (1, 0, 1),
    ]

    def __init__(self, w, h, n):
        self.width = w
        self.height = h
        self.galaxy_count = n
        self.universe = None

        # timing variables
        self.last_time      = 0
        self.compute_time   = 0
        self.draw_time      = 0
        self.num_steps      = 0
        self.num_frames     = 0

        # rendering 
        self.mouse_drag_origin = np.array([0, 0], dtype=np.int32)
        self.shader = None
  
        self._init()


    def _init(self):
        global shader, x_id, y_id, z_id
        global projection_matrix_id, modelview_matrix_id
        global color_id

        # shader setup
        shader = gl.shaders.compileProgram(
            shaders.compileShader(vertex_shader, gl.GL_VERTEX_SHADER),
            shaders.compileShader(fragment_shader, gl.GL_FRAGMENT_SHADER),
        )
        gl.glLinkProgram(shader)

        # get memory offsets for shader variables
        color_id                = gl.glGetAttribLocation(shader, b'color')
        x_id                    = gl.glGetUniformLocation(shader, b'x')
        y_id                    = gl.glGetUniformLocation(shader, b'y')
        z_id                    = gl.glGetUniformLocation(shader, b'z')
        modelview_matrix_id     = gl.glGetUniformLocation(shader, b'modelview_matrix')
        projection_matrix_id    = gl.glGetUniformLocation(shader, b'projection_matrix')
        
            
        # initialize the scene
        galaxies = []
        for i in range(3):
            pos = Transform.random_vector() * 5
            #pos = mathutils.Vector((0, 0, 0))
            galaxies.append(Galaxy(position=pos, mass=1, body_count=-1, color=self.COLORS[i]))

        self.universe = Universe(galaxies)

        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE)
        
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
        #gl.glEnable(gl.GL_DEPTH_TEST)
        #gl.glDepthFunc(gl.GL_LESS)
        #gl.glDepthMask(gl.GL_FALSE)
        #gl.glEnable(gl.GL_DEPTH_CLAMP)
        gl.glEnable(gl.GL_POINT_SMOOTH)
        gl.glEnable(gl.GL_LINE_SMOOTH)
        #gl.glPointSize(2.0)
        gl.glLineWidth(1.0)

        self.last_time = time.time()

    def cl_init(self):
        self.universe.cl_init()

    def quit(self):
        self.universe.cleanup()
        draw_time = self.draw_time / max(1, self.num_frames)
        comp_time = self.compute_time / max(1, self.num_steps)
        print('Average draw time:  {:.2f} ms'.format(1000 * draw_time))
        print('Average frame time: {:.2f} ms'.format(1000 * comp_time))
        print('Average fps:        {:.2f} fps'.format(1.0 / (draw_time + comp_time)))

    def mouse_drag_begin(self, location):
        self.mouse_drag_origin = location

    def compute(self):
        t0 = time.time()

        delta_time = time.time() - self.last_time
        self.last_time += delta_time

        camera.step(delta_time)
        self.universe.step(delta_time)

        self.num_steps += 1
        self.compute_time += time.time() - t0


    def draw(self):
        t0 = time.time()

        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        glu.gluPerspective(30, 1, 1, 20)
        x, y, z = camera.position
        glu.gluLookAt(x, y, z, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)

        if draw_grid:
            grid_width = 10
            gl.glBegin(gl.GL_LINES)
            for i in range(-grid_width, grid_width + 1):
                gl.glColor4f(1, 1, 1, 0.25 - 0.25 *(i / 10) ** 2)
                gl.glVertex3f(i, 0, -grid_width); gl.glVertex3f(i, 0, grid_width)
            for i in range(-grid_width, grid_width + 1):
                gl.glColor4f(1, 1, 1, 0.25 - 0.25 *(i / 10) ** 2)
                gl.glVertex3f(-grid_width, 0, i); gl.glVertex3f(grid_width, 0, i)
            gl.glEnd()

        if draw_axes:
            gl.glBegin(gl.GL_LINES)
            gl.glColor4f(1, 0, 0, 0.2); gl.glVertex3f(0, 0, 0); gl.glVertex3f(10, 0, 0)
            gl.glColor4f(0, 1, 0, 0.2); gl.glVertex3f(0, 0, 0); gl.glVertex3f(0, 10, 0)
            gl.glColor4f(0, 0, 1, 0.2); gl.glVertex3f(0, 0, 0); gl.glVertex3f(0, 0, 10)
            gl.glEnd()

        self.universe.draw()

        gl.glFlush()

        self.num_frames += 1
        self.draw_time += time.time() - t0

def initialize_scene():
    return
    gl.glClearColor(0, 1, 1, 1)
    gl.glColor3f(1, 1, 1)

    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE)
    #gl.glEnable(gl.GL_DEPTH_CLAMP)

t0 = time.time()
def idle_cb():
    global controller
    global t0

    controller.compute()
    glut.glutPostRedisplay()


def display_cb():
    global controller
    controller.draw()
    
def reshape_cb(width, height):
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    gl.glViewport(0, 0, width, height)
    ar = width / height
    gl.glOrtho(-2 * ar, 2 * ar, -2, 2, -10, 10)

def keyboard_cb(key, x, y):
    global controller, draw_axes, draw_grid, draw_stars, auto_rotate, pause

    if key == b'q':
        controller.quit()
        exit()
    elif key == b'f':
        glut.glutFullScreen()
    elif key == b'r':
        auto_rotate = not auto_rotate
    elif key == b'd':
        draw_axes = not draw_axes
    elif key == b'g':
        draw_grid = not draw_grid
    elif key == b's':
        draw_stars = not draw_stars
    elif key == b'p':
        pause = not pause
    pass

# mouse button state, True means pressed
mouse_buttons = [False, False, False]
click_location = np.array([-1, -1], dtype=np.uint32)
def mouse_cb(button, state, x, y):
    global controller, mouse_buttons, click_location

    # button clicked
    if button >= 0 and button <= 2:
        if state == 0:
            # mouse was clicked
            mouse_buttons[button] = True
            click_location[0] = x
            click_location[1] = y
        else:
            # mouse was released
            mouse_buttons[button] = False
            if button == MOUSE_BUTTON_RIGHT:
                camera.mouse_drag_end()
    else:
        # wheel scrolled
        if state:
            return
        if button == MOUSE_BUTTON_WHEEL_UP:
            camera.zoom_in()
        elif button == MOUSE_BUTTON_WHEEL_DOWN:
            camera.zoom_out()

def mouse_motion_cb(x, y):
    global click_location
    drag = np.array([x, y], dtype=np.int32)
    if mouse_buttons[2]:
        camera.mouse_drag(drag - click_location)
    else:
        pass

if __name__ == "__main__":

    # Initialize GLUT library
    glut.glutInit()

    # Setup window
    screen_width = glut.glutGet(glut.GLUT_SCREEN_WIDTH)
    screen_height = glut.glutGet(glut.GLUT_SCREEN_HEIGHT)

    glut.glutInitWindowSize(screen_width // 2, screen_height // 2)
    glut.glutInitWindowPosition(0, 0)
    glut.glutInitDisplayMode(glut.GLUT_SINGLE | glut.GLUT_RGB)
    glut.glutCreateWindow(b'glGalaxy')

    # Register event callbacks
    glut.glutIdleFunc(idle_cb)
    glut.glutDisplayFunc(display_cb)
    glut.glutKeyboardFunc(keyboard_cb)
    glut.glutReshapeFunc(reshape_cb)
    glut.glutMouseFunc(mouse_cb)
    glut.glutMotionFunc(mouse_motion_cb)

    # must create VBOs before initializing openCL context???

    # Initialize simulation controller
    controller = Controller(screen_width, screen_height, 0)
    camera = Camera(0.05, 10)

    # Initialize OpenCL
    cl_platform = cl.get_platforms()[0]
    cl_context  = cl.Context(
        properties=[(cl.context_properties.PLATFORM, cl_platform)] + get_gl_sharing_context_properties())
    cl_queue    = cl.CommandQueue(cl_context)
    cl_program  = cl.Program(cl_context, open('gravity.c', 'r').read()).build()
    kernel_step = cl_program.gravity
    kernel_init = cl_program.initialize

    controller.cl_init()

    
    # Begin main loop
    initialize_scene()
    glut.glutMainLoop()

    #glutFullScreen()
