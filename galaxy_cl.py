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
controller = None
screen_width = 640
screen_height = 480
camera_position = [1, 1, 1]

shader = None
x_id = None
y_id = None
z_id = None

draw_axes = True
draw_grid = True
pause     = True

MOUSE_BUTTON_LEFT = 0
MOUSE_BUTTON_MIDDLE = 1
MOUSE_BUTTON_RIGHT = 2
MOUSE_BUTTON_WHEEL_UP = 3
MOUSE_BUTTON_WHEEL_DOWN = 4


# Shaders
vertex_shader = open("vertex.c", "r").read()
fragment_shader = open("fragment.c", "r").read()
 

class Galaxy():
    '''
    A class representing a galaxy, consisting of a center of mass and
    a collection of orbiting particles.
    '''

    def __init__(self, position, mass=1, body_count=-1, color=(1, 1, 1), ecc=1.0, camera=None):
        global shader, eye_id
        self.transform = Transform(pos=position, rot=-1)
        self.velocity = mathutils.Vector(-Universe.dt * self.position + (Universe.dt / 3) * Transform.random_vector())
        self.mass = mass
        self.body_count = body_count
        self.color = color
        self.ecc = ecc
        self.camera=camera

        self.body_positions = None
        self.body_positions_vbo = None
        self.body_positions_cl_buffer = None
        self.body_velocities = None
        self.body_velocities_vbo = None
        self.body_velocities_cl_buffer = None

        if self.body_count < 0:
            self.body_count = 1000000

        self.vertex_array = None
        self.vertex_buffer = None

        self._init()

    @property
    def position(self): return self.transform.position
    @position.setter
    def position(self, new_position): self.transform.position = new_position

    # Orientation Vectors
    @property
    def up(self):       return self.transform.up
    @property 
    def forward(self):  return self.transform.forward
    @property
    def right(self):    return self.transform.right

    def _init(self):
        self._generate_bodies()
        self._gl_setup()

    def _generate_bodies(self):
        self.body_positions = np.ndarray((self.body_count, 4), dtype=np.float32)
        self.body_velocities = np.ndarray((self.body_count, 4), dtype=np.float32)

        r1 = 2
        r2 = 10
        sigma = 1

        mu = (r1 + r2) / 2
        
        for i in range(self.body_count):

            # radius is normally distributed around mu
            # phase is uniformly distributed 
            theta = random.random() * 2 * math.pi
            r = random.normalvariate(mu, sigma)

            #height = 0.1 * (random.random() * 2 - 1) * math.exp(-((r - mu) ** 2.0) / (2.0 * sigma ** 2.0))
            height = 0
            body_position = self.position + self.transform.rot * \
                mathutils.Vector((r * math.sin(theta), height, r * math.cos(theta)))
            #body_position = self.transform.rot * body_position
            #body_position += self.position

            self.body_positions[i][0] = body_position.x
            self.body_positions[i][1] = body_position.y
            self.body_positions[i][2] = body_position.z
            self.body_positions[i][3] = 1
  
            # position relative to galaxy center
            delta = body_position - self.position

            # tangential orbital velocity        
            velocity = self.up.cross(delta).normalized() * (Universe.G / delta.length) ** 0.5
            #velocity = (Universe.G * self.mass / r) ** 0.5 * velocity
            velocity += self.velocity
            self.body_velocities[i][0] = velocity.x
            self.body_velocities[i][1] = velocity.y
            self.body_velocities[i][2] = velocity.z
            #self.body_velocities[i][3] = 0

        #self.body_positions = np.reshape(self.body_positions, (self.body_count * 3,), order='C')

        # construct vertex buffer objects
        self.body_positions_vbo = vbo.VBO(data=self.body_positions, usage=gl.GL_DYNAMIC_DRAW, target=gl.GL_ARRAY_BUFFER)
        self.body_positions_vbo.bind()
        
        self.body_velocities_vbo = vbo.VBO(data=self.body_velocities, usage=gl.GL_DYNAMIC_DRAW, target=gl.GL_ARRAY_BUFFER)
        self.body_velocities_vbo.bind()

    def cl_init(self):
        self.body_positions_cl_buffer = cl.GLBuffer(cl_context, mf.READ_WRITE, int(self.body_positions_vbo))
        self.body_velocities_cl_buffer = cl.GLBuffer(cl_context, mf.READ_WRITE, int(self.body_velocities_vbo))

    def _gl_setup(self):
        '''Generate buffers'''
        self.vertex_array = gl.glGenVertexArrays(1)
        self.vertex_buffer = gl.glGenBuffers(1)
        #self._gl_pack()

    def cleanup(self):
        self.body_positions_vbo.delete()
        self.body_velocities_vbo.delete()

    def _gl_pack(self):
        '''Transfer body data to gl buffer'''
        gl.glBindVertexArray(self.vertex_array)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.body_positions_vbo)

        position = gl.glGetAttribLocation(shader, b'position')
        gl.glEnableVertexAttribArray(position)

        gl.glVertexAttribPointer(position, 4, gl.GL_FLOAT, False, 0, None)

        #gl.glBufferData(gl.GL_ARRAY_BUFFER, self.body_count * 4 * 4, self.body_positions, gl.GL_DYNAMIC_DRAW)

        
        # unbind stuff
        gl.glBindVertexArray(0)
        #gl.glDisableVertexAttribArray(position)
        #gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

    def draw(self):
        global shader, eye_id
        r, g, b = self.color
        gl.glColor4f(r, g, b, 0.25)
        
        
        """gl.glBegin(gl.GL_POINTS)
        for i in range(self.body_count):
            x, y, z, w = self.body_positions[i]
            #gl.glVertex3f(x, 0, z)
            gl.glVertex3f(x, y, z)
        gl.glEnd()"""
    
        self._gl_pack()

        gl.glUseProgram(shader)

        #position = gl.glGetAttribLocation(shader, b'position')
        #gl.glEnableVertexAttribArray(position)

        perspective_matrix_id = gl.glGetUniformLocation(shader, b'perspective_matrix')
        gl.glUniformMatrix4fv(perspective_matrix_id,
                              1,
                              False,
                              gl.glGetDoublev(gl.GL_PROJECTION_MATRIX))

        modelview_matrix_id = gl.glGetUniformLocation(shader, b'modelview_matrix')
        gl.glUniformMatrix4fv(modelview_matrix_id,
                              1,
                              False,
                              gl.glGetDoublev(gl.GL_MODELVIEW_MATRIX))

        gl.glUniform1f(x_id, self.camera.position[0])
        gl.glUniform1f(y_id, self.camera.position[1])
        gl.glUniform1f(z_id, self.camera.position[2])

        gl.glBindVertexArray(self.vertex_array)
        gl.glDrawArrays(gl.GL_POINTS, 0, self.body_count)
        gl.glUseProgram(0)

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

    G = 100
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
            cl_kernel(cl_queue,
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
                                     0)) for galaxy in self.galaxies]

        for i in range(self.galaxy_count):
            this_galaxy = self.galaxies[i]
            f = mathutils.Vector((0, 0, 0))
            #f = np.zeros((4,), dtype=np.float32)
            for j in self.others(i):
                delta_pos = mathutils.Vector(centers[j] - centers[i]).xyz
                f += delta_pos.normalized() * self.G / delta_pos.length_squared
            this_galaxy.velocity += f * delta_time * self.dt
            this_galaxy.position += this_galaxy.velocity * delta_time * self.dt

          
            """for index in range(this_galaxy.body_count):
                f = mathutils.Vector((0, 0, 0, 0))

                # update position
                this_galaxy.body_positions[index] += this_galaxy.body_velocities[index] * delta_time * self.dt
                pos = mathutils.Vector(this_galaxy.body_positions[index])

                for k in range(self.galaxy_count):
                    delta_pos = (centers[k] - pos)
                    f += delta_pos.normalized() * self.G / delta_pos.length_squared
                    #f += (delta_pos / np.linalg.norm(delta_pos)) * self.G / (np.linalg.norm(delta_pos) ** 2)
                                
                # update velocity
                this_galaxy.body_velocities[index] += f * delta_time * self.dt"""

    def draw(self):
        for galaxy in self.galaxies:
            galaxy.draw()

class Camera():
    '''Represents the camera in the scene'''

    OOTP = (0.5 / math.pi) # one over two pi

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
        self.theta_offset = 0
        self.phi_offset = 0
        self.lookAt = mathutils.Vector((0, 0, 0)) if not target else target
        self.up = Transform.UP
        self.rotate = True

    def step(self, delta_time):
        if self.rotate:
            self.theta += delta_time * self.frequency * 2.0 * math.pi

    @property
    def x(self):
        return self.r * math.sin(self.theta + self.theta_offset) * math.sin(self.phi + self.phi_offset)

    @property
    def y(self):
        return self.r * math.cos(self.phi + self.phi_offset)

    @property
    def z(self):
        return self.r * math.cos(self.theta + self.theta_offset) * math.sin(self.phi + self.phi_offset)

    @property
    def position(self):
        return [self.x, self.y, self.z]

    def toggle_rotation(self):
        self.rotate = not self.rotate

    def set_offset(self, delta):
        self.theta_offset = -0.05 * float(delta[0]) * self.OOTP
        self.phi_offset = 0.02 * float(delta[1]) * self.OOTP

    def apply_offset(self):
        self.theta += self.theta_offset
        self.phi += self.phi_offset
        self.theta_offset = 0
        self.phi_offset = 0


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

        self.last_time = 0
        self.compute_time = 0
        self.draw_time = 0
        self.num_steps = 0
        self.num_frames = 0

        self.camera = None
        self.mouse_drag_origin = np.array([0, 0], dtype=np.int32)

        self.universe = None

        self.shader = None

        self._init()


    def _init(self):
        global shader, x_id, y_id, z_id
        shader = gl.shaders.compileProgram(
            shaders.compileShader(vertex_shader, gl.GL_VERTEX_SHADER),
            shaders.compileShader(fragment_shader, gl.GL_FRAGMENT_SHADER),
        )
        gl.glLinkProgram(shader)
        x_id = gl.glGetUniformLocation(shader, b'x')
        y_id = gl.glGetUniformLocation(shader, b'y')
        z_id = gl.glGetUniformLocation(shader, b'z')

        self.camera = Camera(0.1, 10)
        galaxies = []
        for i in range(1):
            #pos = Transform.random_vector() * 5
            pos = mathutils.Vector((0, 0, 0))
            galaxies.append(Galaxy(position=pos, mass=1, body_count=-1, color=self.COLORS[i], camera=self.camera))

        self.universe = Universe(galaxies)
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

    def mouse_drag(self, delta):
        self.camera.set_offset(delta)

    def mouse_drag_end(self):
        self.camera.apply_offset()

    def zoom_in(self):
        self.camera.r /= 1.1

    def zoom_out(self):
        self.camera.r *= 1.1

    def compute(self):
        t0 = time.time()

        delta_time = time.time() - self.last_time
        self.last_time += delta_time
        self.camera.step(delta_time)
        self.universe.step(delta_time)

        self.num_steps += 1
        self.compute_time += time.time() - t0
        pass

    def draw(self):
        t0 = time.time()

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        
        glu.gluPerspective(30, 1, 1, 100)

        x, y, z = self.camera.position
        glu.gluLookAt(x, y, z, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)

        gl.glColor3f(1, 1, 1)

        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE)
        gl.glEnable(gl.GL_DEPTH_CLAMP)
        gl.glEnable(gl.GL_POINT_SMOOTH)
        gl.glEnable(gl.GL_LINE_SMOOTH)
        gl.glPointSize(2.0)
        gl.glLineWidth(2.0)

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

    def toggle_camera_rotation(self):
        self.camera.toggle_rotation()

def initialize_scene():
    return
    gl.glClearColor(0, 0, 0, 1)
    gl.glColor3f(1, 1, 1)

    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE)
    gl.glEnable(gl.GL_DEPTH_CLAMP)

    #gl.glMatrixMode(gl.GL_PROJECTION)
    #gl.glLoadIdentity()
    #gl.glOrtho(-2.0, 2.0, -2.0, 2.0, 1, 20000.0)
    #gl.glMatrixMode(gl.GL_MODELVIEW)
    #gl.glLoadIdentity()
    #glu.gluLookAt(1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)

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

    #glu.gluPerspective(45, width / height, 1, 100)

def keyboard_cb(key, x, y):
    global controller, draw_axes, draw_grid, pause

    if key == b'q':
        controller.quit()
        exit()
    elif key == b'f':
        glut.glutFullScreen()
    elif key == b'r':
        controller.toggle_camera_rotation()
    elif key == b'd':
        draw_axes = not draw_axes
    elif key == b'g':
        draw_grid = not draw_grid
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

            if button == MOUSE_BUTTON_RIGHT:
                controller.mouse_drag_begin(click_location)
        else:
            # mouse was released
            mouse_buttons[button] = False

            if button == MOUSE_BUTTON_RIGHT:
                controller.mouse_drag_end()
    else:
        # wheel scrolled
        if state:
            return
        if button == MOUSE_BUTTON_WHEEL_UP:
            controller.zoom_in()
        elif button == MOUSE_BUTTON_WHEEL_DOWN:
            controller.zoom_out()

def mouse_motion_cb(x, y):
    global controller, click_location
    drag = np.array([x, y], dtype=np.int32)
    if mouse_buttons[2]:
        #print(drag - click_location)
        controller.mouse_drag(drag - click_location)
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

    # Initialize OpenCL
    cl_platform = cl.get_platforms()[0]
    cl_context  = cl.Context(
        properties=[(cl.context_properties.PLATFORM, cl_platform)] + get_gl_sharing_context_properties())
    cl_queue    = cl.CommandQueue(cl_context)
    cl_program  = cl.Program(cl_context, open('gravity.c', 'r').read()).build()
    cl_kernel   = cl_program.gravity

    controller.cl_init()

    
    # Begin main loop
    initialize_scene()
    glut.glutMainLoop()

    #glutFullScreen()
