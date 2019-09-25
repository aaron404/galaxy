import math
import random

import mathutils

class Transform():

    RIGHT   = mathutils.Vector((1, 0, 0))
    UP      = mathutils.Vector((0, 1, 0))
    FORWARD = mathutils.Vector((0, 0, 1))

    def __init__(self, pos=None, rot=None, scale=None):
        self._pos = None
        self.rot = None
        self.scale = None

        if not pos:
            self._pos = mathutils.Vector((0, 0, 0))
        else:
            self._pos = mathutils.Vector(pos)

        if not rot:
            self.rot = mathutils.Quaternion()
        elif rot == -1:
            theta = math.acos(2.0 * random.random() - 1.0)
            phi = 2.0 * math.pi * random.random()

            x = math.cos(theta) * math.cos(phi)
            y = math.sin(theta) * math.cos(phi)
            z = math.sin(phi)

            psi = random.random() * 2.0 * math.pi
            self.rot = mathutils.Quaternion((x, y, z), psi)
        else:
            self.rot = mathutils.Quaternion(rot)

        if not scale:
            self.scale = mathutils.Vector((0, 0, 0))
        else:
            self.scale = mathutils.Vector(scale)

    @property
    def position(self):
        return self._pos

    @position.setter
    def position(self, new_position):
        self._pos = new_position

    @property
    def up(self):
        return self.rot * self.UP

    @property
    def forward(self):
        return self.rot * self.FORWARD

    @property
    def right(self):
        return self.rot * self.RIGHT

    @staticmethod
    def random_vector():
        theta = math.acos(2.0 * random.random() - 1.0)
        phi = 2.0 * math.pi * random.random()

        x = math.cos(theta) * math.cos(phi)
        y = math.sin(theta) * math.cos(phi)
        z = math.sin(phi)

        return mathutils.Vector((x, y, z))

    def translate(self, delta):
        self.position += delta

    def rotate_around(self, axis, angle):
        pass


    