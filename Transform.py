import math
import random

import mathutils
import numpy as np

class Transform():

    RIGHT   = mathutils.Vector((1, 0, 0))
    UP      = mathutils.Vector((0, 1, 0))
    FORWARD = mathutils.Vector((0, 0, 1))

    def __init__(self, pos=None, rot=None, scale=None):
        self._pos = None
        self._rot = None
        self.scale = None

        if not pos:
            self._pos = mathutils.Vector((0, 0, 0))
        else:
            self._pos = mathutils.Vector(pos)

        if not rot:
            self._rot = mathutils.Quaternion()
        elif rot == -1:
            theta = math.acos(2.0 * random.random() - 1.0)
            phi = 2.0 * math.pi * random.random()

            x = math.cos(theta) * math.cos(phi)
            y = math.sin(theta) * math.cos(phi)
            z = math.sin(phi)

            psi = random.random() * 2.0 * math.pi
            self._rot = mathutils.Quaternion((x, y, z), psi)
        else:
            self._rot = mathutils.Quaternion(rot)

        if not scale:
            self.scale = mathutils.Vector((0, 0, 0))
        else:
            self.scale = mathutils.Vector(scale)

    @property
    def position(self):
        return self._pos

    def position4(self):
        return mathutils.Vector((self._pos.x, self._pos.y, self_pos.z, 0))

    @position.setter
    def position(self, new_position):
        self._pos = new_position

    @property
    def rotation(self):
        return self._rot

    @property
    def up(self):
        return self._rot * self.UP

    @property
    def forward(self):
        return self._rot * self.FORWARD

    @property
    def right(self):
        return self._rot * self.RIGHT

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


    