import math

#COLOR CLASSES
class Radiance:
    def __init__(self, color, lum):
        self.color = color
        self.luminance = lum

    def __mul__(self, scalar):
        return Radiance(self.color*scalar, self.luminance*scalar)

    def __add__(self, other):
        if isinstance(other.color, Color_Alpha):
            other_color = other.color.color
            r = other_color.r
            g = other_color.g
            b = other_color.b
            return Radiance(Color_Alpha(self.color.r+r, self.color.g+g, self.color.b+b, other.color.a), self.luminance + other.luminance)
        else:
            return Radiance(self.color+other.color, self.luminance+other.luminance)


class Color:
    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b

    def multiply(self, other):
        return Color(self.r * other, self.g * other, self.b * other)

    def __mul__(self, other):
        return Color(self.r * other, self.g * other, self.b * other)

    def __add__(self, other):
        return Color(self.r + other.r, self.g + other.g, self.b + other.b)

    def __neg__(self):
        return Color(-self.r, -self.g, -self.b)

class Color_Alpha:
    def __init__(self, r, g, b, a):
        self.color = Color(r, g, b)
        self.a = a

    def __mul__(self, fac):
        scaled_color = self.color*fac
        return Color_Alpha(scaled_color.r, scaled_color.g, scaled_color.b, self.a*fac)

    def __add__(self, other):
        return Color_Alpha(self.color.r + other.r, self.color.g + other.g, self.color.b + other.b, self.a)

class HDRImage:
    def __init__(self, im_width, im_height):
        self.hdr = []  # np.zeros((num_cells, self.im_height, self.im_width, 3))
        # the image are stored as Color 2D arrays
        for j in range(im_height):
            hdr_row = []
            for k in range(im_width):
                hdr_row.append(Radiance(Color(0, 0, 0), 777))
            self.hdr.append(hdr_row)

# Object Classes

class Triangle:
    def __init__(self, point_1, point_2, point_3, material):
        self.v0 = point_1
        self.v1 = point_2
        self.v2 = point_3
        self.material = material

    def normal_at(self, point_of_interest):
        edge1 = (self.v1 - self.v0).to_vector()
        edge2 = (self.v2 - self.v1).to_vector()

        perp_vec = cross(edge1, edge2)
        normal_vec = perp_vec.normalize()
        return normal_vec

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        else:
            return self.v0 == other.v0 and self.v1 == other.v1 and self.v2 == other.v2

    def intersection(self, ray):
        edge1 = self.v1 - self.v0
        edge2 = self.v2 - self.v0

        EPSILON = .00001

        h = cross(ray.direction, edge2.to_vector())
        a = dot(edge1.to_vector(), h)

        if -EPSILON < a < EPSILON:
            return False, -1
        f = 1/a
        s = ray.origin - self.v0
        u = f * dot(s.to_vector(), h)
        if u < 0 or u > 1:
            return False, -1
        q = cross(s.to_vector(), edge1.to_vector())
        v = f * dot(ray.direction, q)
        if v < 0 or u + v > 1:
            return False, -1
        t = f * dot(edge2.to_vector(), q)
        if t > 0:
            return True, t
        else:
            return False, -1


class Light:
    def __init__(self, pos, illumination, light_color=Color(1,1,1)):
        self.pos = pos
        self.illumination = illumination
        self.color = light_color

    def get_lums_from_point(self, sample_point):
        return self.illumination / pow(dist(sample_point, self.pos), 2)


class SpotLight:
    def __init__(self, pos, illumination, light_color, direction, angle):
        self.pos = pos
        self.illumination = illumination
        self.color = light_color
        self.direction = direction
        self.angle = angle

    def get_lums_from_point(self, sample_point):
        light_to_point_vec = vec_from_p_to_p(self.pos, sample_point)

        point_angle = dot(light_to_point_vec.normalize(), self.direction.normalize())
        if point_angle > self.angle:
            # print("spotlight casting")
            return self.illumination / pow(dist(sample_point, self.pos), 2)
        else:
            return 0

def angle(vec1, vec2):
    dp = dot(vec1, vec2)
    return math.acos(dp/(vec1.magnitude() * vec2.magnitude()))

def vec_from_p_to_p(p1, p2):
    # returns vec from p1 to p2
    return (p2 - p1).to_vector()


def dist(point_1, point_2):
    if point_1 is None:
        print("A")
    if point_2 is None:
        print("BBB")
    return math.sqrt(
        math.pow(point_1.x - point_2.x, 2) + math.pow(point_1.y - point_2.y, 2) + math.pow(point_1.z - point_2.z,
                                                                                               2))


class AxisPlane:
    def __init__(self, point_1, point_2, material):
        self.p1 = point_1
        self.p2 = point_2
        self.p3 = Point(0,0,0)
        self.p4 = Point(0,0,0)
        self.AXIS = ""

        epsilon = .001

        diff_x = abs(self.p1.x - self.p2.x)
        diff_y = abs(self.p1.y - self.p2.y)
        diff_z = abs(self.p1.z - self.p2.z)
        if diff_x < epsilon:
            self.AXIS = "X"
            self.p3 = Point(self.p1.x, self.p1.y, self.p2.z)
            self.p4 = Point(self.p1.x, self.p2.y, self.p1.z)
        elif diff_y < epsilon:
            self.AXIS = "Y"
            self.p3 = Point(self.p1.x, self.p1.y, self.p2.z)
            self.p4 = Point(self.p2.x, self.p1.y, self.p1.z)
        elif diff_z < epsilon:
            self.AXIS = "Z"
            self.p3 = Point(self.p1.x, self.p2.y, self.p1.z)
            self.p4 = Point(self.p2.x, self.p1.y, self.p1.z)
        else:
            print("NO AXIS FOUDN")

        self.t1 = Triangle(self.p1, self.p2, self.p3, material)
        self.t2 = Triangle(self.p1, self.p3, self.p2, material)
        self.t3 = Triangle(self.p1, self.p4, self.p2, material)
        self.t4 = Triangle(self.p1, self.p2, self.p4, material)

        self.triangles = [self.t1, self.t2, self.t3, self.t4]

    def normal_at(self, point_of_interest):
        return None

    def intersection(self, ray):
        for triangle in self.triangles:
            does_intersect, t_of_intersect = triangle.intersection(ray)
            if does_intersect:
                return True, t_of_intersect
        return False, -1


class AxisBox:
    def __init__(self, point_1, point_2, material):
        self.p1 = point_1
        self.p2 = point_2
        self.p3 = Point(self.p1.x, self.p1.y, self.p2.z)
        self.p4 = Point(self.p1.x, self.p2.y, self.p2.z)
        self.p5 = Point(self.p1.x, self.p2.y, self.p1.z)
        self.p6 = Point(self.p2.x, self.p2.y, self.p1.z)
        self.p7 = Point(self.p2.x, self.p1.y, self.p1.z)
        self.p8 = Point(self.p2.x, self.p1.y, self.p2.z)

        self.plane1 = AxisPlane(self.p5, self.p3, material)
        self.plane2 = AxisPlane(self.p5, self.p2, material)
        self.plane3 = AxisPlane(self.p5, self.p7, material)
        self.plane4 = AxisPlane(self.p8, self.p4, material)
        self.plane5 = AxisPlane(self.p1, self.p8, material)
        self.plane6 = AxisPlane(self.p8, self.p6, material)

        self.planes = [self.plane1, self.plane2, self.plane3, self.plane4, self.plane5, self.plane6]

        self.material = material

    def normal_at(self, point_of_interest):
        return None

    def intersection(self, ray):
        min_t = math.inf
        num_intersections = 0
        for plane in self.planes:
            intersect_bool, intersect_t = plane.intersection(ray)
            if intersect_bool:
                num_intersections += 1
                if 0 < intersect_t < min_t:
                    min_t = intersect_t
        if num_intersections == 1 or num_intersections == 2:
            return True, min_t
        return False, -1

class Sphere:
    def __init__(self, position, radius, material):
        self.position = position
        self.radius = radius
        self.material = material

    def intersection(self, ray):
        o = ray.origin
        d = ray.direction

        if o is None:
            return False, -1

        c = self.position

        a = math.pow(d.dx, 2) + math.pow(d.dy, 2) + math.pow(d.dz, 2)
        b = 2 * (d.dx * (o.x - c.x) + d.dy * (o.y - c.y) + d.dz * (o.z - c.z))
        c = math.pow(o.x, 2) + math.pow(o.y, 2) + math.pow(o.z, 2) - math.pow(self.radius, 2) - 2*(c.x*o.x + c.y*o.y + c.z*o.z) + math.pow(c.x, 2) + math.pow(c.y, 2) + math.pow(c.z, 2)

        quad_hit, t0, t1 = quadratic(a, b, c)
        if not quad_hit:
            return False, -1
        # check quadratic shape t0 and t1 for nearest intersection
        if t1 <= 0 or t0 <= 0:
            return False, -1
        t_shape_hit = t0

        return True, t_shape_hit

    def normal_at(self, point_of_interest):
        normal_vec = (point_of_interest - self.position).to_vector().normalize()
        return normal_vec

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        else:
            does_equal = (self.position - other.position).to_vector().magnitude() < .01
            return does_equal

#vector stuff


class Vector:
    def __init__(self, dx, dy, dz):
        self.dx = dx
        self.dy = dy
        self.dz = dz

    def magnitude(self):
        return math.sqrt(math.pow(self.dx, 2) + math.pow(self.dy, 2) + math.pow(self.dz, 2))

    def normalize(self):
        mag = self.magnitude()
        if not mag == 0:
            return Vector(self.dx/mag, self.dy/mag, self.dz/mag)
        else:
            return Vector(0, 0, 0)

    def multiply(self, scalar):
        return Vector(self.dx * scalar, self.dy * scalar, self.dz * scalar)

    def add(self, other_vec):
        return Vector(self.dx + other_vec.dx, self.dy + other_vec.dy, self.dz + other_vec.dz)

    def __neg__(self):
        return Vector(-self.dx, -self.dy, -self.dz)

    def to_point(self):
        return Point(self.dx, self.dy, self.dz)


def dot(vec1, vec2):
    return vec1.dx * vec2.dx + vec1.dy * vec2.dy + vec1.dz * vec2.dz


def proj(vec1, vec2):
    v_norm = vec1.normalize()
    # project vec1 on vec2
    return (dot(vec1, vec2) / v_norm ** 2) * vec1


def rotate_point(point, theta, axis):
    rotated_point = Point(point.x, point.y, point.z)
    if axis == "Z":
        rotated_point.x = point.y * math.sin(theta)
        rotated_point.y = point.y * math.cos(theta)
    return rotated_point


def cross(vec1, vec2):
    return Vector((vec1.dy * vec2.dz - vec1.dz * vec2.dy), -(vec1.dx * vec2.dz - vec1.dz * vec2.dx),
                  (vec1.dx * vec2.dy - vec1.dy * vec2.dx))

    # def rotate(self, axis, direction):
    #     if axis == 0:
    #         self.dx +=


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    def pos_at_t(self, t):
        t_pos = Point(self.direction.dx*t, self.direction.dy*t, self.direction.dz*t)
        return self.origin.add_point(t_pos)

    # def rotate(self, axis, angle):
    #     self.direction.rotate

class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def add_point(self, other_point):
        return Point(self.x + other_point.x, self.y + other_point.y, self.z + other_point.z)

    def to_vector(self):
        return Vector(self.x, self.y, self.z)

    def __add__(self, other_point):
        return Point(self.x + other_point.x, self.y + other_point.y, self.z + other_point.z)

    def __neg__(self):
        return Point(-self.x, -self.y, -self.z)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        return self + (-other)

    def __str__(self):
        return str(self.x)+", "+str(self.y)+", "+str(self.z)

def normalize(val, sigma, offset):
    return math.atan((val - offset) / sigma) / math.pi + .5

def quadratic(a, b, c):
    # Find quadratic discriminant
    discrim = b * b - 4 * a * c
    if discrim < 0:
        return False, -1, -1
    root_discrim = math.sqrt(discrim)

    # Compute quadratic t values
    q = 0
    if b < 0:
        q = -.5 * (b - root_discrim)
    else:
        q = -.5 * (b + root_discrim)
    t0 = q / a
    t1 = c / q
    if t0 > t1:
        return True, t1, t0
    return True, t0, t1


def reflect(reflector, normal):
    reflected = normal.multiply(dot(reflector, normal)).multiply(2).add(-reflector)
    return reflected
