import importlib
import cv2
import numpy as np
import math
import noise
import random
from perlin_noise import PerlinNoise
import threading
import sys
from raytracer_classes import *

TURNING = False

REFLECTION_LIMIT = 3
NUMBER_OF_CELLS = 2

MAX_CLOUD_DENSITY = 1

RESOLUTION = 100
RATIO = 1
UPSIZE = 2

CLOUD_FALLOFF = .9
ALPHA_DENSITY = .12
# 1 - solid object
# .2 - solid sturdy cloud
# .15 - whispy
# .1 - effite, almost invisible
# .05 - a spring london afternoon fog

LIGHTING_DENSITY = .1
# .2
# .15 - a cloud of moderate spookiness
# .1 - dark, spooky storm cloud
# .05 - dark, light doesn't hit

CLOUD_SHADOW_MULTIPLIER = .6

CLOUD_AMBIENT_SCALE = .6

DENSITY_STEP_SIZE = .1
RADIANCE_STEP_SIZE = .05

# BACKGROUND_RADIANCE = Radiance(Color(.1,.35,.9), 1)
# BACKGROUND_RADIANCE = Radiance(Color(.1,.1,.1), .1) # NIGHT
BACKGROUND_RADIANCE = Radiance(Color(.9,.5,.2), .5) # DAY

WORLD_LIGHTS = list()
# WORLD_LIGHTS.append(SpotLight(Point(1,-2,4), 4, Color(.2,.2,.9), Vector(0,.3,-.6).normalize(), .99))
# WORLD_LIGHTS.append(SpotLight(Point(-1.5,-2,4), 4, Color(.2,.9,.2), Vector(.3,.3,-.8).normalize(), .995))

WORLD_LIGHTS.append(Light(Point(4, -4, 4), 30, Color(.2, .9, .7))) # SUN
# WORLD_LIGHTS.append(Light(Point(-5, -8, 4), 180, Color(.9, .6, .2))) # blue light
# WORLD_LIGHTS.append(Light(Point(0, 0, -1.5), 1, Color(.2, .9, .7))) # SUN
# WORLD_LIGHTS.append(Light(Point(-2, 1.5, -.85), 1, Color(.2, .9, .7))) # SUN

# WORLD_LIGHTS.append(Light(Point(-3, 0, 1), 1.3, Color(.9, .6, 0)))

# WORLD_LIGHTS.append(Light(Point(-4, 2, 8), 10, Color(.2, .9, .9)))

# WORLD_LIGHTS.append(Light(Point(3, -3, 10), 30, Color(.2, .9, .9)))
# WORLD_LIGHTS.append(Light(Point(4, -3, 0), 10, Color(0, 1, 0)))
# WORLD_LIGHTS.append(Light(Point(-4, -3, 0), 10, Color(0, 0, 1)))

# WORLD_LIGHTS.append(Light(Point(3, 3, 2), 5, Color(1, 0, 1))) PURPLE
#
# WORLD_LIGHTS.append(Light(Point(-3, 0, -1), 5, Color(0, 0, 1))) RED
# WORLD_LIGHTS.append(Light(Point(-2, -.2, 10), 30))
# WORLD_LIGHTS.append(Light(Point(2, 4, 5), 10))


def mix_color(color_1, color_2, alpha):
    mix_r = color_1.r + (color_2.r - color_1.r) * alpha
    mix_g = color_1.g + (color_2.g - color_1.g) * alpha
    mix_b = color_1.b + (color_2.b - color_1.b) * alpha
    return Color(mix_r, mix_g, mix_b)


class Camera:
    def __init__(self, position, up, look_at, focal_length, im_width, im_height, upsize):
        self.position = position # a point
        self.up = up             # a vector
        self.look_at = look_at   # a vector
        self.focal_length = focal_length # a scalar
        self.im_width = im_width
        self.im_height = im_height
        self.N = NUMBER_OF_CELLS
        self.upsize = upsize
        num_cells = pow(self.N, 2)
        self.thread_hdrs = []#np.zeros((num_cells, self.im_height, self.im_width, 3))
        # the images are stored as Color 2D arrays
        for i in range(num_cells):
            thread_image = []
            for j in range(self.im_height):
                ims_row = []
                for k in range(self.im_width):
                    ims_row.append(Radiance(Color(0,0,0), 0))
                thread_image.append(ims_row)
            self.thread_hdrs.append(thread_image)

    def get_pixel_radiance(self, world, row, col):
        center_ray = Ray(self.position, self.look_at)

        row_coor = (row - self.im_height / 2) * 100/self.im_height
        col_coor = (col - self.im_width / 2) * 100/RATIO/self.im_width

        camera_vector = center_ray.direction
        horizontal_vector = cross(camera_vector, self.up).normalize()
        vertical_vector = camera_vector.add(horizontal_vector.multiply(col_coor / self.focal_length))
        vision_vector = vertical_vector.add(self.up.multiply(-row_coor / self.focal_length))

        vision_ray = Ray(center_ray.origin, vision_vector)

        return world.get_radiance_from_ray(vision_ray)

    # def snap(self, world):
    #     # image = np.zeros((self.im_height, self.im_width, 3))
    #     image = []
    #
    #     for row in range(self.im_height):
    #         for col in range(self.im_width):
    #             pixel_color = self.get_pixel_color(world, row, col)
    #
    #             image[row, col] = pixel_color
    #     return image

    def cont_render(self, world, upsize):
        sum_image = np.zeros((self.im_height * upsize, self.im_width * upsize, 3))
        component_images = []

        num_cells = pow(self.N, 2)

        threads = list()

        for render_step in range(num_cells):
            print("started thread "+str(render_step))
            # component_image = render_pixel_subgroup(self, world, render_step)
            x = threading.Thread(target=render_pixel_subgroup, args=(self, world, render_step,))
            threads.append(x)
            x.start()

        for index, thread in enumerate(threads):
            thread.join()
            print("joined thread  "+str(index))
            # component_image = np.array(self.thread_images[index])
            np_im = hdr_to_im(self.thread_hdrs[index])
            # cv2.imshow("hi", np_im)
            resized_component = cv2.resize(np_im, (self.im_width * upsize, self.im_height * upsize), interpolation=cv2.INTER_NEAREST)
            sum_image += resized_component

            cv2.imshow("result", sum_image)
            cv2.waitKey(1)
        return sum_image


def hdr_to_im(hdr_array):
    im_height = len(hdr_array)
    im_width = len(hdr_array[0])

    image = np.zeros((im_height, im_width, 3))
    # turns a 2d array of colors into a 3d numpy array
    for row in range(im_height):
        for col in range(im_width):
            radiance = hdr_array[row][col]
            clr = radiance.color
            lum = radiance.luminance

            image[row, col, 0] = clr.r * lum
            image[row, col, 1] = clr.g * lum
            image[row, col, 2] = clr.b * lum
    return image


def render_pixel_subgroup(camera, world, render_step):
    image = np.zeros((camera.im_height, camera.im_width, 3))
    for row in range(int(render_step / camera.N), camera.im_height, camera.N):
        for col in range(render_step % camera.N, camera.im_width, camera.N):
            pixel_radiance = camera.get_pixel_radiance(world, row, col)

            camera.thread_hdrs[render_step][row][col] = pixel_radiance
            # cv2.imshow("result", image)
    return image


class Cloud:
    def __init__(self, world, internal_world, cloud_color=Color(.5,.5,.5)):
        self.world = world
        # self.noise1 = PerlinNoise(octaves=8, seed=random.randrange(0, 50000))
        # self.noise2 = PerlinNoise(octaves=8, seed=random.randrange(0, 50000))
        self.cloud_world = internal_world
        self.cloud_color = cloud_color

    def get_sample(self, object_of_interest, sample_point, view_ray, light):

        if self.cloud_world.inside(sample_point):
            vec_to_view = -view_ray.direction

            lights = self.world.lights
            # return density, radiance

            total_lum = 0

            end_color = Color(0,0,0)

            result_density = LIGHTING_DENSITY

            cloud_line_top = -.72
            cloud_line_bottom = -.8

            # z_pos = sample_point.z
            # if z_pos < cloud_line_top:
            #     result_density *= (1-(z_pos-cloud_line_top))/(cloud_line_top-cloud_line_bottom)
            #     result_density

            # result_density /= abs(z_pos - cloud_line_bottom)^2

            # for light in lights:
            # vec_to_light = (light.pos - sample_point).to_vector()
            vec_to_light = vec_from_p_to_p(sample_point, light.pos)
            ang = angle(vec_to_view, -vec_to_light)

            scatter_scale = 1+1*pow((1+math.cos(ang))/2,2)

            light_pos = light.pos
            # scatter_scale = min(pow(min(dot(vec_to_light, vec_to_view), 0), 2)*.8 + .2, 1)
            # scatter_scale = 1
            total_lum += self.get_frac_transmitted_from_light(object_of_interest, sample_point, light_pos) * light.get_lums_from_point(sample_point)*scatter_scale
            end_color += light.color * (total_lum)
            return result_density, Radiance(end_color, total_lum)
        else:
            return 0, Radiance(Color(0,0,0),0)

    def get_density(self, intersection_point, sample_ray, step_size=DENSITY_STEP_SIZE, mult_fac=1.0):
        total_density = 0
        travel_point = intersection_point
        while self.cloud_world.inside(travel_point):  # step thru the actual dang cloud
            if total_density >= MAX_CLOUD_DENSITY:
                # print("mcd")
                break
            # accumulate density
            if self.cloud_world.inside(travel_point):
                total_density += ALPHA_DENSITY*mult_fac

            travel_point += sample_ray.direction.multiply(step_size).to_point()
        return total_density

    def get_radiance(self, object_of_interest, intersection_point, view_ray, step_size=RADIANCE_STEP_SIZE):
        EPSILON = .01
        point_iterated = intersection_point

        fire_point = point_iterated + view_ray.direction.multiply(EPSILON).to_point()

        backwall_ray = Ray(fire_point, view_ray.direction)
        backwall_object, end_point, _ = self.world.intersects(backwall_ray)

        radiance_from_lights = Radiance(Color(0, 0, 0), 0)
        if end_point is not None and backwall_object == object_of_interest: #you're dealing with an actual volume
            travel_point = intersection_point
            cloud_travel = dist(intersection_point, end_point)

            # print("the distance of CLOUD: "+str(cloud_travel))

            _, cloud_point, _ = self.cloud_world.intersects(view_ray)

            will_hit_object = cloud_point is not None

            is_inside_already = self.cloud_world.inside(intersection_point)

            light_radiances = list()
            if will_hit_object or is_inside_already:
                if not is_inside_already:
                    travel_point = cloud_point

                for light in self.world.lights:
                    total_lums = 0
                    total_density = 0
                    light_color = self.cloud_color
                    sum_color = self.cloud_color

                    sample_radiances = []
                    while dist(travel_point, intersection_point) < cloud_travel: #step thru the actual dang cloud
                        if total_density >= MAX_CLOUD_DENSITY:
                            break
                        # accumulate density and radiance along ray in discrete samples


                        # cloud_sample_stack = list()
                        sample_density, sample_radiance = self.get_sample(object_of_interest, travel_point, view_ray, light)
                        sample_lums = sample_radiance.luminance
                        # cloud_color = sample_radiance.color
                        light_color = sample_radiance.color

                        # cloud_sample_stack.append((sample_density, sample_radiance))

                        total_density += sample_density
                        if sample_density != 0:
                            total_lums = (total_lums * CLOUD_FALLOFF) + (sample_lums * (1-CLOUD_FALLOFF))

                        # total_lums = max(sample_lums, total_lums)
                        # total_lums = max(total_lums, sample_radiance.luminance)
                        travel_point += view_ray.direction.multiply(step_size).to_point()

                    if total_density != 0:
                        light_radiances.append(Radiance(Color_Alpha(light_color.r,light_color.g,light_color.b, total_density), total_lums))
                    else:
                        return Radiance(Color_Alpha(0,0,0,0),0)
            else:
                return Radiance(Color_Alpha(0, 0, 0, 0), 0)
            # =============================================

            # collapse the radiance stack
            # collapsed_radiance = radiance_stack[len(radiance_stack) - 1]

            num_lights = len(light_radiances)
            if num_lights != 0:
                combined_colors = self.cloud_color
                total_lums = 0
                min_alpha = 1
                for radiance_index in range(num_lights):
                    rad_w_alpha = light_radiances[radiance_index]
                    min_alpha = max(rad_w_alpha.color.a, min_alpha)
                    total_lums += rad_w_alpha.luminance
                    radiance_sample_color = rad_w_alpha.color.color

                    combined_colors += rad_w_alpha.color.color
                    rad_lum = rad_w_alpha.luminance

                    # print(rad_lum)

                    # rad_no_alpha = Radiance(Color_Alpha(self.world.lights[radiance_index].color, rad_lum)
                    # radiance_from_lights += rad_no_alpha

                r = combined_colors.r
                g = combined_colors.g
                b = combined_colors.b
                radiance_from_lights = Radiance(Color_Alpha(r,g,b,min_alpha), total_lums/num_lights)

            # if total_lums != 0:
            #     cloud_color = cloud_color + sum_color
        else:
            return Radiance(Color_Alpha(0,0,0,0),0)

        return BACKGROUND_RADIANCE*CLOUD_AMBIENT_SCALE + radiance_from_lights

    def get_frac_transmitted_from_light(self, shadow_object, shadow_point, light_point, stp_size=.1, mult_fac=1.0):
        vector_to_light = vec_from_p_to_p(shadow_point, light_point).normalize()
        ray_density = self.get_density(shadow_point, Ray(shadow_point, vector_to_light), step_size=stp_size, mult_fac=mult_fac)
        fraction_transmitted = (1 - ray_density)

        return max(fraction_transmitted, 0)


class Pitch:
    def __init__(self, world):
        self.world = world

    def get_radiance(self, object_hit, intersection_point, view_ray):
        return Radiance(Color(0,0,0),0)


class Mirror:
    def __init__(self, world, tint):
        self.world = world
        self.tint = tint

    def get_radiance(self, object_hit, intersection_point, view_ray, depth=0):
        V_vector = -view_ray.direction
        N_vector = object_hit.normal_at(intersection_point)

        R_vector = reflect(V_vector, N_vector).normalize()

        return self.world.get_radiance_from_ray(Ray(intersection_point, R_vector), exclude_object=object_hit)


class Phong:
    def __init__(self, world, ka, kd, ks, ke, diffuse_color, specular_color, shininess):
        self.world = world
        self.ambient_color = Color(.01, .01, .01)
        self.diffuse_color = diffuse_color
        self.specular_color = specular_color
        self.shininess = shininess
        self.ka = ka
        self.kd = kd
        self.ks = ks
        self.ke = ke

        # ambient, diffuse, sepcular, exponent

    def get_radiance(self, object_hit, intersection_point, view_ray):
        N_vector = object_hit.normal_at(intersection_point)
        V_vector = -view_ray.direction

        total_radiance = Radiance(self.diffuse_color, BACKGROUND_RADIANCE.luminance)

        shadow_origin = intersection_point + (N_vector.multiply(.1).to_point())
        for light in self.world.lights:
            L_vector = (light.pos - intersection_point).to_vector().normalize()
            R_vector = reflect(L_vector, N_vector).normalize()

            # ray trace to the lights
            ray_to_light = Ray(shadow_origin, L_vector)
            shadow_object, shadow_point, shadow_normal_vec = self.world.intersects(ray_to_light)

            if shadow_object is None:  # out of shadow
                # adding diffuse
                total_radiance.color += self.diffuse_color.multiply(self.kd * dot(L_vector, N_vector))
                # adding specular
                total_radiance.color += self.specular_color.multiply(
                    self.ks * pow(abs(dot(R_vector, V_vector)), self.shininess))
                total_radiance.luminance += light.get_lums_from_point(intersection_point)
            else:  # in shadow
                # total_radiance.color += self.diffuse_color.multiply(self.kd * dot(L_vector, N_vector))

                if isinstance(shadow_object.material, Cloud): # if it's in a cloud's shadow
                    cloud = shadow_object.material

                    fraction_transmitted = cloud.get_frac_transmitted_from_light(shadow_object, shadow_point, light.pos, mult_fac=CLOUD_SHADOW_MULTIPLIER)

                    lighted_diff = self.diffuse_color+light.color*.1
                    total_radiance.color += lighted_diff.multiply(self.kd * dot(L_vector, N_vector))
                    # adding specular
                    lighted_spec = self.specular_color+light.color*.1
                    total_radiance.color += lighted_spec.multiply(self.ks * pow(abs(dot(R_vector, V_vector)), self.shininess)) * fraction_transmitted
                    total_radiance.luminance += light.get_lums_from_point(intersection_point) * fraction_transmitted

        return total_radiance


class Chessboard:
    def __init__(self, world):
        self.world = world
        pass

    def get_radiance(self, object_hit, intersection_point, view_ray):
        white = Color(1, 1, 1)
        scale = 1.8

        # color1 = Color(3/255, 186/255, 252/255)
        # color2 = Color(7/255, 60/255, 145/255)
        color1 = Color(180 / 255, 180 / 255, 180 / 255)
        color2 = Color(20/255, 20/255, 20/255)

        if (int(intersection_point.x*scale-100)+int(intersection_point.y*scale-100)+int(intersection_point.z*scale)) % 2 == 1:
            mat = Phong(self.world, .2, .7, .4, 1,  color1, white, .2)
            return mat.get_radiance(object_hit, intersection_point, view_ray)
        else:
            mat = Phong(self.world, .2, .7, .4, 1, color2, white, .2)
            return mat.get_radiance(object_hit, intersection_point, view_ray)


def mix_radiance(rad_1, rad_2, fac): # fac 0 is rad_1 and fac 1 is rad_2
    c1 = rad_1.color
    c2 = rad_2.color
    cm = c1 + (c2 - c1) * fac

    l1 = rad_1.luminance
    l2 = rad_2.luminance
    lm = l1 + (l2 - l1) * fac
    return Radiance(cm, lm)


def add_radiance(rad_1, rad_2, alpha_2):
    # alpha_2 = (1-alpha_2)
    return Radiance(rad_1.color*(1-alpha_2) + rad_2.color*alpha_2, rad_1.luminance*(1-alpha_2) + rad_2.luminance*alpha_2)


class World:
    def __init__(self,  objects):
        self.objects = objects
        self.lights = []
        # self.camera = camera

    def add_triangle(self, p1, p2, p3, material):
        self.objects.append(Triangle(p1, p2, p3, material))
        self.objects.append(Triangle(p1, p3, p2, material))

    def get_radiance_from_ray(self, vision_ray, exclude_object=None):
        hit_object, hit_point, normal_vector = self.intersects(vision_ray, exclude_object=exclude_object)

        shadow_object = None
        radiance_stack = [] # this is the stack of transparencies that needs to be collapsed
        if hit_object is not None: # if there is something seen
            radiance = hit_object.material.get_radiance(hit_object, hit_point, vision_ray)

            while isinstance(radiance.color, Color_Alpha): # if the first thing you see is transparent
                radiance_stack.append(radiance)
                
                hit_object_2, hit_point_2, normal_vector = self.intersects(Ray(hit_point, vision_ray.direction), exclude_object=hit_object)
                if hit_object_2 is None:
                    radiance_stack.append(BACKGROUND_RADIANCE)
                    break
                else:
                    next_rad = hit_object_2.material.get_radiance(hit_object_2, hit_point_2, vision_ray)
                    radiance_stack.append(next_rad)
                    radiance = next_rad

                    hit_object = hit_object_2
                    hit_point = hit_point_2

            # collapese the radiance stack
            if len(radiance_stack) >= 2:
                # print(radiance_stack)
                collapsed_radiance = radiance_stack[len(radiance_stack)-1]
                for radiance_index in range(len(radiance_stack)-2, -1, -1):
                    rad_w_alpha = radiance_stack[radiance_index]
                    rad_alpha = rad_w_alpha.color.a
                    rad_col = rad_w_alpha.color.color
                    rad_lum = rad_w_alpha.luminance

                    # print(rad_lum)

                    rad_no_alpha = Radiance(rad_col, rad_lum)
                    collapsed_radiance = add_radiance(collapsed_radiance, rad_no_alpha, rad_alpha)

                radiance = collapsed_radiance

            return radiance
        else:
            return BACKGROUND_RADIANCE #background radiance

    def intersects(self, ray, exclude_object=None): # should instead return location of intersection point and object reference.
        intersections = []
        objects_list = self.objects.copy()
        if exclude_object in objects_list:
            objects_list.remove(exclude_object)
        for ob in objects_list:
            does_hit, dist_of_hit = ob.intersection(ray)
            if exclude_object is not None:
                if ob == exclude_object:
                    does_hit = False
            if does_hit:
                intersections.append((ob, dist_of_hit))

        intersections_past_zero = []
        for intersection in intersections:
            if intersection[1] >= .001:
                # print("intersection_past_zero")
                intersections_past_zero.append(intersection)
        if len(intersections_past_zero) == 0:
            return None, None, None
        # find minimum distance hit
        min_dist = math.inf
        closest_object = None
        for intersection in intersections_past_zero:
            intersection_object = intersection[0]
            intersection_dist = intersection[1]

            if intersection_dist < min_dist:
                min_dist = intersection_dist
                closest_object = intersection_object
        intersection_point = ray.pos_at_t(min_dist)
        normal_vector = closest_object.normal_at(intersection_point)
        return closest_object, intersection_point, normal_vector

    def inside(self, point):
        for ob in self.objects:
            if isinstance(ob, Sphere) and dist(ob.position, point) < ob.radius:
                return True
        return False


def render_turning(world, the_camera, upsize):
    num_wedges = 32
    og_cam_pos = the_camera.position
    og_cam_lookat = the_camera.look_at.to_point()
    for camera_rotation in range(num_wedges):
        rot_angle = camera_rotation/num_wedges*math.pi*2
        the_camera.position = rotate_point(og_cam_pos, -rot_angle, "Z")
        the_camera.look_at = rotate_point(og_cam_lookat, rot_angle, "Z").to_vector().normalize()

        the_camera.look_at.dx *= -1

        print("rendering frame "+str(camera_rotation))
        # print("POSITION: " + str(the_camera.position.x) + ", " + str(the_camera.position.y))
        # print("LOOK AT: "+str(the_camera.look_at.dx)+", "+str(the_camera.look_at.dy))
        image = the_camera.cont_render(world, upsize)
        resized_image = cv2.resize(image, (the_camera.im_width * upsize, the_camera.im_height * upsize), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite("cloud_"+str(camera_rotation)+".png", resized_image * 255)

        cv2.waitKey(1000)

def make_cloud(world, x):
    # creating the cloud material
    cloud_objects = list()
    m = Pitch(world)

    # cloud_objects.append(Sphere(Point(0,0,.5), .6, m))
    # cloud_objects.append(Sphere(Point(-.5, 0, .7), .3, m))

    sc = .4
    cloud_objects.append(Sphere(Point(-2.89 * sc+x, .120 * sc, -1.1 * sc), 1.71 * sc, m))
    cloud_objects.append(Sphere(Point(-1.38 * sc+x, .414 * sc, .857 * sc), 1.91 * sc, m))
    cloud_objects.append(Sphere(Point(.987 * sc+x, -.243 * sc, -.764 * sc), 1.34 * sc, m))
    cloud_objects.append(Sphere(Point(-.894 * sc+x, -1.26 * sc, -1.08 * sc), 2.05 * sc, m))
    cloud_objects.append(Sphere(Point(.121 * sc+x, -.331 * sc, .196 * sc), 1.6 * sc, m))
    cloud_objects.append(Sphere(Point(1.51 * sc+x, 1.23 * sc, 1.41 * sc), 1.87 * sc, m))

    cloud_material = Cloud(world, World(cloud_objects))

    cloud = AxisBox(Point(-5 * sc+x, -4 * sc, -2 * sc), Point(4 * sc+x, 4 * sc, 4 * sc), cloud_material)
    return cloud

def main():
    # set up world
    world = World([])

    ka = .01
    kd = .6
    ks = .6
    ke = .2
    diffuse_color = Color(.6, .2, .2)
    specular_color = Color(.3, .8, .4)
    shininess = 2
    material_1 = Phong(world, ka, kd, ks, ke, diffuse_color, specular_color, shininess)

    ka = .3
    kd = .8
    ks = 4
    ke = 2
    diffuse_color = Color(1, .3, .6)
    specular_color = Color(.8, .5, .4)
    shininess = 2
    material_2 = Phong(world, ka, kd, ks, ke, diffuse_color, specular_color, shininess)

    SETUP = "cloud"

    camera_pos = Point(.6276, -6.926, 1.232)

    look_at = Vector(0, 1, -.09).normalize()

    up_vec = Vector(0, 0, 1)
    im_width = 200
    im_height = 100

    chess_material = Chessboard(world)
    mirror_material = Mirror(world, Color(1, 1, 1))

    if SETUP == "scene1":

        world.objects.append(Sphere(Point(-.238, .03, .778), .655, mirror_material))
        world.objects.append(Sphere(Point(.734, 3, 1.31), .64, material_2))

        # world.objects.append(Sphere(Point(0,0,0), .4, mirror_material))

        # points of the plane
        p1 = Point(-4.42, -3.24, 0)
        p2 = Point(2.4, -3.24, 0)
        p3 = Point(2.4, 23.5, 0)
        p4 = Point(-4.42, 23.5, 0)

        world.objects.append(Triangle(p1, p2, p3, chess_material))
        world.objects.append(Triangle(p3, p4, p1, chess_material))

    if SETUP == "cloud":
        im_width = int(RESOLUTION/RATIO)
        im_height = RESOLUTION

        # world.objects.append(make_cloud(world, -5))
        world.objects.append(make_cloud(world, 0))
        # world.objects.append(make_cloud(world, 5))

        #over
        # camera_pos = Point(0, -10, 1)
        # look_at = Vector(0, 10, -1).normalize()
        #under
        camera_pos = Point(0, -15, -1)
        look_at = Vector(0, 15, 1).normalize()

        plane_w = 30
        plane_h = 10
        plane_z = -2.3
        p1 = Point(-plane_w/2, -plane_h/2, plane_z)
        p2 = Point(plane_w/2, -plane_h/2, plane_z)
        p3 = Point(plane_w/2, plane_h/2, plane_z)
        p4 = Point(-plane_w/2, plane_h/2, plane_z)

        world.objects.append(Triangle(p1, p2, p3, chess_material))
        world.objects.append(Triangle(p3, p4, p1, chess_material))

    world.lights = WORLD_LIGHTS

    # set up camera
    the_camera = Camera(camera_pos, up_vec, look_at, 200, im_width, im_height, UPSIZE)

    if TURNING:
        render_turning(world, the_camera, UPSIZE)
    else:

        sum_im = the_camera.cont_render(world, UPSIZE)

        cv2.imwrite("latest_render.png", sum_im*255)
        while True:
            key = cv2.waitKey(1)
            if key == ord('c'):
                pass
                # the_camera.position = Point(0,0,-2.5)
                # the_camera.look_at = Vector(0,0,1).normalize()
                # the_camera.up = Vector(0,1,0)
                # the_camera.cont_render(world, UPSIZE)
            if key == ord('q'):
                break


if __name__ == "__main__":
    main()