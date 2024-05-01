
# -*- coding: utf-8 -*-
#
# Copyright (C) 2022 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#

import os
import torch
import numpy as np
from psbody.mesh import Mesh, MeshViewers
from psbody.mesh.sphere import Sphere
from psbody.mesh.colors import name_to_rgb
from psbody.mesh.lines import Lines
import scenepic as sp


# from tools.train_tools import point2point_signed
from tools.utils import aa2rotmat
from tools.utils import makepath
from tools.utils import to_cpu

def points_to_spheres(points, radius=0.1, vc=name_to_rgb['blue']):

    spheres = Mesh(v=[], f=[])
    for pidx, center in enumerate(points):
        clr = vc[pidx] if len(vc) > 3 else vc
        spheres.concatenate_mesh(Sphere(center, radius).to_mesh(color=clr))
    return spheres

def cage(length=1,vc=name_to_rgb['black']):

    cage_points = np.array([[-1., -1., -1.],
                            [1., 1., 1.],
                            [1., -1., 1.],
                            [-1., 1., -1.]])
    c = Mesh(v=length * cage_points, f=[], vc=vc)
    return c


def create_video(path, fps=30,name='movie'):
    import os
    import subprocess

    src = os.path.join(path,'%*.png')
    movie_path = os.path.join(path,'%s.mp4'%name)
    i = 0
    while os.path.isfile(movie_path):
        movie_path = os.path.join(path,'%s_%02d.mp4'%(name,i))
        i +=1


    cmd = 'ffmpeg -f image2 -r %d -i %s -b:v 6400k -pix_fmt yuv420p %s' % (fps, src, movie_path)
    subprocess.call(cmd.split(' '))
    while not os.path.exists(movie_path):
        continue


def get_ground(cage_size = 7, grnd_size = 5, axis_size = 1):
    ax_v = np.array([[0., 0., 0.],
                     [1.0, 0., 0.],
                     [0., 1., 0.],
                     [0., 0., 1.]])
    ax_e = [(0, 1), (0, 2), (0, 3)]

    axis_l = Lines(axis_size*ax_v, ax_e, vc=np.eye(4)[:, 1:])

    g_points = np.array([[-.2, 0.0, -.2],
                         [.2, 0.0, .2],
                         [.2, 0.0, -0.2],
                         [-.2, 0.0, .2]])
    g_faces = np.array([[0, 1, 2], [0, 3, 1]])
    grnd_mesh = Mesh(v=grnd_size * g_points, f=g_faces, vc=name_to_rgb['gray'])

    cage_points = np.array([[-.2, .0, -.2],
                            [.2, .2, .2],
                            [.2, 0., 0.2],
                            [-.2, .2, -.2]])
    cage = [Mesh(v=cage_size * cage_points, f=[], vc=name_to_rgb['black'])]
    return grnd_mesh, cage, axis_l

"""Example script demonstrating the basic ScenePic functionality."""

import argparse

import numpy as np
import scenepic as sp


def _parse_args():
    parser = argparse.ArgumentParser("Getting Started with ScenePic")
    parser.add_argument("--script", action="store_true",
                        help="Whether to save the scenepic as a JS file")
    return parser.parse_args()


def _main():
    args = _parse_args()

    # when we build a ScenePic we are essentially building a web
    # page, and the ScenePic will automatically populate parts of
    # that webpage.

    # the scene object acts as the root of the entire ScenePic environment
    scene = sp.Scene()

    # you can use it to create one or more canvases to display 3D or 2D
    # objects. Canvas objects display Frames. For a static ScenePic, there
    # will only be a single Frame, but you can use multiple frames to create
    # an animation or to display a range of visualizations in the same visual
    # space. We will create one 3D canvas to display the full scene, and then
    # some 2D canvases which will show projections of the scene.
    main = scene.create_canvas_3d(width=600, height=600)
    projx = scene.create_canvas_2d("projx", width=200, height=200)
    projy = scene.create_canvas_2d("projy", width=200, height=200)
    projz = scene.create_canvas_2d("projz", width=200, height=200)

    # the scene object is also used to create Mesh objects that will be added
    # to frames. We are going to create an animation of some spheres orbiting
    # a fixed cube, so let's create a default unit cube to start.
    cube = scene.create_mesh("cube")

    # the Mesh object has a variety of methods for adding primitive objects
    # or loading arbitrary mesh geometry. In this example, we will just
    # be using primitives, but the python tutorial shows all the various
    # methods for adding geometry to a mesh.
    cube.add_cube(color=sp.Colors.White)

    # let's create our spheres as well, using some different colors
    sphere_names = ["red", "green", "blue"]
    sphere_colors = [sp.Colors.Red, sp.Colors.Green, sp.Colors.Blue]
    spheres = []
    for name, color in zip(sphere_names, sphere_colors):
        # by placing each sphere on a different layer, we can toggle them on and off
        sphere = scene.create_mesh("{}_sphere".format(name), layer_id=name)
        sphere.add_sphere(color=color, transform=sp.Transforms.scale(0.5))
        spheres.append(sphere)

    # now we will iteratively create each frame of the animation.
    for i in range(180):
        # first we create a frame object. This will be used to populate
        # the 3D canvas.
        main_frame = main.create_frame()

        # Now that we have a frame, we can add the cube mesh to the frame
        main_frame.add_mesh(cube)

        # Next, we add the spheres. ScenePic has a variety of useful tools
        # for operating on 3D data. Some of the most useful enable us to
        # create transforms to move meshes around. Let's create the
        # transforms for our three rotating spheres and add them to the frame.
        # NB The Python interface uses numpy for matrices and vectors.
        positions = np.concatenate([np.eye(3, dtype=np.float32) * 1.3,
                                    np.ones((3, 1), dtype=np.float32)], axis=-1)
        inc = 2 * np.pi / 180
        positions[0] = sp.Transforms.RotationAboutYAxis(inc * i) @ positions[0].T
        positions[1] = sp.Transforms.RotationAboutZAxis(2 * inc * i) @ positions[1].T
        positions[2] = sp.Transforms.RotationAboutXAxis(3 * inc * i) @ positions[2].T
        positions = positions[:, :3]
        for sphere, position in zip(spheres, positions):
            transform = sp.Transforms.translate(position)
            main_frame.add_mesh(sphere, transform=transform)

        # now we'll populate our projections
        for j, proj in enumerate([projx, projy, projz]):
            proj_frame = proj.create_frame()

            # 2D frames work in pixels (as oppose to world units) so we need
            # to convert positions to pixels.
            proj_frame.add_rectangle(75, 75, 50, 50, fill_color=sp.Colors.White)
            points = np.roll(positions, j, axis=1)[:, 1:]
            points[:, 1] *= -1
            points = points * 50 + 100

            for point, color in zip(points, sphere_colors):
                proj_frame.add_circle(point[0], point[1], 12.5, fill_color=color)

            # let's add some label text
            proj_frame.add_text(proj.canvas_id, 10, 190, size_in_pixels=16)

    # this will make user interactions happen to all canvases simultaneously
    scene.link_canvas_events(main, projx, projy, projz)

    # ScenePic provides some useful layout controls by exposing CSS grid commands
    scene.grid(width="800px", grid_template_rows="200px 200px 200px", grid_template_columns="600px 200px")
    scene.place(main.canvas_id, "1 / span 3", "1")
    scene.place(projx.canvas_id, "1", "2")
    scene.place(projy.canvas_id, "2", "2")
    scene.place(projz.canvas_id, "3", "2")

    # The scene is complete, so we write it to a standalone file.
    if args.script:
        # If you have an existing HTML page you want to add a scenepic
        # to, then you can save the scenepic as a self-contained
        # Javascript file.
        scene.save_as_script("getting_started.js", standalone=True)
    else:
        # However, ScenePic will also create a basic HTML wrapper
        # and embed the Javascript into the file directly so you
        # have a single file containing everything.
        scene.save_as_html("getting_started.html", title="Getting Started")



class sp_animation():
    def __init__(self,
                 canvs = {'main':[1600, 1600, ['1','1']]},
                 grid = ['1600px', '1600px', '1600px'],
                 bg_color = sp.Colors.Black,
                 ):
        super(sp_animation, self).__init__()

        self.scene = sp.Scene()
        # self.scene.create_text_panel("text", "Hello World")
        # self.main = self.scene.create_canvas_3d(width=width, height=height)

        self.canvs = {}
        for name, v in canvs.items():
            width, height, pos = v
            self.canvs[name] = self.scene.create_canvas_3d(name, width=width, height=height, shading=sp.Shading(bg_color=bg_color))
            self.scene.place(self.canvs[name].canvas_id, pos[0], pos[1])

        self.scene.grid(width=grid[0], grid_template_rows=grid[1], grid_template_columns=grid[2])
        # self.scene.link_canvas_events(*list(self.canvs.values()))

        self.colors = sp.Colors

    def meshes_to_sp(self,meshes_list):

        sp_meshes = {}

        for name, m in meshes_list.items():
            params = {'vertices' : m.v.astype(np.float32),
                      'normals' : m.estimate_vertex_normals().astype(np.float32),
                      'triangles' : m.f,
                      'colors' : m.vc.astype(np.float32)}
            # params = {'vertices' : m.v.astype(np.float32), 'triangles' : m.f, 'colors' : m.vc.astype(np.float32)}
            # sp_m = sp.Mesh()
            sp_m = self.scene.create_mesh(layer_id = name)
            sp_m.add_mesh_with_normals(**params)
            if name == 'ground_mesh':
                sp_m.double_sided=True
            sp_meshes[name] = sp_m

        return sp_meshes

    def add_frame(self,meshes_list_ps,canvs_meshes=None, focus = 'all'):
        

        meshes_list = self.meshes_to_sp(meshes_list_ps)
        # if not hasattr(self,'focus_point'):
        if focus == 'all':
            focus_point = np.array([0,0,0])
        else:
            focus_point = meshes_list_ps[focus].v.mean(0)
            # center = self.focus_point
            # center[2] = 4
            # rotation = sp.Transforms.rotation_about_z(0)
            # self.camera = sp.Camera(center=center, rotation=rotation, fov_y_degrees=30.0)

        if canvs_meshes is None:
            name = list(self.canvs.keys())[0]           
            canvs_meshes = {name: list(meshes_list.keys())}
        
        # canv.add_text(name, 10, canv.height-10, size_in_pixels=16)
        for canv_name, meshes in canvs_meshes.items():
            canv = self.canvs[canv_name]
            main_frame = canv.create_frame(focus_point=focus_point)
            for i, m in enumerate(meshes):
                # self.main.set_layer_settings({layer_names[i]:{}})
                main_frame.add_mesh(meshes_list[m])

    def save_animation(self, sp_anim_name):
        # self.scene.link_canvas_events(self.main)
        self.scene.save_as_html(sp_anim_name, title=sp_anim_name.split('/')[-1])

class sp_animation_old():
    def __init__(self,
                 width = 1600,
                 height = 1600,
                 bg_color = sp.Colors.Black,
                 ):
        super(sp_animation_old, self).__init__()

        self.scene = sp.Scene()
        self.main = self.scene.create_canvas_3d(width=width, height=height, shading=sp.Shading(bg_color=bg_color))
        self.colors = sp.Colors

    def meshes_to_sp(self,meshes_list, layer_names):

        sp_meshes = []



        for i, m in enumerate(meshes_list):
            params = {'vertices' : m.v.astype(np.float32),
                      'normals' : m.estimate_vertex_normals().astype(np.float32),
                      'triangles' : m.f,
                      'colors' : m.vc.astype(np.float32)}
            # params = {'vertices' : m.v.astype(np.float32), 'triangles' : m.f, 'colors' : m.vc.astype(np.float32)}
            # sp_m = sp.Mesh()
            sp_m = self.scene.create_mesh(layer_id = layer_names[i])
            sp_m.add_mesh_with_normals(**params)
            if layer_names[i] == 'ground_mesh':
                sp_m.double_sided=True
            sp_meshes.append(sp_m)

        return sp_meshes

    def add_frame(self,meshes_list_ps, layer_names):

        meshes_list = self.meshes_to_sp(meshes_list_ps, layer_names)
        if not hasattr(self,'focus_point'):
            self.focus_point = meshes_list_ps[1].v.mean(0)
            # center = self.focus_point
            # center[2] = 4
            # rotation = sp.Transforms.rotation_about_z(0)
            # self.camera = sp.Camera(center=center, rotation=rotation, fov_y_degrees=30.0)
        for i, n in enumerate(layer_names):
            if 'obj' in n:
                self.focus_point = meshes_list_ps[i].v.mean(0)
                break
        main_frame = self.main.create_frame(focus_point=self.focus_point)
        for i, m in enumerate(meshes_list):
            # self.main.set_layer_settings({layer_names[i]:{}})
            main_frame.add_mesh(m)
        return main_frame
    
    def add_lines(self, lines_s, lines_e, lines_name, frame=None, color=None):
        
        if frame is None:
            frame = self.main.create_frame()

        for i, n in enumerate(lines_name):
            sp_m = self.scene.create_mesh(layer_id = n)
            sp_m.add_lines(lines_s[i], lines_e[i])
            frame.add_mesh(sp_m)


    def save_animation(self, sp_anim_name):
        self.scene.link_canvas_events(self.main)
        self.scene.save_as_html(sp_anim_name, title=sp_anim_name.split('/')[-1])



import trimesh
from tools.utils import to_np
def simplify_mesh(mesh=None, v=None, f=None, n_faces=None, vc=name_to_rgb['pink'], remove_verts = None):

    if mesh is None:
        mesh_tri = trimesh.Trimesh(vertices=v, faces=f, process=False)
    else:
        mesh_tri = trimesh.Trimesh(vertices=mesh.v, faces=mesh.f, process=False)

    if remove_verts is not None:
        verts_mask = np.ones(mesh_tri.vertices.shape[0])
        verts_mask[to_np(remove_verts)] = 0
        mesh_tri.update_vertices(verts_mask.astype(np.bool_))
    if n_faces is not None:
        mesh_tri = mesh_tri.simplify_quadratic_decimation(n_faces)
    # mesh_tri = mesh_tri.simplify_quadratic_decimation(n_faces)
    return Mesh(v=mesh_tri.vertices, f=mesh_tri.faces, vc=vc)

# import open3d as o3d
# import numpy as np

# def draw_geometries(pcds):
#     """
#     Draw Geometries
#     Args:
#         - pcds (): [pcd1,pcd2,...]
#     """
#     o3d.visualization.draw_geometries(pcds)

# def get_o3d_FOR(origin=[0, 0, 0],size=10):
#     """ 
#     Create a FOR that can be added to the open3d point cloud
#     """
#     mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
#     size=size)
#     mesh_frame.translate(origin)
#     return(mesh_frame)

# def vector_magnitude(vec):
#     """
#     Calculates a vector's magnitude.
#     Args:
#         - vec (): 
#     """
#     magnitude = np.sqrt(np.sum(vec**2))
#     return(magnitude)


# def calculate_zy_rotation_for_arrow(vec):
#     """
#     Calculates the rotations required to go from the vector vec to the 
#     z axis vector of the original FOR. The first rotation that is 
#     calculated is over the z axis. This will leave the vector vec on the
#     XZ plane. Then, the rotation over the y axis. 

#     Returns the angles of rotation over axis z and y required to
#     get the vector vec into the same orientation as axis z
#     of the original FOR

#     Args:
#         - vec (): 
#     """
#     # Rotation over z axis of the FOR
#     gamma = np.arctan(vec[1]/vec[0])
#     Rz = np.array([[np.cos(gamma),-np.sin(gamma),0],
#                    [np.sin(gamma),np.cos(gamma),0],
#                    [0,0,1]])
#     # Rotate vec to calculate next rotation
#     vec = Rz.T@vec.reshape(-1,1)
#     vec = vec.reshape(-1)
#     # Rotation over y axis of the FOR
#     beta = np.arctan(vec[0]/vec[2])
#     Ry = np.array([[np.cos(beta),0,np.sin(beta)],
#                    [0,1,0],
#                    [-np.sin(beta),0,np.cos(beta)]])
#     return(Rz, Ry)

# def create_arrow(scale=10):
#     """
#     Create an arrow in for Open3D
#     """
#     cone_height = scale*0.2
#     cylinder_height = scale*0.8
#     cone_radius = scale/10
#     cylinder_radius = scale/20
#     mesh_frame = o3d.geometry.TriangleMesh.create_arrow(cone_radius=1,
#         cone_height=cone_height,
#         cylinder_radius=0.5,
#         cylinder_height=cylinder_height)
#     return(mesh_frame)

# def get_arrow(origin=[0, 0, 0], end=None, vec=None):
#     """
#     Creates an arrow from an origin point to an end point,
#     or create an arrow from a vector vec starting from origin.
#     Args:
#         - end (): End point. [x,y,z]
#         - vec (): Vector. [i,j,k]
#     """
#     scale = 10
#     Ry = Rz = np.eye(3)
#     T = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
#     T[:3, -1] = origin
#     if end is not None:
#         vec = np.array(end) - np.array(origin)
#     elif vec is not None:
#         vec = np.array(vec)
#     if end is not None or vec is not None:
#         scale = vector_magnitude(vec)
#         Rz, Ry = calculate_zy_rotation_for_arrow(vec)
#     mesh = create_arrow(scale)
#     # Create the arrow
#     mesh.rotate(Ry, center=np.array([0, 0, 0]))
#     mesh.rotate(Rz, center=np.array([0, 0, 0]))
#     mesh.translate(origin)
#     return(mesh)

# def make_arrow():
#     # Create a Cartesian Frame of Reference
#     FOR = get_o3d_FOR()
#     # Create an arrow from point (5,5,5) to point (10,10,10)
#     # arrow = get_arrow([5,5,5],[10,10,10])

#     # Create an arrow representing vector vec, starting at (5,5,5)
#     # arrow = get_arrow([5,5,5],vec=[5,5,5])

#     # Create an arrow in the same place as the z axis
#     arrow = get_arrow()

#     # Draw everything
#     draw_geometries([FOR,arrow])