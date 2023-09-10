import numpy as np
import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
import pyrender


class MeshViewer(object):
    def __init__(self, img_h=1024, img_w=667, body_color=(1.0, 1.0, 0.9, 1.0), registered_keys=None):
        super(MeshViewer, self).__init__()

        if registered_keys is None:
            registered_keys = dict()

        import trimesh

        self.mat_constructor = pyrender.MetallicRoughnessMaterial
        self.mesh_constructor = trimesh.Trimesh
        self.trimesh_to_pymesh = pyrender.Mesh.from_trimesh
        self.transf = trimesh.transformations.rotation_matrix

        self.body_color = body_color
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 1.0], ambient_light=(0.3, 0.3, 0.3))

        self.light = pyrender.DirectionalLight(color=np.ones(3), intensity=5.0)

        self.img_h, self.img_w = img_h, img_w

    def render(self, camera_pose, intrinsic, coor_type="multiface"):
        assert len(camera_pose.shape) == 2
        for node in self.scene.get_nodes():
            if node.name in ["cam", "d_light"]:
                self.scene.remove_node(node)

        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy, znear=900, zfar=1300)
        if coor_type == "multiface":
            camera_pose[1:3] *= -1
        if camera_pose.shape[0] == 3:
            camera_pose = np.concatenate([camera_pose, np.array([[0, 0, 0, 1]])], axis=0)
            camera_pose = np.linalg.inv(camera_pose)

        self.scene.add(camera, pose=camera_pose, name="cam")
        self.scene.add(self.light, pose=camera_pose, name="d_light")

        renderer = pyrender.OffscreenRenderer(self.img_w, self.img_h)
        color, depth = renderer.render(self.scene)

        return color, depth

    def create_mesh(self, vertices, faces, color=(0.3, 0.3, 0.3, 1.0), wireframe=False):
        for node in self.scene.get_nodes():
            if node.name == "body_mesh":
                self.scene.remove_node(node)
                break

        material = self.mat_constructor(metallicFactor=0.0, alphaMode="BLEND", baseColorFactor=color)

        mesh = self.mesh_constructor(vertices, faces)

        # rot = self.transf(np.radians(180), [1, 0, 0])
        # mesh.apply_transform(rot)

        self.scene.add(self.trimesh_to_pymesh(mesh, material=material), name="body_mesh")

    def update_mesh(self, vertices, faces):
        for node in self.scene.get_nodes():
            if node.name == "body_mesh":
                self.scene.remove_node(node)
                break

        body_mesh = self.create_mesh(vertices, faces, color=self.body_color)
        self.scene.add(body_mesh, name="body_mesh")
