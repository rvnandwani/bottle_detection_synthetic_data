import bpy
import glob
import os
import numpy as np
import sys
from bpy_extras.object_utils import world_to_camera_view
import json

sys.path.append("./")


def clean_enviroment():
    # Deselect all objects
    bpy.ops.object.select_all(action="DESELECT")

    # Select all objects
    bpy.ops.object.select_all(action="SELECT")

    # Delete all selected objects
    bpy.ops.object.delete()

    # Remove all meshes
    bpy.ops.outliner.orphans_purge(do_recursive=True)

    # Remove all materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material)

    # Remove all textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture)

    # Remove all images
    for image in bpy.data.images:
        bpy.data.images.remove(image)

    # Remove all cameras
    for camera in bpy.data.cameras:
        bpy.data.cameras.remove(camera)

    # Remove all lights
    for light in bpy.data.lights:
        bpy.data.lights.remove(light)

    # Remove all meshes
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)

    # Remove all curves
    for curve in bpy.data.curves:
        bpy.data.curves.remove(curve)

    # Remove all metaballs
    for metaball in bpy.data.metaballs:
        bpy.data.metaballs.remove(metaball)

    # Remove all fonts
    for font in bpy.data.fonts:
        bpy.data.fonts.remove(font)

    # Remove all armatures
    for armature in bpy.data.armatures:
        bpy.data.armatures.remove(armature)

    # Remove all actions
    for action in bpy.data.actions:
        bpy.data.actions.remove(action)

    # Remove all node groups
    for node_group in bpy.data.node_groups:
        bpy.data.node_groups.remove(node_group)

    # Remove all collections
    for collection in bpy.data.collections:
        bpy.data.collections.remove(collection)

    # Remove all particles
    for particle in bpy.data.particles:
        bpy.data.particles.remove(particle)

    # Remove all speakers
    for speaker in bpy.data.speakers:
        bpy.data.speakers.remove(speaker)

    # Remove all sounds
    for sound in bpy.data.sounds:
        bpy.data.sounds.remove(sound)

    # Remove all grease pencils
    for grease_pencil in bpy.data.grease_pencils:
        bpy.data.grease_pencils.remove(grease_pencil)

    # Remove all lattice
    for lattice in bpy.data.lattices:
        bpy.data.lattices.remove(lattice)

    # Remove all armatures
    for armature in bpy.data.armatures:
        bpy.data.armatures.remove(armature)

    # Remove all node groups
    for node_group in bpy.data.node_groups:
        bpy.data.node_groups.remove(node_group)

    # Remove all objects
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj)

    # Remove all brushes
    for brush in bpy.data.brushes:
        bpy.data.brushes.remove(brush)

    # Remove all palettes
    for palette in bpy.data.palettes:
        bpy.data.palettes.remove(palette)

    # Remove all lights
    for light in bpy.data.lights:
        bpy.data.lights.remove(light)

    # Remove all worlds
    # for world in bpy.data.worlds:
    #     bpy.data.worlds.remove(world)

    # Remove all cache files
    for cache_file in bpy.data.cache_files:
        bpy.data.cache_files.remove(cache_file)

    # Remove all libraries
    for library in bpy.data.libraries:
        bpy.data.libraries.remove(library)

    # Remove all images
    for image in bpy.data.images:
        bpy.data.images.remove(image)

    # Remove all keying sets
    # for keying_set in bpy.data.keying_sets:
    #     bpy.data.keying_sets.remove(keying_set)

    # Remove all line styles
    for line_style in bpy.data.linestyles:
        bpy.data.linestyles.remove(line_style)

    # Remove all mask
    for mask in bpy.data.masks:
        bpy.data.masks.remove(mask)

    # Remove all paint curves
    for paint_curve in bpy.data.paint_curves:
        bpy.data.paint_curves.remove(paint_curve)

    # Remove all point clouds
    for point_cloud in bpy.data.pointclouds:
        bpy.data.pointclouds.remove(point_cloud)

    # Remove all simulations
    # for simulation in bpy.data.simulations:
    #     bpy.data.simulations.remove(simulation)

    # Remove all text blocks
    for text in bpy.data.texts:
        bpy.data.texts.remove(text)

    # Remove all hair curves
    for hair_curve in bpy.data.hair_curves:
        bpy.data.hair_curves.remove(hair_curve)

    # Remove all curves
    for curve in bpy.data.curves:
        bpy.data.curves.remove(curve)

    # Remove all lattices
    for lattice in bpy.data.lattices:
        bpy.data.lattices.remove(lattice)


def set_resolution(image_size):
    """
    sets rendered images size
    https://docs.blender.org/api/current/bpy.types.RenderSettings.html#bpy.types.RenderSettings.resolution_x
    """
    bpy.context.scene.render.resolution_x = image_size[0]
    bpy.context.scene.render.resolution_y = image_size[1]


def jitter_position_xy(obj, x_range=(-0.3, 0.3), y_range=(-0.3, 0.3)):
    """
    jitter given objects position in x & y axes
    """
    obj.location.x = np.random.uniform(x_range[0], x_range[1])
    obj.location.y = np.random.uniform(y_range[0], y_range[1])
    obj.location.z = 0.0


def render_image(filepath="example_image.png"):
    """
    renders image to given path.
    https://docs.blender.org/api/current/bpy.ops.render.html#bpy.ops.render.render
    """
    bpy.context.scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)


def add_light_source(
    light_name,
    x_range=(-3, 3),
    y_range=(-3, 3),
    z_range=(1, 3),
    strength_range=(0.5, 15),
):
    """
    makes a 'SUN' type light source to the scene and jitter its position
    """

    light_data = bpy.data.lights.new(name=light_name, type="SUN")
    light_data.energy = np.random.uniform(strength_range[0], strength_range[1])
    # light_data.energy = 0.5
    light_source = bpy.data.objects.new(name="light", object_data=light_data)
    bpy.context.collection.objects.link(light_source)
    light_source.location.x = np.random.uniform(x_range[0], x_range[1])
    light_source.location.y = np.random.uniform(y_range[0], y_range[1])
    light_source.location.z = np.random.uniform(z_range[0], z_range[1])


# def setup_new_camera(location=(0.,1.2,0.9), rotation_euler=(np.deg2rad(-130), np.deg2rad(180), np.deg2rad(0))):
def setup_new_camera(
    img_id,
    location=(-1.2523229420976516, -0.42750112949390956, 0.2375570138718806),
    rotation_euler=(np.deg2rad(81.933), np.deg2rad(-8.241), np.deg2rad(-71.152)),
):
    """
    creates camera object sets pose with given args
    """
    cam = bpy.data.cameras.new(name=f"Camera_{img_id}")
    cam = bpy.data.objects.new("Camera_{img_id}_object", object_data=cam)
    bpy.context.collection.objects.link(cam)
    cam.location = location
    cam.rotation_euler = rotation_euler
    bpy.context.scene.camera = cam
    return cam


def remove_camera(camera_object_name):
    camera_object = bpy.data.objects.get("Camera_1_object")
    if camera_object:
        bpy.context.collection.objects.unlink(camera_object)
        bpy.context.view_layer.update()
        bpy.data.cameras.remove(camera_object.data)
        bpy.data.objects.remove(camera_object)


def add_ground_plane(hide=False):
    """
    adds ground plane
    """
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, -0.01), scale=(3, 3, 3))
    ground = bpy.context.selected_objects[-1]

    # make invisible on rendered image
    if hide:
        ground.hide_render = True


def add_hdri_background(hdri_path):
    """
    adds hdri background to the scene
    used for reference: https://blender.stackexchange.com/questions/209584/using-python-to-add-an-hdri-to-world-node
    you can easily get hdri files for free example from: https://polyhaven.com/hdris
    """
    # get shader nodes
    world = bpy.data.worlds[0]
    node_tree = world.node_tree
    nodes = node_tree.nodes

    # clear all nodes
    nodes.clear()

    # add new background node
    node_background = nodes.new(type="ShaderNodeBackground")

    # add environment texture node
    node_environment = nodes.new("ShaderNodeTexEnvironment")

    # load hdr from the passed path
    node_environment.image = bpy.data.images.load(hdri_path)

    # add output node
    node_output = nodes.new(type="ShaderNodeOutputWorld")

    # Link all nodes
    links = node_tree.links
    link = links.new(node_environment.outputs["Color"], node_background.inputs["Color"])
    link = links.new(
        node_background.outputs["Background"], node_output.inputs["Surface"]
    )


def calculate_2d_bounding_box(obj, camera):
    """
    Calculates the 2D bounding box of an object in screen space using the world_to_camera_view function.
    """
    # Ensure the correct context
    bpy.context.view_layer.update()

    # get current blender scene
    scene = bpy.context.scene

    # Initialize min and max coordinates
    min_co = [float("inf")] * 2
    max_co = [-float("inf")] * 2

    # loop through all 3d model vertices to get 2d bounding box
    camera_coords = []
    for v in obj.data.vertices:
        world_co = obj.matrix_world @ v.co
        camera_co = world_to_camera_view(scene, camera, world_co)
        camera_coords.append(camera_co)

        # Convert camera coordinates to pixel coordinates
        x = int(camera_co.x * scene.render.resolution_x)
        y = int((1 - camera_co.y) * scene.render.resolution_y)

        # Update min and max coordinates
        min_co[0] = min(min_co[0], x)
        min_co[1] = min(min_co[1], y)
        max_co[0] = max(max_co[0], x)
        max_co[1] = max(max_co[1], y)

    # save center of volume to check at later stage if drink occluded or not
    center_of_volume = np.mean(np.stack(camera_coords), 0)

    return [min_co[0], min_co[1], max_co[0], max_co[1]], center_of_volume


def IoU(bbox1, bbox2):
    """
    Computes intersection over Union between two bounding boxes
    """
    xA = max(bbox1[0], bbox2[0])
    yA = max(bbox1[1], bbox2[1])
    xB = min(bbox1[2], bbox2[2])
    yB = min(bbox1[3], bbox2[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the Union area
    boxAArea = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    boxBArea = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)
    union = boxAArea + boxBArea - interArea
    iou = interArea / union if union != 0 else 0
    return iou


def box_fully_covered(bbox1, bbox2):
    """
    Checks if 'bbox1' is fully inside 'bbox2' boundaries in 2d
    """
    return (
        bbox1[0] >= bbox2[0]
        and bbox1[1] >= bbox2[1]
        and bbox1[2] <= bbox2[2]
        and bbox1[2] <= bbox2[2]
    )


def check_occlusion(ann1, ann2, overlap_thre=0.7):
    """
    Compares how much box is overlapping with other boxes one at a time.
    If box is overlapping above threshold then it will get removed if further away from the camera than the other box
    """
    a1_box = ann1["bbox"]
    a2_box = ann2["bbox"]

    is_occluded = False
    if IoU(a1_box, a2_box) > overlap_thre or box_fully_covered(a1_box, a2_box):
        a1_z = ann1["center_of_volume"][2]
        a2_z = ann2["center_of_volume"][2]
        if a1_z > a2_z:
            is_occluded = True

    return is_occluded


def remove_light_source(light_name):
    light_object = bpy.data.objects.get(light_name)

    if light_object:
        bpy.context.collection.objects.unlink(light_object)
        bpy.data.lights.remove(light_object.data)
        bpy.data.objects.remove(light_object)


def check_area(box):
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1) >= 1000


def check_rectangle_position(in_rect, out_rect):
    x1, y1, x2, y2 = in_rect[0], in_rect[1], in_rect[2], in_rect[3]
    X1, Y1, X2, Y2 = out_rect[0], out_rect[1], out_rect[2], out_rect[3]

    def is_point_inside(px, py):
        return X1 <= px <= X2 and Y1 <= py <= Y2

    corners_inside = [
        is_point_inside(x1, y1),
        is_point_inside(x1, y2),
        is_point_inside(x2, y1),
        is_point_inside(x2, y2),
    ]
    if all(corners_inside):
        if check_area(in_rect):
            return in_rect
        else:
            None
    elif any(corners_inside):
        if x1 < X1:
            x1 = X1
        if y1 < Y1:
            y1 = Y1
        if x2 > X2:
            x2 = X2
        if y2 > Y2:
            y2 = Y2
        box = [x1, y1, x2, y2]
        if check_area(box):
            return box
        else:
            None
    else:
        return None


def remove_occluded_annotations(annotations, ann_count, img_id, image_name, image_size):
    """
    A simple method to remove occluded object annotations using IoU and drink distance in z-axis w.r.t to camera
    """
    indices_to_remove = []
    # loop through all the annotations
    for a1_idx, ann1 in enumerate(annotations):
        for a2_idx, ann2 in enumerate(annotations):
            if a1_idx == a2_idx:
                continue
            # check occlusion
            occluded = check_occlusion(ann1, ann2)
            # add to removal list if occluded
            if occluded:
                indices_to_remove.append(a1_idx)
    ann_ret = []
    for idx, ann in enumerate(annotations):
        ann["bbox"] = check_rectangle_position(
            ann["bbox"], (0, 0, image_size[0] - 1, image_size[1] - 1)
        )
        if idx not in indices_to_remove and ann["bbox"]:
            obj_id = 1 if ann["class"] == "bottle" else 2
            ann_ret.append(
                {
                    "id": ann_count,
                    "image_id": img_id,
                    "img_name": image_name,
                    "categry_id": obj_id,
                    "category": ann["class"],
                    "bbox": ann["bbox"],
                }
            )
            ann_count += 1
    return ann_ret, ann_count


if __name__ == "__main__":

    assets_dir = "./assets"
    save_blend = True
    add_ground = False
    save_viz = False  # NOTE: requires opencv-python. Setting arg to 'True' will make script to install opencv-python automatically
    image_savepath = "example_image.png"
    annotations_savepath = "example_annotations.json"
    annotations = []

    # get 3d model paths
    bottles_path = sorted(
        glob.glob(os.path.join(assets_dir, "bottle", "**", "*.obj"), recursive=True)
    )
    print(f"found {len(bottles_path)} bottles")

    # clean scene
    clean_enviroment()

    # load hdri background
    add_hdri_background("hdris/fireplace_4k.hdr")

    # get camera
    cam = setup_new_camera()

    # set resolution for rendered image
    image_size = (640, 640)
    set_resolution(image_size)

    # add "ground" below drinks
    if add_ground:
        add_ground_plane(hide=False)

    # add "sun" light source
    add_light_source(light_name="light1")

    # load all found .obj models
    for bottle_path in bottles_path:
        bpy.ops.import_scene.obj(filepath=bottle_path)
        # get newly added object
        bottle = bpy.context.selected_objects[-1]
        # rescale object x,y,z dimensions to fit image better
        bottle.scale = bottle.scale * 0.1
        # place drink randomly in x & y plane
        jitter_position_xy(bottle)
        # calculate and store the 2D bounding box of the bottle
        bbox, center_of_volume = calculate_2d_bounding_box(bottle, cam)
        annotations.append(
            {"class": "bottle", "bbox": bbox, "center_of_volume": center_of_volume}
        )

    # render image
    render_image(image_savepath)

    # simple (and not exaustive) logic to remove annotations that are occluded by other ones
    annotations_clean = remove_occluded_annotations(annotations)

    # save annotations as json file
    with open(annotations_savepath, "w") as json_file:
        json.dump(annotations_clean, json_file, indent=4)
    print(f"annotations file saved to: {annotations_savepath}")

    # save visualization image
    if save_viz:
        try:
            import cv2
        except:
            from subprocess import sys, call

            print("installing opencv-python...")
            call([sys.executable, "-m", "ensurepip"])
            call([sys.executable] + "-m pip install -U pip setuptools wheel".split())
            call([sys.executable] + "-m pip install -U opencv-python".split())
            import cv2

        rendered_image_org = cv2.imread(image_savepath)
        rendered_image = rendered_image_org.copy()
        sfilename, ext = os.path.splitext(image_savepath)
        for ann in annotations:
            bbox = ann["bbox"]
            cv2.rectangle(
                rendered_image,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                color=(0, 255, 0),
                thickness=2,
            )
        cv2.imwrite(f"{sfilename}_viz_preclean{ext}", rendered_image)

        rendered_image = rendered_image_org.copy()
        for ann in annotations_clean:
            bbox = ann["bbox"]
            cv2.rectangle(
                rendered_image,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                color=(0, 255, 0),
                thickness=2,
            )
        cv2.imwrite(f"{sfilename}_viz{ext}", rendered_image)

    # save blend file. Requires absolute path
    if save_blend:
        savepath = f"{os.getcwd()}/example.blend"
        bpy.ops.wm.save_mainfile(filepath=savepath)
        print(f"blender file saved to: {savepath}")
