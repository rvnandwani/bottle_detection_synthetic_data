from datetime import datetime
import numpy as np
import os
import json
import glob
import bpy
from blender_helper import (
    clean_enviroment,
    add_hdri_background,
    setup_new_camera,
    set_resolution,
    add_light_source,
    add_ground_plane,
    jitter_position_xy,
    calculate_2d_bounding_box,
    render_image,
    remove_occluded_annotations,
    remove_light_source,
    remove_camera,
)
import random
import math
import sys
import argparse


def random_bool():
    return random.choice([True, False])


def Get3DObject(assetdir):
    labelpath = {}
    for asset in os.listdir(assetdir):
        labelpath[asset] = sorted(
            glob.glob(os.path.join(assetdir, asset, "**", "*.obj"), recursive=True)
        )
    return labelpath


def GetCameraPosition():
    x = random.uniform(0.3, 1.3) if random_bool() else random.uniform(0.3, 1.3) * -1
    y = random.uniform(0.3, 1.3) if random_bool() else random.uniform(0.3, 1.3) * -1
    z = random.uniform(0.1, 0.3)

    r_x = round(
        np.deg2rad(math.degrees(math.atan2(math.sqrt(x**2 + y**2), z - 0.05))), 3
    )
    r_y = round(np.deg2rad(random.uniform(-10.0, 10.0)), 3)
    r_z = round(np.deg2rad(90 + math.degrees(math.atan2(y, x))), 3)

    return ((round(x, 3), round(y, 3), round(z, 3)), (r_x, r_y, r_z))


if __name__ == "__main__":

    # Setting Dependencies
    assets_dir = "./assets"  # 3D Objects Directory
    hdris_dir = "./hdris"  # .HDR backgrounds directory
    image_size = (640, 640)  # Setting image resolution

    img_count = 0
    ann_count = 0
    ann_image = []
    images = []
    annotations = []

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dir",
        default="dataset/train/images",
        type=str,
        help="path to image storage directory",
    )
    parser.add_argument(
        "-a",
        "--ann",
        default="dataset/train/annotations_train.json",
        type=str,
        help="annotations file path with name",
    )
    parser.add_argument(
        "-b",
        "--batches",
        default=1,
        type=int,
        help="number of batches",
    )
    args = vars(parser.parse_args())

    # Retrieving arguments
    image_dir = args["dir"]
    annotations_savepath = args["ann"]
    batches = args["batches"]

    # Ensuring directories and format
    os.makedirs(image_dir, exist_ok=True)
    if annotations_savepath.endswith(".json"):
        if len(os.path.dirname(annotations_savepath)) > 0:
            os.makedirs(os.path.dirname(annotations_savepath), exist_ok=True)
    else:
        print("Please provide a filename and path with .json entension")
        sys.exit()

    # Gathering all the assets required to create the scene
    backgrounds = sorted(glob.glob(os.path.join(hdris_dir, "*.hdr")))
    assets = Get3DObject(assetdir=assets_dir)

    # Setting annotation json default values
    ann_full = {
        "info": {
            "description": "Bottle and Can detection",
            "date_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "categories": [
            {
                "id": 1,
                "name": "bottle",
            },
            {
                "id": 2,
                "name": "can",
            },
        ],
    }

    # Setting output image resolution
    set_resolution(image_size)

    for _ in range(batches):
        for bgr in backgrounds:
            print(f"#### Background : {bgr} ####")

            # Generating random camera positions
            cameras = [GetCameraPosition() for i in range(5)]

            for cam_loc in cameras:

                # Rendering scene objects
                clean_enviroment()
                add_hdri_background(bgr)
                add_light_source(light_name=f"light{img_count}")
                loc, rot = cam_loc
                image_savepath = f"Image_{img_count}.png"
                print(f"!!!!! Camera for {image_savepath} at {loc} and {rot} !!!!!")
                cam = setup_new_camera(img_count, location=loc, rotation_euler=rot)
                if random_bool():
                    add_ground_plane(True)

                image_savepath = f"Image_{img_count}.png"

                ann_temp = []
                for category in assets:
                    for obj in assets[category]:
                        for _ in range(random.randint(1, 3)):

                            # Rendering our assets
                            bpy.ops.import_scene.obj(filepath=obj)
                            bottle = bpy.context.selected_objects[-1]
                            bottle.scale = bottle.scale * 0.1
                            jitter_position_xy(bottle)
                            bbox, center_of_volume = calculate_2d_bounding_box(
                                bottle, cam
                            )
                            ann_temp.append(
                                {
                                    "class": category,
                                    "bbox": bbox,
                                    "center_of_volume": center_of_volume,
                                }
                            )

                # Saving image
                render_image(os.path.join(image_dir, image_savepath))
                images.append(
                    {
                        "id": img_count,
                        "width": image_size[0],
                        "height": image_size[1],
                        "filename": image_savepath,
                        "date_saved": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
                annotations_clean, ann_count = remove_occluded_annotations(
                    ann_temp, ann_count, img_count, image_savepath, image_size
                )
                ann_image += annotations_clean

                # Remove light source
                remove_light_source(light_name=f"light{img_count}")
                img_count += 1

    ann_full["images"] = images
    ann_full["annotations"] = ann_image

    # Saving annotation file
    with open(annotations_savepath, "w") as json_file:
        json.dump(ann_full, json_file, indent=4)
    print(f"annotations file saved to: {annotations_savepath}")

    print("total images = ", img_count)
    print("total annotations = ", ann_count)
