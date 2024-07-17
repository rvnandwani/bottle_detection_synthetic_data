from datetime import datetime
import numpy as np
import os
import json
import glob
import bpy
from blender_helper import clean_enviroment,add_hdri_background,setup_new_camera,set_resolution,add_light_source,add_ground_plane,jitter_position_xy,calculate_2d_bounding_box,render_image,remove_occluded_annotations,remove_light_source,remove_camera
import random
import math

def check_rectangle_position(in_rect, out_rect):
    x1, y1, x2, y2 = in_rect[0],in_rect[1],in_rect[2],in_rect[3]
    X1, Y1, X2, Y2 = out_rect[0],out_rect[1],out_rect[2],out_rect[3]

    def is_point_inside(px, py):
        return X1 <= px <= X2 and Y1 <= py <= Y2

    corners_inside = [
        is_point_inside(x1, y1),
        is_point_inside(x1, y2),
        is_point_inside(x2, y1),
        is_point_inside(x2, y2)
    ]
    if all(corners_inside):
        return in_rect
    elif any(corners_inside):
        if x1<X1:
            x1=X1
        if y1<Y1:
            y1=Y1
        if x2>X2:
            x2=X2
        if y2>Y2:
            y2=Y2
        return [x1,y1,x2,y2]
    else:
        return None
    
def random_bool():
    return random.choice([True, False])

def Get3DObject(assetdir):
    labelpath={}
    for asset in os.listdir(assetdir):
        labelpath[asset] = sorted(glob.glob(os.path.join(assetdir, asset, "**", "*.obj"),recursive=True))
    return labelpath

def GetCameraPosition():
    x = random.uniform(0.3,1.3) if random_bool() else random.uniform(0.3,1.3)*-1
    y = random.uniform(0.3,1.3) if random_bool() else random.uniform(0.3,1.3)*-1
    z = random.uniform(0.1,0.3)
    
    r_x = round(np.deg2rad(math.degrees(math.atan2(math.sqrt(x**2+y**2),z-0.05))),3)
    r_y = round(np.deg2rad(random.uniform(-10.0,10.0)),3)
    r_z = round(np.deg2rad(90+math.degrees(math.atan2(y,x))),3)
    
    return ((round(x,3),round(y,3),round(z,3)),(r_x,r_y,r_z))

    
if __name__ == "__main__":

    assets_dir = "./assets"
    hdris_dir = "./hdris"
    
    image_size = (640, 640)
    set_resolution(image_size)

    images = []
    annotations_savepath = "example_annotations_auto.json"
    
    ann_count = 0
    annotations = []
    backgrounds = sorted(glob.glob(os.path.join(hdris_dir, "*.hdr")))
    assets = Get3DObject(assetdir=assets_dir)
    img_count = 0
    ann_full = {"info" : 
        {"description" : "Bottle and Can detection" ,
         "date_time" : str(datetime.now())},
        "categories" : [
            {
                "id" : 1,
                "name" : "bottle",
            },
            {
                "id" : 2,
                "name" : "can",
            }
            
        ]}
    # for _ in range(10):
    ann_image = []
    images = []
    for bgr in backgrounds:
      print(f"#### Background : {bgr} ####")
      
      
      cameras = [GetCameraPosition() for i in range(7)]
      
      for cam_loc in cameras:
        clean_enviroment()
        add_hdri_background(bgr)
        add_light_source(light_name = f"light{img_count}")
        loc,rot = cam_loc
        image_savepath = f"Image_{img_count}.png"
        annotations_savepath = f"annotations/example_annotations_auto_{img_count}.json"
        print(f"!!!!! Camera for {image_savepath} at {loc} and {rot} !!!!!")
        cam = setup_new_camera(img_count,location=loc,rotation_euler=rot)
        
        ann_temp = []
        for category in assets:
            for obj in assets[category]:
                # for i in range(random.randint(2,4)):
                for i in range(2):
                    
                    bpy.ops.import_scene.obj(filepath = obj)
                    bottle = bpy.context.selected_objects[-1]
                    bottle.scale = bottle.scale * .1
                    jitter_position_xy(bottle)
                    bbox,center_of_volume = calculate_2d_bounding_box(bottle,cam)
                    ann_temp.append({"class" :obj , "bbox" : bbox , "center_of_volume" : center_of_volume})
                    
        render_image("train_images/"+image_savepath)
        images.append({"id":img_count,"width":image_size[0],"height":image_size[1],"filename":image_savepath,"date_saved":datetime.now()})
        annotations_clean,ann_count = remove_occluded_annotations(ann_temp,ann_count,img_count,image_savepath,image_size)
        ann_image+=annotations_clean 
        with open(annotations_savepath, "w") as json_file:
            json.dump(annotations_clean, json_file, indent=4)
        print(f"indiviual annotations file saved to: {annotations_savepath}")
        remove_light_source(light_name = f"light{img_count}")
        # remove_camera(cam.name)
        img_count+=1
    ann_full["images"] = images
    ann_full["annotations"] = ann_image
    
    with open("example_annotations_auto.json", "w") as json_file:
        json.dump(ann_full, json_file, indent=4)
    print(f"annotations file saved to: example_annotations_auto.json")
    
    print("total images = ",img_count)
    print("total annotations = ",ann_count)