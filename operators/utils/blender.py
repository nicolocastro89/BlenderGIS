
from __future__ import annotations
import os 
import re
import bpy, bmesh
import addon_utils 
from ... import bl_info
from mathutils import Vector
BOC = bpy.ops.curve

def appendObjectsFromFile(filepath, collection, names:list[str]|str):
    with bpy.data.libraries.load(filepath) as (data_from, data_to):
        # a Python list (not a Python tuple!) must be set to <data_to.objects>
        if isinstance(names, list):
            data_to.objects = list(names)
        else:
            object_match = re.compile(names)
            data_to.objects = [obj_name for obj_name in data_from.objects if object_match.match(obj_name) and obj_name not in data_to.objects]
    if collection:
        # append the objects to the Blender scene
        for obj in data_to.objects:
            if obj:
                    collection.objects.link(obj)
                    obj.select_set(False)
    # return the appended Blender objects
    return data_to.objects

def appendObjectsFromAssets(assetFile:str, collection, names:list[str]|str):
    for mod in addon_utils.modules():
        if mod.bl_info['name'] == bl_info['name']:
            dirpath = os.path.dirname(mod.__file__)
        else:
            pass
    if dirpath:
        return appendObjectsFromFile(os.path.join(dirpath,'assets',assetFile), collection, names)
    return []

def createCollection(name, parent=None, hide_viewport=False, hide_select=False, hide_render=False):
    collection = bpy.data.collections.new(name)
    if not parent:
        parent = bpy.context.scene.collection
    parent.children.link(collection)
    collection.hide_viewport = hide_viewport
    collection.hide_select = hide_select
    collection.hide_render = hide_render
    return collection

def almost_overlapping(point1: Vector, point2: Vector, max_distance = 0.5):
        return xy_distance(point1,point2)<=max_distance
    
def xy_distance(point1: Vector, point2: Vector):
    return (point1.xy-point2.xy).magnitude

def add_spline_point(end_point, location):
    BOC.select_all(action='DESELECT')
    select_spline_point(end_point) # the new point will be connected here
    BOC.vertex_add(location=location)

def select_spline_point(spline, point):
    if isinstance(spline.points[point], bpy.types.BezierSplinePoint):
        spline.points[point].select_control_point = True
        spline.points[point].select_left_handle = True
        spline.points[point].select_right_handle = True
    else:
        spline.points[point].select = True

def merge_splines(curve_object, spline1, control_point_1, spline2, control_point_2):
    spline1_idx = list(curve_object.data.splines).index(spline1)
    spline2_idx = list(curve_object.data.splines).index(spline2)
    override = bpy.context.copy()
    override["selected_objects"] = [curve_object]
    override["selected_editable_objects"] = [curve_object]
    override["active_object"] = curve_object
    override["objects_in_mode"] = [curve_object]
    override["object"] = curve_object
    override["edit_object"] = curve_object
    override['view_layer']=bpy.context.view_layer
    override["mode"] = 'EDIT_CURVE'

    screen = bpy.context.window.screen
    # bpy.context.view_layer.objects.active = your_object
    area = next(a for a  in screen.areas if a.type=='VIEW_3D')
    region = next(r for r in area.regions if r.type == 'WINDOW')
    override["area"] = area
    override['region'] = region
    current_active = bpy.context.view_layer.objects.active
    try:
        with bpy.context.temp_override(**override):                
            bpy.context.view_layer.objects.active = curve_object
            bpy.ops.object.mode_set(mode='EDIT')
            
            bpy.ops.curve.select_all(action='DESELECT')

            select_spline_point(curve_object.data.splines[spline1_idx], control_point_1)
            select_spline_point(curve_object.data.splines[spline2_idx], control_point_2)
            bpy.ops.curve.make_segment()
    except Exception as e:
        print(e)
    finally:
        bpy.context.view_layer.objects.active = current_active
        bpy.ops.object.mode_set(mode = 'OBJECT')
        # control_point_1.select_control_point = True
        # control_point_1.select_left_handle = True
        # control_point_1.select_right_handle = True

        # control_point_2.select_control_point = True
        # control_point_2.select_left_handle = True
        # control_point_2.select_right_handle = True

        # bpy.ops.curve.bpy.ops.curve()
            

        # if area.type == 'VIEW_3D':
        #     override["area"] = area
        #     with bpy.context.temp_override(**override):
        #         mode = bpy.context.active_object.mode
        #         if mode == 'OBJECT':
        #             bpy.ops.object.mode_set(mode='EDIT')
        #         BOC.select_all(action='DESELECT')
        #         select_spline_point(control_point_1)
        #         select_spline_point(control_point_2)
        #         BOC.bpy.ops.curve()

        #         if mode=='OBJECT':
        #             bpy.ops.object.mode_set(mode=mode)
    