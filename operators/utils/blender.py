
from __future__ import annotations
import os 
import re
import bpy, bmesh
import addon_utils 
from ... import bl_info
from mathutils import Vector

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