
from __future__ import annotations
import itertools
from numbers import Number
import os 
import re
import bpy, bmesh
import addon_utils 
from ... import bl_info
from mathutils import Vector
from .bgis_utils import DropToGround
BOC = bpy.ops.curve

def appendObjectsFromFile(filepath, collection, names:list[str]|str):
    with bpy.data.libraries.load(filepath) as (data_from, data_to):
        # a Python list (not a Python tuple!) must be set to <data_to.objects>
        if isinstance(names, list):
            data_to.objects = list(names)
        else:
            object_match = re.compile(names)
            data_to.objects = [obj_name for obj_name in data_from.objects if object_match.match(obj_name) and obj_name.split('.')[0] not in data_to.objects]
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

def createCollection(name, parent=None, hide_viewport=False, hide_select=False, hide_render=False)->bpy.types.Collection:
    collection = bpy.data.collections.get(name,None)
    if collection:
        return collection
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

def convert_obj_to_curve(object):
    override = bpy.context.copy()
    override["selected_objects"] = [object]
    override["selected_editable_objects"] = [object]
    override["active_object"] = object
    override["objects_in_mode"] = [object]
    override["object"] = object
    override["edit_object"] = object
    override['view_layer']=bpy.context.view_layer
    override["mode"] = 'OBJECT'

    area_type = 'VIEW_3D' # change this to use the correct Area Type context you want to process in
    areas = [area for area in bpy.context.window.screen.areas if area.type == area_type]

    if len(areas) <= 0:
        raise Exception(f"Make sure an Area of type {area_type} is open or visible in your screen!")
    override["area"] = areas[0]
    # save and reset state of selection
    #selected_objects = bpy.context.selected_objects
    # active_object = bpy.context.active_object
    # for obj in selected_objects:
    #     obj.select_set(False)
    
    # current_active = bpy.context.view_layer.objects.active
    try:
        with bpy.context.temp_override(**override):  
            bpy.context.view_layer.objects.active = object
            selection = object.select_get()
            object.select_set(True)
            bpy.ops.object.convert(target='CURVE', thickness = 1)
            object.select_set(selection)
            
    except Exception as e:
        print(f'FAILED TO CONVERT TO CURVE')
        print(e)
    finally:
        print('finished')
        #bpy.context.view_layer.objects.active = current_active
        # restore saved state of selection
        # for obj in selected_objects:
        #     obj.select_set(True)
    
def convert_curve_to_obj(object):
    override = bpy.context.copy()
    override["selected_objects"] = [object]
    override["selected_editable_objects"] = [object]
    override["active_object"] = object
    override["objects_in_mode"] = [object]
    override["object"] = object
    override["edit_object"] = object
    override['view_layer']=bpy.context.view_layer
    override["mode"] = 'OBJECT'

    # save and reset state of selection
    #selected_objects = bpy.context.selected_objects
    # active_object = bpy.context.active_object
    # for obj in selected_objects:
    #     obj.select_set(False)
    
    # current_active = bpy.context.view_layer.objects.active
    try:
        with bpy.context.temp_override(**override):  
            bpy.context.view_layer.objects.active = object
            selection = object.select_get()
            object.select_set(True)
            bpy.ops.object.convert(target='MESH')
            object.select_set(selection)
            
    except Exception as e:
        print(e)
    finally:
        print('finished')
        #bpy.context.view_layer.objects.active = current_active
        # restore saved state of selection
        # for obj in selected_objects:
        #     obj.select_set(True)

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
        
def solidify_terrain(elevObj:bmesh.types.Object)->bmesh.types.Object:
    print('solidifying terrain')
    with bpy.context.temp_override(active_object=elevObj, object=elevObj, selected_objects= [elevObj], selected_editable_objects= [elevObj]):
            for modifier in elevObj.modifiers:
                bpy.ops.object.modifier_apply(modifier=modifier.name)
    terrain_mesh = bmesh.new()
    terrain_mesh.from_mesh(elevObj.data)
    extrusion = bmesh.ops.extrude_face_region(terrain_mesh, geom=[f for f in terrain_mesh.faces])
    bbox_corners = min([(elevObj.matrix_world @ Vector(corner)).z for corner in elevObj.bound_box])
    for vertex in [v for v in extrusion['geom'] if isinstance(v,bmesh.types.BMVert)]:
        vertex.co.z = bbox_corners-10
    mesh = bpy.data.meshes.new(f"{elevObj.name}_Solid")
    terrain_mesh.to_mesh(mesh)
    terrain_mesh.free()
    mesh.update()#calc_edges=True)
    mesh.validate()
    obj = bpy.data.objects.new(f"{elevObj.name}_Solid", mesh)
    bpy.context.scene.collection.objects.link(obj)
    print('solidifying terrain finished')
    return obj  

def create_prism_from_vertices(bm: bmesh.types.BMesh, vertices: list[bmesh.types.BMVert], height:Number, ray_caster:DropToGround = None, extrusion_axis = 'Z'):
    shifted_vert = itertools.cycle(vertices)
    try:
        next(shifted_vert)
    except Exception as e:
        print(e)
        raise e
        
    edges=[]
    for v in zip(vertices,shifted_vert):
        if v[1] not in [x for y in [a.verts for a in v[0].link_edges] for x in y if x != v[0]]:
            edges.append(bm.edges.new(v))
    try:
        face = bm.faces.new(vertices)
    except:
        pass

    
    face.normal_update()
    if face.normal.z > 0:
        face.normal_flip()

    
    extrusion = bmesh.ops.extrude_face_region(bm, geom=[face]+edges, use_keep_orig=True) 

    faces = [f for f in extrusion['geom'] if isinstance(f,bmesh.types.BMFace)]
    verts = [v for v in faces[0].verts]
    edges = [e for e in faces[0].edges]

    up_face = face if face.normal.z>0 else faces[0]
    
    if ray_caster:
        #Making flat roof
        z = max([v.co.z for v in up_face.verts]) + height #get max z coord
        for v in up_face.verts:
            v.co.z = z
    else:
        if extrusion_axis == 'NORMAL':
            normal = face.normal
            vect = normal * height
        elif extrusion_axis == 'Z':
            vect =(0, 0,  height)
        bmesh.ops.translate(bm, verts=up_face.verts, vec=vect)

    
                
def polish_mesh(bm: bmesh.types.BMesh):
    # Remove double vertices
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.001) #merge vertices less than 1mm

    bmesh.ops.dissolve_limit(bm, verts=bm.verts, edges=bm.edges)

    override = bpy.context.copy()
    override["selected_objects"] = [bm]
    override["selected_editable_objects"] = [bm]
    override["active_object"] = bm
    override["objects_in_mode"] = [bm]
    override["object"] = bm
    override["edit_object"] = bm
    override['view_layer']=bpy.context.view_layer
    override["mode"] = 'EDIT_MESH'

    screen = bpy.context.window.screen
    # bpy.context.view_layer.objects.active = your_object
    area = next(a for a  in screen.areas if a.type=='VIEW_3D')
    region = next(r for r in area.regions if r.type == 'WINDOW')
    override["area"] = area
    override['region'] = region
    with bpy.context.temp_override(**override):    
        current_mode = bpy.context.object.mode
        bpy.ops.object.mode_set(mode='EDIT') 
        bpy.ops.mesh.select_all(action='SELECT')           
        bpy.ops.mesh.select_interior_faces()
        
        bmesh.ops.delete(bm, geom = [f for f in bm.faces if f.select], context='FACES_ONLY')# 3 is only faces
        bpy.ops.object.mode_set(mode=current_mode)

def polish_object(obj):

    override = bpy.context.copy()
    override["selected_objects"] = [obj]
    override["selectable_objects"] = [obj]
    override["selected_editable_objects"] = [obj]
    override["editable_objects"] = [obj]
    override["active_object"] = obj
    override["objects_in_mode"] = [obj]
    override["object"] = obj
    override["edit_object"] = obj
    override['view_layer']=bpy.context.view_layer
    override["mode"] = 'EDIT_MESH'

    screen = bpy.context.window.screen
    # bpy.context.view_layer.objects.active = your_object
    area = next(a for a  in screen.areas if a.type=='VIEW_3D')
    region = next(r for r in area.regions if r.type == 'WINDOW')
    override["area"] = area
    override['region'] = region
    with bpy.context.temp_override(**override):  
        current_mode = bpy.context.object.mode  
        bpy.ops.object.mode_set(mode='EDIT')
        try:
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.remove_doubles(threshold=0.1)

            bpy.ops.mesh.select_all(action='DESELECT')
        except:
            pass
            # print('Failed to select all')
            # print(f'{bpy.context.selected_objects}')
            # print(f'{bpy.context.selected_editable_objects}')
            # print(f'{bpy.context.active_object}')
            # print(f'{bpy.context.object} {type(bpy.context.object)}')
        try:
            bpy.ops.mesh.select_interior_faces()
            bpy.ops.mesh.delete(type='FACE')
        except:
            pass
            # print('Failed to delete interio faces')
            # print(f'{bpy.context.selected_objects}')
            # print(f'{bpy.context.selected_editable_objects}')
            # print(f'{bpy.context.active_object}')
            # print(f'{bpy.context.object} {type(bpy.context.object)}')
        
        bpy.ops.object.mode_set(mode=current_mode)