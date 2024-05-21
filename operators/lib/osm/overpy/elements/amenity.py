from __future__ import annotations
import itertools
import math
import pprint
import random
from typing import Any, ClassVar, TypeVar
from xml.etree.ElementTree import Element

from .....utils.bgis_utils import DropToGround
from .node import OSMNode
from .way import OSMWay
from .highway import OSMHighway
from .building import OSMPyramidalRoof, OSMFlatRoof
from mathutils import Vector
from mathutils.kdtree import KDTree

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..OSMLibrary import OSMLibrary
    from . import OSMMultiPolygonRelation

import bpy
import bmesh
from bpy.types import Operator, Panel, AddonPreferences
from bpy.props import StringProperty, IntProperty, FloatProperty, BoolProperty, EnumProperty, FloatVectorProperty

xAxis = Vector((1., 0., 0.))
yAxis = Vector((0., 1., 0.))
zAxis = Vector((0., 0., 1.))
#import bmesh


T = TypeVar('T', bound='OSMAmenity')

class OSMAmenity(OSMWay):
    '''A tag for identifying man-made (artificial) structures added to the landscape
    '''
    blender_mesh_name: ClassVar[str] = "Amenity"
    _osm_sub_name: ClassVar[str] = 'amenity'
    _osm_sub_type: ClassVar[str] = ''
    detail_level: ClassVar[int] = -1 #  man made should be an abstract class and should never be used not subclassed

     
    def __str__(self):
        return f"OSMWay of type amenity {self._osm_sub_type} with id: {self._id}, made up of {len(self._node_ids)} nodes(s) and tags:\n{pprint.pformat(self._tags)}"

    def __init__(self, **kwargs):
        super(OSMAmenity,self).__init__(**kwargs)
        self._parts=[]

    @classmethod
    def is_valid_xml(cls, xml_element:Element) -> bool:
        # for c in xml_element.iter('tag'):
        return (super(OSMAmenity, cls).is_valid_xml(xml_element) and 
                any(c.attrib['k'] == cls._osm_sub_name and c.attrib['v'] == cls._osm_sub_type for c in xml_element.iter('tag')))
    # and not any(c.attrib['k'] in ['building','building:part'] for c in xml_element.iter('tag'))

    @classmethod
    def is_valid_json(cls, json_element:dict) -> bool:
        return super(OSMAmenity, cls).is_valid_json(json_element) and json_element.get('tags',{}).get(cls._osm_sub_name,None) == cls._osm_sub_type


    def build_instance(self, geoscn, reproject, ray_caster:DropToGround = None, build_parameters:dict = {}) -> bpy.types.Object|None:
        #Create a new bmesh
        print(f'Building {self._id}')
        bm = bmesh.new()
        self._build_instance(bm, geoscn=geoscn, reproject=reproject, ray_caster=ray_caster, build_parameters = build_parameters)

        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0001)
        mesh = bpy.data.meshes.new(f"{self._id}")
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()#calc_edges=True)
        mesh.validate()
        obj = bpy.data.objects.new(f"{self._id}", mesh)
        geoscn.scn.collection.objects.link(obj)
        obj.select_set(True)
        return obj
            
    def _build_instance(self, bm, geoscn, reproject, ray_caster:DropToGround = None, build_parameters:dict={})->bmesh:
        return

class OSMFountain(OSMAmenity):

    _osm_sub_type: ClassVar[str] = 'fountain'
    detail_level: ClassVar[int] = 3


    def __str__(self):
        return f"OSMAmenity of type Fountain with id: {self._id}, made up of {len(self._node_ids)} nodes(s) and tags:\n{pprint.pformat(self._tags)}"

    def __init__(self, **kwargs):
        super(OSMFountain,self).__init__(**kwargs)


    def preprocess_instance(self, geoscn, ray_caster:DropToGround):
        """
        """
        if self._is_preprocessed:
            return
        
        self.add_reference_to_nodes()
            
        self._is_preprocessed = True
        
        
    def build_instance(self, geoscn, reproject, ray_caster: DropToGround = None, build_parameters: dict = {}) -> Any | None:
        bm = bmesh.new()

        bm =  self._build_instance(bm, geoscn, reproject, ray_caster, build_parameters)

        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0001)
        mesh = bpy.data.meshes.new(f"{self._id}")
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()
        mesh.validate()
        obj = bpy.data.objects.new(f"{self._id}", mesh)
        geoscn.scn.collection.objects.link(obj)
        obj.select_set(True)
        return obj
    
    def _build_instance(self, bm, geoscn, reproject, ray_caster:DropToGround = None, build_parameters:dict={})->bmesh:
        fountain_tiers = build_parameters.get('fountain_tiers', 3)
        fountain_height = build_parameters.get('fountain_height', 1)
        for t in range(fountain_tiers):
            scale_factor = math.pow(0.8,t)
            plant_verts = self.get_vertices(bm, geoscn=geoscn, reproject=reproject, ray_caster=ray_caster)
            for v in plant_verts:
                v.co.z += fountain_height*t
            bottom_face = bm.faces.new(plant_verts)
            # #ensure face is up (anticlockwise order)
            # #because in OSM there is no particular order for closed ways
            bottom_face.normal_update()
            if bottom_face.normal.z > 0:
                bottom_face.normal_flip()

            c = bottom_face.calc_center_median()
            for v in bottom_face.verts:
                v.co = c + scale_factor * (v.co - c)
            
            #Extrude
            

            extrusion = bmesh.ops.extrude_edge_only(bm, edges = bottom_face.edges)
            top_verts = [v for v in extrusion['geom'] if isinstance(v,bmesh.types.BMVert)]
            edges = [v for v in extrusion['geom'] if isinstance(v,bmesh.types.BMEdge)]
            for vert in top_verts:
                vert.co.z += fountain_height
            top_face = bm.faces.new(top_verts)

        #bmesh.ops.triangulate(bm, faces=[bottom_face], quad_method='BEAUTY', ngon_method='BEAUTY')
        return bm
 