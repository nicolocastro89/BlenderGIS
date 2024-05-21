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

from .....utils.bgis_utils import DropToGround, parse_measurement
from .....utils.blender import create_prism_from_vertices
import bpy
import bmesh
from bpy.types import Operator, Panel, AddonPreferences
from bpy.props import StringProperty, IntProperty, FloatProperty, BoolProperty, EnumProperty, FloatVectorProperty

xAxis = Vector((1., 0., 0.))
yAxis = Vector((0., 1., 0.))
zAxis = Vector((0., 0., 1.))
#import bmesh


T = TypeVar('T', bound='OSMLeisure')

class OSMLeisure(OSMWay):
    '''A tag for identifying leisure structures added to the landscape
    '''
    blender_mesh_name: ClassVar[str] = "Leisure"
    _osm_sub_name: ClassVar[str] = 'leisure'
    _osm_sub_type: ClassVar[str] = ''
    detail_level: ClassVar[int] = -1 #  man made should be an abstract class and should never be used not subclassed

     
    def __str__(self):
        return f"OSMWay of type amenity {self._osm_sub_type} with id: {self._id}, made up of {len(self._node_ids)} nodes(s) and tags:\n{pprint.pformat(self._tags)}"

    def __init__(self, **kwargs):
        super(OSMLeisure,self).__init__(**kwargs)
        self._parts=[]

    @classmethod
    def is_valid_xml(cls, xml_element:Element) -> bool:
        # for c in xml_element.iter('tag'):
        return (super(OSMLeisure, cls).is_valid_xml(xml_element) and 
                any(c.attrib['k'] == cls._osm_sub_name and c.attrib['v'] == cls._osm_sub_type for c in xml_element.iter('tag')))
    # and not any(c.attrib['k'] in ['building','building:part'] for c in xml_element.iter('tag'))

    @classmethod
    def is_valid_json(cls, json_element:dict) -> bool:
        return super(OSMLeisure, cls).is_valid_json(json_element) and json_element.get('tags',{}).get(cls._osm_sub_name,None) == cls._osm_sub_type


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

class OSMSwimmingPool(OSMLeisure):

    _osm_sub_type: ClassVar[str] = 'swimming_pool'
    detail_level: ClassVar[int] = 3


    def __str__(self):
        return f"OSMLeisure of type Fountain with id: {self._id}, made up of {len(self._node_ids)} nodes(s) and tags:\n{pprint.pformat(self._tags)}"

    def __init__(self, **kwargs):
        super(OSMSwimmingPool,self).__init__(**kwargs)

    def get_depth(self, build_parameters)->int:
        pool_depth=0
        if "depth" in self._tags:
            pool_depth = parse_measurement(self._tags["depth"])

        else:
            pool_depth = build_parameters.get('pool_depth',3)
            
        return pool_depth
       
    def preprocess_instance(self, geoscn, ray_caster:DropToGround):
        """
        """
        if self._is_preprocessed:
            return
        
        self.add_reference_to_nodes()
            
        self._is_preprocessed = True
        
        
    def build_instance(self, geoscn, reproject, ray_caster: DropToGround = None, build_parameters: dict = {}) -> Any | None:
        negative_collection = bpy.data.collections.get('NegativeCollection')
        bm = bmesh.new()

        bm =  self._build_instance(bm, geoscn, reproject, ray_caster, build_parameters)

        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0001)
        mesh = bpy.data.meshes.new(f"{self._id}")
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()
        mesh.validate()
        obj = bpy.data.objects.new(f"{self._id}", mesh)
        negative_collection.objects.link(obj)
        obj.select_set(True)
        return None
    
    def _build_instance(self, bm, geoscn, reproject, ray_caster:DropToGround = None, build_parameters:dict={})->bmesh:
        plant_verts = self.get_vertices(bm, geoscn=geoscn, reproject=reproject, ray_caster=ray_caster, straight_line_toll=build_parameters.get('straight_line_threshold',4.5))
        pool_depth = self.get_depth(build_parameters=build_parameters)
        create_prism_from_vertices(bm=bm, 
                                   vertices=plant_verts, 
                                   height=2*pool_depth,
                                   ray_caster=ray_caster,
                                   extrusion_axis= build_parameters.get('extrusion_axis', 'Z'))
        # #plant edges
        # shifted_vert = itertools.cycle(plant_verts)
        # try:
        #     next(shifted_vert)
        # except Exception as e:
        #     print(self._id)
        #     print(e)
        #     raise e
            
        # edges=[]
        # for v in zip(plant_verts,shifted_vert):
        #     if v[1] not in [x for y in [a.verts for a in v[0].link_edges] for x in y if x != v[0]]:
        #         edges.append(bm.edges.new(v))
        # try:
        #     face = bm.faces.new(plant_verts)
        # except:
        #     pass

        
        # face.normal_update()
        # if face.normal.z > 0:
        #     face.normal_flip()

        # pool_depth = self.get_depth(build_parameters=build_parameters)
        


        # #Extrude
        
        # if build_parameters.get('extrusion_axis', 'Z') == 'NORMAL':
        #     normal = face.normal
        #     vect = normal * 2 *  pool_depth
        # elif build_parameters.get('extrusion_axis', 'Z') == 'Z':
        #     vect =(0, 0,  2*pool_depth)

        # # extrusion = bmesh.ops.extrude_edge_only(bm, edges = edges)
        # extrusion = bmesh.ops.extrude_face_region(bm, geom=[face]+edges, use_keep_orig=True) #return {'faces': [BMFace]} extrude_edge_only

        # faces = [f for f in extrusion['geom'] if isinstance(f,bmesh.types.BMFace)]
        # verts = [v for v in faces[0].verts]
        # edges = [e for e in faces[0].edges]

        # up_face = face if face.normal.z>0 else faces[0]
        # # extrusion = bmesh.ops.extrude_edge_only(bm, edges = edges)
        # # # bmesh.ops.extrude_face_region(bm, faces=[face]) #return {'faces': [BMFace]} extrude_edge_only
        # # verts = [v for v in extrusion['geom'] if isinstance(v,bmesh.types.BMVert)]
        # # edges = [v for v in extrusion['geom'] if isinstance(v,bmesh.types.BMEdge)]
        
        # if ray_caster:
        #     #Making flat roof
        #     z = max([v.co.z for v in up_face.verts]) + 2*pool_depth #get max z coord
        #     for v in up_face.verts:
        #         v.co.z = z
        # #bmesh.ops.translate(bm, verts=verts, vec=vect)
        bmesh.ops.translate(bm, verts=bm.verts, vec=(0, 0,  -1*pool_depth))
        return bm
    