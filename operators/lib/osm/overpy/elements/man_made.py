from __future__ import annotations
import pprint
import random
from typing import Any, ClassVar, TypeVar
from xml.etree.ElementTree import Element

from .....utils.bgis_utils import DropToGround
from .node import OSMNode
from .way import OSMWay
from .highway import OSMHighway
from mathutils import Vector

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


T = TypeVar('T', bound='OSMManMade')

class OSMManMade(OSMWay):
    '''A tag for identifying man-made (artificial) structures added to the landscape
    '''

    _osm_sub_name: ClassVar[str] = 'man_made'
    _osm_sub_type: ClassVar[str] = ''
    detail_level: ClassVar[int] = -1 #  man made should be n abstract class and should never be used not subclassed

     
    def __str__(self):
        return f"OSMWay of type man_made {self.osm_sub_type} with id: {self._id}, made up of {len(self._node_ids)} nodes(s) and tags:\n{pprint.pformat(self._tags)}"

    def __init__(self, **kwargs):
        super(OSMManMade,self).__init__(**kwargs)
        self._parts=[]

    @classmethod
    def is_valid_xml(cls, xml_element:Element) -> bool:
        # for c in xml_element.iter('tag'):
        return super(OSMManMade, cls).is_valid_xml(xml_element) and any(c.attrib['k'] == cls._osm_sub_name and c.attrib['v'] == cls._osm_sub_type for c in xml_element.iter('tag'))

    @classmethod
    def is_valid_json(cls, json_element:dict) -> bool:
        return super(OSMManMade, cls).is_valid_json(json_element) and json_element.get('tags',{}).get(cls._osm_sub_name,None) == cls._osm_sub_type


    def build_instance(self, geoscn, reproject, ray_caster:DropToGround = None, build_parameters:dict = {}) -> bpy.types.Object|None:
        #Create a new bmesh
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


    
class OSMBridge(OSMManMade):

    _osm_sub_type: ClassVar[str] = 'bridge'
    detail_level: ClassVar[int] = 3

    _structure_of: OSMHighway 

    def __str__(self):
        return f"OSMManMade of type Bridge with id: {self._id}, made up of {len(self._node_ids)} nodes(s) and tags:\n{pprint.pformat(self._tags)}"

    def __init__(self, **kwargs):
        super(OSMBridge,self).__init__(**kwargs)


    def preprocess_instance(self):
        """
        """
        if self._is_preprocessed:
            return
        
        self.add_reference_to_nodes()
            
        self.assign_to_highway()
        self._is_preprocessed = True
        
    
    def assign_to_highway(self):
        """Find the highway of the bridge
        """
        #No relation for now
        #Find which building the part belongs to
        # from .relation import OSMRelation
        # relation = next((r for r in self._library.get_elements(OSMRelation).values() if self._id in r.members), None)
        # if relation:
        #     outline_id = relation.outline
        #     outline = self._library.get_element_by_id(outline_id)
        #     if isinstance(outline, OSMBuilding):
        #         outline.add_part(self)
        #     return


        # If no relation is found find the building which uses all/most of the nodes in the part 
        
        shared_by = {} # Dictionary of building Ids and how many nodes are encompassed by it

        free_node = None
        for node in self._nodes:
            referenced_by = node.get_referenced_from().get(OSMHighway,set())
            if node._id in referenced_by:
                for ref in referenced_by:
                    shared_by[ref] = shared_by.get(ref, 0) + 1
            else:
                if free_node is None:
                    free_node = node._id
        
        # Find the building parts with the most candidates
        max_references = max((len(s) for s in shared_by.values()), default = None)
        candidates = [b for b in shared_by.keys() if len(shared_by[b])==max_references] if max_references else []

        # If not all nodes are withing a building check if the parts is ray cast within a building
        if free_node:
            if len(candidates) == 1:
                # To save time we won't check if all free OSM nodes of <part>
                # are located inside the building
                self._library.get_element_by_id(candidates[0]).add_part(self)
            else:
                # Take the first encountered free node <freeNode> and
                # calculated if it is located inside any building from <self.buildings>
                bvhTree = self._library.bvh_tree
                coords = next(n for n in self._nodes if n._id == free_node)
                # Cast a ray from the point with horizontal coords equal to <coords> and
                # z = -1. in the direction of <zAxis>
                buildingIndex = bvhTree.ray_cast((coords._lat, coords._lon, -1.), zAxis)[2]
                if buildingIndex is not None:
                    # we consider that <part> is located inside <buildings[buildingIndex]>
                    self._library.get_element_by_id(self._library.bvh_tree_index[buildingIndex]).add_part(self)
        else:
            # all OSM nodes of <part> are used by one or more buildings from <self.buildings>
            # the case numCandidates > 1 probably means some weird configuration, so skip that <part>
            if len(candidates) == 1:
                self._library.get_element_by_id(candidates[0]).add_part(self)
            
    def build_instance(self, geoscn, reproject, ray_caster: DropToGround = None, build_parameters: dict = {}) -> Any | None:
        return 
    
    def _build_instance(self, bm, geoscn, reproject, ray_caster:DropToGround = None, build_parameters:dict={})->bmesh:
        verts = self.get_vertices(bm, geoscn=geoscn, reproject=reproject, ray_caster=ray_caster)
        
        
        
        face = bm.faces.new(verts)
        #ensure face is up (anticlockwise order)
        #because in OSM there is no particular order for closed ways
        face.normal_update()
        if face.normal.z < 0:
            face.normal_flip()

        offset = build_parameters.get('bridge_height', 5)
        
        #Extrude
        
        if build_parameters.get('extrusion_axis', 'Z') == 'NORMAL':
            normal = face.normal
            vect = normal * offset
        elif build_parameters.get('extrusion_axis', 'Z') == 'Z':
            vect = (0, 0, offset)

        faces = bmesh.ops.extrude_discrete_faces(bm, faces=[face]) #return {'faces': [BMFace]}
        verts = faces['faces'][0].verts
        if ray_caster:
            #Making flat roof
            z = max([v.co.z for v in verts]) + offset #get max z coord
            for v in verts:
                v.co.z = z
        else:
            bmesh.ops.translate(bm, verts=verts, vec=vect)


        return bm
    
