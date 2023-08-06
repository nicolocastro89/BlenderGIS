from __future__ import annotations
import itertools
import pprint
import random
from typing import Any, ClassVar, TypeVar
from xml.etree.ElementTree import Element

from .....utils.bgis_utils import DropToGround
from .node import OSMNode
from .way import OSMWay
from .highway import OSMHighway
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


T = TypeVar('T', bound='OSMManMade')

class OSMManMade(OSMWay):
    '''A tag for identifying man-made (artificial) structures added to the landscape
    '''

    _osm_sub_name: ClassVar[str] = 'man_made'
    _osm_sub_type: ClassVar[str] = ''
    detail_level: ClassVar[int] = -1 #  man made should be n abstract class and should never be used not subclassed

     
    def __str__(self):
        return f"OSMWay of type man_made {self._osm_sub_type} with id: {self._id}, made up of {len(self._node_ids)} nodes(s) and tags:\n{pprint.pformat(self._tags)}"

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

class OSMBreakwater(OSMManMade):

    _osm_sub_type: ClassVar[str] = 'breakwater'
    detail_level: ClassVar[int] = 3


    def __str__(self):
        return f"OSMManMade of type Breakwater with id: {self._id}, made up of {len(self._node_ids)} nodes(s) and tags:\n{pprint.pformat(self._tags)}"

    def __init__(self, **kwargs):
        super(OSMBreakwater,self).__init__(**kwargs)


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
    
    def _build_instance(self, bm, geoscn, reproject, ray_caster:DropToGround = None, supported_highway_nodes=None, kd_tree:KDTree = None, build_parameters:dict={})->bmesh:
        plant_verts = self.get_vertices(bm, geoscn=geoscn, reproject=reproject, ray_caster=ray_caster)
        

        # shifted_verts = itertools.cycle(plant_verts)
        # next(shifted_verts)

        # edges = [bm.edges.new(v) for v in zip(plant_verts, shifted_verts)]

        bottom_face = bm.faces.new(plant_verts)
        # #ensure face is up (anticlockwise order)
        # #because in OSM there is no particular order for closed ways
        bottom_face.normal_update()
        if bottom_face.normal.z > 0:
            bottom_face.normal_flip()

        offset = build_parameters.get('breakwater_height', 10)
        # bmesh.ops.connect_vert_pair(bm, verts=bottom_face.verts)
        #Extrude
        

        extrusion = bmesh.ops.extrude_edge_only(bm, edges = bottom_face.edges)
        top_verts = [v for v in extrusion['geom'] if isinstance(v,bmesh.types.BMVert)]
        edges = [v for v in extrusion['geom'] if isinstance(v,bmesh.types.BMEdge)]
        for vert in top_verts:
            vert.co.z = offset
        top_face = bm.faces.new(top_verts)

        #bmesh.ops.triangulate(bm, faces=[bottom_face], quad_method='BEAUTY', ngon_method='BEAUTY')
        return bm
    

class OSMBridge(OSMManMade):

    @property
    def layer(self):
        return int(self._tags.get('layer',0))
    
    _osm_sub_type: ClassVar[str] = 'bridge'
    detail_level: ClassVar[int] = 3

    _structure_of: OSMHighway 

    def __str__(self):
        return f"OSMManMade of type Bridge with id: {self._id}, made up of {len(self._node_ids)} nodes(s) and tags:\n{pprint.pformat(self._tags)}"

    def __init__(self, **kwargs):
        super(OSMBridge,self).__init__(**kwargs)


    def preprocess_instance(self, geoscn, ray_caster:DropToGround):
        """
        """
        if self._is_preprocessed:
            return
        
        self.add_reference_to_nodes()
            
        # self.assign_to_highway()
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
            referenced_by = node.get_referenced_from(OSMHighway, True)
            if node._id in referenced_by:
                for ref in referenced_by:
                    shared_by[ref] = shared_by.get(ref, 0) + 1
            else:
                if free_node is None:
                    free_node = node._id
        
        max_references = max((len(s) for s in shared_by.values()), default = None)
        candidates = [b for b in shared_by.keys() if len(shared_by[b])==max_references] if max_references else []

        # If not all nodes are withing a building check if the parts is ray cast within a building
        if free_node:
            if len(candidates) > 0:
                # To save time we won't check if all free OSM nodes of <part>
                # are located inside the building
                self._library.get_element_by_id(candidates[0]).add_part(self)
            else:
                # Take the first encountered free node <freeNode> and
                # calculated if it is located inside any building from <self.buildings>

                (bvhTree, bvh_tree_index) = self._library.get_bvh_tree([OSMHighway,*OSMHighway.__subclasses__()])
                coords = next(n for n in self._nodes if n._id == free_node)
                # Cast a ray from the point with horizontal coords equal to <coords> and
                # z = -1. in the direction of <zAxis>
                buildingIndex = bvhTree.ray_cast((coords._lat, coords._lon, -1.), zAxis)[2]
                if buildingIndex is not None:
                    # we consider that <part> is located inside <buildings[buildingIndex]>
                    self._library.get_element_by_id(bvh_tree_index[buildingIndex]).add_part(self)
        else:
            # all OSM nodes of <part> are used by one or more buildings from <self.buildings>
            # the case numCandidates > 1 probably means some weird configuration, so skip that <part>
            if len(candidates) == 1:
                self._library.get_element_by_id(candidates[0]).add_part(self)
            
    def build_instance(self, geoscn, reproject, ray_caster: DropToGround = None, build_parameters: dict = {}) -> Any | None:
        return 
    
    def _build_instance(self, bm, geoscn, reproject, ray_caster:DropToGround = None, supported_highway_nodes=None, kd_tree:KDTree = None, build_parameters:dict={})->bmesh:
        from mathutils.bvhtree import BVHTree
        verts = self.get_vertices(bm, geoscn=geoscn, reproject=reproject, ray_caster=ray_caster, subdivision_size=10)
        
        
        shifted_verts = itertools.cycle(verts)
        next(shifted_verts)

        edges = [bm.edges.new(v) for v in zip(verts, shifted_verts)]

        #geom = bmesh.ops.beautify_fill(bm, edges=edges)
        # faces = [v for v in geom['geom'] if isinstance(v, bmesh.types.BMFace)]
        top_faces = bm.faces.new(verts)
        # #ensure face is up (anticlockwise order)
        # #because in OSM there is no particular order for closed ways
        for face in [top_faces]:
            face.normal_update()
            if face.normal.z < 0:
                face.normal_flip()

        bvh_tree = BVHTree.FromBMesh(bm)
        direction = Vector((0,0,-1))
        for vert in verts:
            _, match_idx, _ = kd_tree.find(co = vert.co, 
                                            filter = lambda idx: bvh_tree.ray_cast(supported_highway_nodes[idx].co, direction)[0] is not None)
            vert.co.z = supported_highway_nodes[match_idx].co.z

        offset = build_parameters.get('bridge_height', 5)
        
        

        extrusion = bmesh.ops.extrude_edge_only(bm, edges = top_faces.edges)
        lower_verts = [v for v in extrusion['geom'] if isinstance(v,bmesh.types.BMVert)]
        edges = [v for v in extrusion['geom'] if isinstance(v,bmesh.types.BMEdge)]
        for vert in lower_verts:
            vert.co.z -= offset
        lower_faces = bm.faces.new(lower_verts)
        
        #bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
        #bmesh.ops.connect_verts_nonplanar(bm, faces = [top_faces,lower_faces])
        # geom = bmesh.ops.triangle_fill(bm, edges=edges)
        # if ray_caster:
        #     #Making flat roof
        #     z = max([v.co.z for v in verts]) + offset #get max z coord
        #     for v in verts:
        #         v.co.z = z
        # else:
        #     bmesh.ops.translate(bm, verts=verts, vec=vect)
        supports = self.get_referenced_from(OSMBridgeSupport)
        if supports:
            kd = KDTree(len(lower_verts))
            for i, v in enumerate(lower_verts):
                kd.insert(Vector((v.co[0], v.co[1], 0)), i)
            kd.balance()
            for structure in self.get_referenced_from(OSMBridgeSupport):
                self._library.get_element_by_id(structure)._build_instance(bm, geoscn, reproject, ray_caster, lower_verts, kd, build_parameters)
        bmesh.ops.triangulate(bm, faces=bm.faces, quad_method='BEAUTY', ngon_method='BEAUTY')
        return bm
    
class OSMBridgeSupport(OSMManMade):

    _osm_sub_name: ClassVar[str] = 'bridge:support'
    detail_level: ClassVar[int] = 3

    _support_of: OSMHighway 

    @classmethod
    def is_valid_xml(cls, xml_element:Element) -> bool:
        # for c in xml_element.iter('tag'):
        return super(OSMManMade, cls).is_valid_xml(xml_element) and any(c.attrib['k'] == cls._osm_sub_name for c in xml_element.iter('tag'))

    @classmethod
    def is_valid_json(cls, json_element:dict) -> bool:
        return super(OSMManMade, cls).is_valid_json(json_element) and cls._osm_sub_name in json_element.get('tags',{})


    def __str__(self):
        return f"OSMWay of type BridgeSupport with id: {self._id}, made up of {len(self._node_ids)} nodes(s) and tags:\n{pprint.pformat(self._tags)}"

    def __init__(self, **kwargs):
        super(OSMBridgeSupport,self).__init__(**kwargs)


    def preprocess_instance(self, geoscn, ray_caster:DropToGround):
        """
        """
        if self._is_preprocessed:
            return
        
        self.add_reference_to_nodes()

        self.assign_to_best_overlap(OSMBridge,allow_multiple=True)   
        #self.assign_to_bridge()
        self._is_preprocessed = True
        
    
    def assign_to_bridge(self):
        """Find the bridge of the support
        """
        
        shared_by = {} # Dictionary of building Ids and how many nodes are encompassed by it

        free_node = None
        for node in self._nodes:
            referenced_by = node.get_referenced_from(OSMBridge)
            if referenced_by:
                for ref in referenced_by:
                    shared_by[ref] = shared_by.get(ref, 0) + 1
            else:
                if free_node is None:
                    free_node = node._id
        
        # Find the building parts with the most candidates
        max_references = max((s for s in shared_by.values()), default = None)
        candidates = [b for b in shared_by.keys() if shared_by[b]==max_references] if max_references else []

        # If not all nodes are withing a building check if the parts is ray cast within a building
        if free_node:
            if len(candidates) == 1:
                # To save time we won't check if all free OSM nodes of <part>
                # are located inside the building
                self._library.get_element_by_id(candidates[0]).add_reference(OSMBridgeSupport,self._id)
            else:
                # Take the first encountered free node <freeNode> and
                # calculated if it is located inside any building from <self.buildings>
                (bvhTree,bvh_tree_index) = self._library.get_bvh_tree(OSMBridge)
                shared_by = {}
                for node in self.nodes:
                    buildingIndex = bvhTree.ray_cast((node._lat, node._lon, -1.), zAxis)[2]
                    if buildingIndex is not None:
                        shared_by[bvh_tree_index[buildingIndex]] = shared_by.get(bvh_tree_index[buildingIndex], 0) + 1
                    
                max_references = max((len(s) for s in shared_by.values()), default = None)
                candidates = [b for b in shared_by.keys() if len(shared_by[b])==max_references] if max_references else []
                    
                # Cast a ray from the point with horizontal coords equal to <coords> and
                # z = -1. in the direction of <zAxis>
               
                    # we consider that <part> is located inside <buildings[buildingIndex]>
                for candidate in candidates:
                    self._library.get_element_by_id(candidate).add_reference(OSMBridgeSupport,self._id)
        else:
            # all OSM nodes of <part> are used by one or more buildings from <self.buildings>
            # the case numCandidates > 1 probably means some weird configuration, so skip that <part>
            if len(candidates) == 1:
                self._library.get_element_by_id(candidates[0]).add_reference(OSMBridgeSupport,self._id)
            
    def build_instance(self, geoscn, reproject, ray_caster: DropToGround = None, build_parameters: dict = {}) -> Any | None:
        return 
    
    def _build_instance(self, bm, geoscn, reproject, ray_caster:DropToGround = None, supported_highway_nodes=None, kd_tree:KDTree = None, build_parameters:dict={})->bmesh:
        verts = self.get_vertices(bm, geoscn=geoscn, reproject=reproject, ray_caster=ray_caster)
        
        
        shifted_verts = itertools.cycle(verts)
        next(shifted_verts)
        #edges =[bm.edges.new(v) for v in zip(verts, shifted_verts)] 
        face = bm.faces.new(verts)
        #ensure face is up (anticlockwise order)
        #because in OSM there is no particular order for closed ways
        # face.normal_update()
        # if face.normal.z < 0:
        #     face.normal_flip()

        #extrusion_return = bmesh.ops.extrude_face_region(bm, geom=[face]) #return {'faces': [BMFace]}
        extrusion = bmesh.ops.extrude_edge_only(bm, edges = face.edges)
        top_verts = [v for v in extrusion['geom'] if isinstance(v, bmesh.types.BMVert)]
        #[v for f in faces['faces'] for v in f.verts]

        for vert in top_verts:
            _, match_idx, _ = kd_tree.find(vert.co)
            desired_z = supported_highway_nodes[match_idx].co.z-vert.co.z
            bmesh.ops.translate(bm, vec=Vector((0,0,desired_z)), verts=[vert])
        
        top_face = bm.faces.new(top_verts)
        self.is_built = True
        return bm
    
