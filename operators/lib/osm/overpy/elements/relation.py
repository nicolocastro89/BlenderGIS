from __future__ import annotations
from enum import Enum, unique
import json
import itertools
import pprint
import random
from typing import Callable, ClassVar, OrderedDict
from xml.etree.ElementTree import Element

from .element import OSMElement
from .way import OSMWay 
from .building import OSMBuilding, OSMBuildingPart
from .node import OSMNode
from .....utils.bgis_utils import DropToGround
from mathutils import Vector

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..OSMLibrary import OSMLibrary

import bpy
import bmesh
from bpy.types import Operator, Panel, AddonPreferences
from bpy.props import StringProperty, IntProperty, FloatProperty, BoolProperty, EnumProperty, FloatVectorProperty

xAxis = Vector((1., 0., 0.))
yAxis = Vector((0., 1., 0.))
zAxis = Vector((0., 0., 1.))

class OSMRelationMember(object):

    type: str
    reference: int
    role: str

    # @property
    # def element(self) -> OSMElement:
    #     if self._element is None:
    #         self._element = self._library.get_element_by_id(self.reference)
    #     return self. _element

    element: OSMElement = None
    # _library: OSMLibrary

    def __str__(self):
        return f"OSMRelationMember of type {self.type}, with role:{self.role} and referencing OSM Element with id: {self.reference}. (Element is {'not ' if self.element is None else ''}loaded)"
    
    def __init__(self, **kwargs) -> None:
        # self._library = library
        self.type = kwargs.get('type', None)
        self.reference = int(kwargs.get('ref', None))
        self.role = kwargs.get('role', None)

    @classmethod
    def load(cls, element) -> 'OSMRelation':
        kwargs = cls._create_init_dict(element=element)
        return cls(**kwargs)

    @classmethod
    def load_from_xml(cls, xml_element: Element) -> 'OSMRelation':
        init_dict = cls._create_init_dict_from_xml(xml_element=xml_element)
        return cls(**init_dict)

    @classmethod
    def load_from_json(cls, json_element: dict) -> 'OSMRelation':
        return cls(**json_element)

    @classmethod
    def _create_init_dict(cls, element) -> dict:
        if isinstance(element, dict) and cls.is_valid_json(element):
            return cls._create_init_dict_from_json(element)
        if isinstance(element, Element) and cls.is_valid_xml(element):
            return cls._create_init_dict_from_xml(element)

    @classmethod
    def _create_init_dict_from_xml(cls, xml_element: Element) -> dict:
        return {
            'type': xml_element.attrib['type'],
            'ref': int(xml_element.attrib['ref']) if 'ref' in xml_element.attrib else None,
            'role': xml_element.attrib['role'],
        }

    @classmethod
    def _create_init_dict_from_json(cls, json_element: dict) -> dict:
        return json_element

    @property
    def nodes(self)->list[OSMNode]:
        if self.element is None:
            print(f'Failed to retrieve nodes for {self}')
            raise ValueError(f'Failed to retrieve nodes for {self}')
                
        return self.element.nodes
    
    def get_nodes(self) -> "list['OSMNode']":
        return self.nodes


class OSMRelation(OSMElement):
    ''' A relation is an element with at least the tag type=* and a group of members 
    which is an ordered list of one or more nodes, ways and/or relations. 
    It is used to define logical or geographic relationships between these different objects. 
    A member of a relation can optionally have a role which describes the part that a particular 
    feature plays within a relation.
    '''

    _osm_name: ClassVar[str] = 'relation'
    _osm_relation_type: ClassVar[str] = None
    _osm_relation_tags: ClassVar["list['str']"] = []
    detail_level: ClassVar[int] = 1

    _node_ids: "list['int']" = []
    _nodes: "list['OSMNode']" = []

    members: OrderedDict[int, OSMRelationMember]

    @property
    def nodes(self)->list[OSMNode]:
        self._nodes = []
        try:
            for (memberId, member) in self.members.items():
                if member.element is None:
                    member.element = self._library.get_element_by_id(memberId)
                
                self._nodes.extend(member.nodes if member.element is not None else [])
        except Exception as e:
            print(f'Failed to retrieve nodes for {self}.\n{e}')
            raise e
                
        return self._nodes
    
    @property
    def outline(self)->int:
        outline = next((member for member in self.members.values() if member.role=='outline'), None)

        return outline.reference if outline is not None else None

    def __str__(self):
        return f"OSMRelation with id: {self._id}, made up of {len(self.members)} members(s) and tags:\n{pprint.pformat(self._tags)}"

    def __init__(self, **kwargs) -> None:
        super(OSMRelation, self).__init__(**kwargs)
        self.members = OrderedDict((m.reference,m) for m in kwargs.get('members'))
        return

    @classmethod
    def is_valid(cls, element) -> bool:
        return super(OSMRelation, cls).is_valid(element)

    @classmethod
    def is_valid_xml(cls, xml_element: Element) -> bool:
        #If a relation subtype (e.g. multipolygon) is present will check it
        relation_sub_type = any(
            child.get('k', None) == 'type'
            and child.get('v', None) == cls._osm_relation_type
            for child in xml_element.findall('tag')) if cls._osm_relation_type else True
        
        element_tags = [child.get('k', None) for child in xml_element.findall('tag')]
        relation_tags = all(tag in element_tags for tag in cls._osm_relation_tags )

        return super(OSMRelation, cls).is_valid_xml(xml_element) and relation_sub_type and relation_tags

    @classmethod
    def is_valid_json(cls, json_element) -> bool:
        #If a relation subtype (e.g. multipolygon) is present will check it
        relation_sub_type =  json_element.get(
                         'tags', {}).get('type', None) == cls._osm_relation_type if cls._osm_relation_type else True
        
        relation_tags = all(tag in json_element.get('tags',{}) for tag in cls._osm_relation_tags)

        return super(OSMRelation, cls).is_valid_json(json_element) and relation_sub_type and relation_tags

    @classmethod
    def _create_init_dict_from_xml(cls, xml_element: Element) -> dict:
        return {
            'id':
            xml_element.attrib['id'],
            'tags': {
                node.attrib['k']: node.attrib['v']
                for node in xml_element.findall('tag')
            },
            'members': [
                OSMRelationMember.load_from_xml(member)
                for member in xml_element.findall('member')
            ]
        }

    def get_nodes(self) -> "list['OSMNode']":
        if len(self._nodes)==0 or len(self._nodes) != len(self._node_ids):
            self._nodes = [node for member in self.members.values() for node in member.get_nodes() ]
            list(self._library.get_elements(OSMNode, self._node_ids).values())
        return self._nodes
    
    def preprocess_instance(self):
        """Preprocess the relationship. Does the following in order:
        - Find and store a reference to all members part of the relationship
        - Preprocess it
        - Adding a reference to the relation in all nodes referenced
        """
        if self._is_preprocessed:
            return
        member_elements = self._library.get_elements_by_ids(list(self.members.keys()))
        for member_element in member_elements:
            member_element.preprocess_instance()
            self.members[member_element._id].element = member_element
        
        if any(m.element is None for m in self.members.values()):
            self.is_valid = False
            return
        for node in self.nodes:
            node.add_referenced_from(self.__class__, self._id)

        self._is_preprocessed = True


@unique
class MultipolygonRoleEnum(Enum):
    """
    ENUM to hold the allowed values for protocol
    """
    INNER: str = 'inner'
    OUTER: str = 'outer'


class OSMMultipolygon(object):

    role: MultipolygonRoleEnum
    ways: list[OSMWay]

    is_ordered: bool = False

    def __str__(self):
        return f"Multipolygon, made up of {len(self.ways)} ways(s) and role:{self.role}"
    
    def __init__(self, role: str, ways: list[OSMWay]) -> None:
        self.role = MultipolygonRoleEnum(role)
        self.ways = ways[:]

    def contains_way(self, id: int) -> bool:
        return any(way._id == id for way in self.ways)

    def is_closed(self) -> bool:
        if not self.is_ordered:
            self.order()
        return self.ways[0]._node_ids[0] == self.ways[-1]._node_ids[-1]

    def end_points(self)->list(OSMNode):
        if len(self.ways)== 0 or self.is_closed():
            return []
        if len(self.ways) == 1:
            return [self.ways[0].get_nodes()[0], self.ways[0].get_nodes()[1]]
        
        start = self.ways[0][0] if self.ways[0][0] in [self.ways[1][0],self.ways[1][-1]] else self.ways[0][-1]
        end = self.ways[-1][0] if self.ways[-1][0] in [self.ways[-2][0],self.ways[-2][-1]] else self.ways[-1][-1]
        return [start, end]

    def can_extend(self, extension: OSMWay | OSMMultipolygon, role: MultipolygonRoleEnum):
        if role != self.role:
            return False

        self.order()
        possible_extension_points = self.end_points()
        if isinstance(extension, OSMWay):
            return extension.get_nodes()[0] in possible_extension_points or extension.get_nodes()[-1] in possible_extension_points
        elif isinstance(extension, OSMMultipolygon):
            new_extension_points = extension.end_points()
            return any((p in possible_extension_points for p in new_extension_points))
        return False

    def extend(self, extension: OSMWay | OSMMultipolygon):
        extension_ways = [extension] if isinstance(extension,OSMWay) else extension.ways
        limits = extension.end_points()
        position = 0 if any(n in [self.ways[0][0], self.ways[0][-1]] for n in limits) else len(self.ways)
        self.ways[position:position] = extension_ways
        return

    def order(self) -> bool:
        if len(self.ways)<=1:
            return True
        
        if all((self.ways[i-1]._node_ids[-1] in [self.ways[i]._node_ids[0],self.ways[i]._node_ids[-1]] or
                self.ways[i-1]._node_ids[0] in [self.ways[i]._node_ids[0],self.ways[i]._node_ids[-1]] 
                for i in range(len(self.ways)))):
            return True
        
        success = False
        first_last = [self.ways[0]._node_ids[0], self.ways[0]._node_ids[-1]]
        try:
            for i in range(0, len(self.ways), 2):
                current = self.ways[i]
                
                next_way = next((way for way in self.ways[i:] if way._node_ids[0] in first_last or way._node_ids[-1] in first_last),None)
                

                before = first_last[0] in [next_way._node_ids[-1], next_way._node_ids[-1]]
                
                replace = 0 if before else -1
                replace_with = next_way._node_ids[-1] if next_way._node_ids[0] == first_last[replace] else next_way._node_ids[0]
                first_last[replace] = replace_with
                if before:
                    self.ways.insert(i, self.ways.remove(next_way))
            success=True
        except Exception as ex:
            pass
        return success

    
    def get_edges(self, bm, geoscn, reproject, ray_caster:DropToGround=None)->"list":
        edges=[]
        for way in self.ways:
            vertices = way.get_vertices(bm, geoscn=geoscn, reproject=reproject, ray_caster=ray_caster)
            shifted_vert = itertools.cycle(vertices)
            next(shifted_vert)
            edge_verts = zip(vertices, shifted_vert)
            edges.extend(
                bm.edges.new( v )
                for v in edge_verts)
        return edges

class OSMMultiPolygonRelation(OSMRelation):
    ''' A polygon relation is a special relation which serve to define areas or complex structures
    https://wiki.openstreetmap.org/wiki/Relation:multipolygon
    '''

    _osm_relation_type: ClassVar[str] = 'multipolygon'
    detail_level: ClassVar[int] = 2

    outer: OrderedDict[int, OSMRelationMember] = {}
    inner: OrderedDict[int, OSMRelationMember] = {}

    polygons: list[OSMMultipolygon]
    _parts: list[OSMMultiPolygonRelation]

    def __str__(self):
        return f"OSMMultipolygonRelation with id: {self._id}, made up of {len(self.members)} members(s) and tags:\n{pprint.pformat(self._tags)}"
    
    def __init__(self, **kwargs) -> None:
        self.polygons = []
        super(OSMMultiPolygonRelation, self).__init__(**kwargs)

        return

    @classmethod
    def is_valid(cls, element) -> bool:
        return super(OSMMultiPolygonRelation, cls).is_valid(element)

    def preprocess_instance(self):
        """Preprocess the relationship. Does the following in order:
        - Find and store a reference to all members part of the relationship
        - Group ways in multipolygons and divide them in inner and outer
        """
        if self._is_preprocessed:
            return
        
        member_elements = self._library.get_elements_by_ids(list(self.members.keys()))
        for member_element in member_elements:
            member_element.preprocess_instance()
            self.members[member_element._id].element = member_element
        
        for node in self.get_nodes():
            node.add_referenced_from(self.__class__, self._id)

        self.generate_polygons()
        self._is_preprocessed = True

    def generate_polygons(self):
        self.outer = {}
        self.inner = {}
        
        # Add all the members in correct Multipolygons and split between inner and outer
        for member in (m for m in self.members.values()
                       if m.element and isinstance(m.element, OSMWay)):
            if member.role == 'outer':
                self.outer[member.reference] = member.element
            elif member.role == 'inner':
                self.inner[member.reference] = member.element

            # If the way is already closed add it as a polygon
            if member.element.is_closed():
                self.polygons.append(
                    OSMMultipolygon(role=member.role, ways=[member.element]))
            else:
                possible_polygon = next((p for p in self.polygons if p.can_extend(member.element,  MultipolygonRoleEnum(member.role))), None)
                if possible_polygon:
                    possible_polygon.extend(member.element)
                else:
                    self.polygons.append(
                        OSMMultipolygon(role=member.role, ways=[member.element]))
        # Iterate to merge polygons
        open = next((p for p in self.polygons if not p.is_closed()),None)
        while open:
            attach_to = next((p for p in self.polygons if p.can_extend(open, open.role) ),None)
            if attach_to is None:
                raise Exception(f"Open Polygon {open} could not be closed")
            self.polygons.remove(open)
            attach_to.extend(open)
            open = next((p for p in self.polygons if not p.is_closed()),None)

class OSMBuildingRelation(OSMRelation, OSMBuilding):
    ''' A building relation is a special relation which serves to define a complex building
    https://wiki.openstreetmap.org/wiki/Relation:building
    '''

    _osm_relation_type: ClassVar[str] = 'building'
    detail_level: ClassVar[int] = 2

    members: OrderedDict[int, OSMRelationMember]

    def __str__(self):
        return f"OSMBuildingRelation with id: {self._id}, made up of {len(self.members)} members(s) and tags:\n{pprint.pformat(self._tags)}"
    
    def __init__(self, **kwargs) -> None:
        super(OSMBuildingRelation, self).__init__(**kwargs)
        return


    def preprocess_instance(self):
        """Preprocess the relationship. Does the following in order:
        - Adding a reference to the relation in all nodes referenced
        - Find and store a reference to all members part of the relationship
        - Find the outline way
        - Assign the other parts to the outline (Might be useless as already done by individual parts)
        """
        if self._is_preprocessed:
            return
        
        member_elements = self._library.get_elements_by_ids(list(self.members.keys()))
        for member_element in member_elements:
            member_element.preprocess_instance()
            self.members[member_element._id].element = member_element
        
        for node in self.get_nodes():
            node.add_referenced_from(self.__class__, self._id)

        #Find outer element
        outline_id, outline_element = next(
            ((member_id, member)
             for (member_id, member) in self.members.items()
             if member.role == 'outline'), (None, None))
        
        # If no no outline is found or outline is not Relation or Building
        if outline_element is None or not (
                isinstance(outline_element, OSMBuilding) or
            (isinstance(outline_element, OSMRelation) and 'building' not in outline_element._tags)):
            self._is_valid = False
            return
        
        # For each element not outline assign the element and add the element to
        # the outline parts
        for (member_id, member) in self.members.items():
            if member_id is not outline_id and member.role == 'part':
                member.element.outline = outline_element
                outline_element._parts.append(member.element)

############################################
# This does not make sense as the elements #
# could simply bubble up one level         #
############################################
# class OSMBuildingPartRelation(OSMRelation, OSMBuildingPart):
#     ''' A building relation which defines a building part
#     '''

#     _osm_relation_type: ClassVar[str] = 'building:part'
#     detail_level: ClassVar[int] = 2

#     members: OrderedDict[int, OSMRelationMember]

#     def __init__(self, **kwargs) -> None:
#         super(OSMBuildingPartRelation, self).__init__(**kwargs)

#         return

#     @classmethod
#     def is_valid(cls, element) -> bool:
#         return super(OSMBuildingPartRelation, cls).is_valid(element)

#     @classmethod
#     def is_valid_xml(cls, xml_element: Element) -> bool:
#         return super(OSMBuildingPartRelation, cls).is_valid_xml(xml_element) and any(
#             child.get('k', None) == 'type'
#             and child.get('v', None) == cls._osm_relation_type
#             for child in xml_element.findall('tag'))

#     @classmethod
#     def is_valid_json(cls, json_element) -> bool:
#         return super(OSMBuildingPartRelation,
#                      cls).is_valid_json(json_element) and json_element.get(
#                          'tags', {}).get('type', None) == cls._osm_relation_type

#     def preprocess_instance(self):
#         """Preprocess the relationship. Does the following in order:
#         - Adding a reference to the way in all nodes referenced
#         - Find and store a reference to all members part of the relationship
#         - Find the outline way
#         - Assign the other parts to the outline (Might be useless as already done by individual parts)
#         """
#         super(OSMBuildingPartRelation, self).preprocess_instance()

#         outline_id, outline_element = next(
#             ((member_id, member)
#              for (member_id, member) in self.members.items()
#              if member.role == 'outline'), (None, None))
#         if outline_element is None or not (
#                 isinstance(outline_element, OSMBuilding) or
#             (isinstance(outline_element, OSMRelation)
#              and 'building' in outline_element._tags)):
#             self._is_valid = False
#             return
#         for (member_id, member) in self.members.items():
#             if member_id is not outline_id and member.role == 'part':
#                 member.element.outline = outline_element
#                 outline_element._parts.append(member.element)

        

#     def assign_to_parent(self):
#         #Find which building the part belongs to
#         from .relation import OSMRelation
#         relation = next((r for r in self._library.get_elements(OSMRelation).values() if self._id in r.members), None)
#         if relation:
#             outline_id = relation.outline
#             outline = self._library.get_element_by_id(outline_id)
#             if isinstance(outline, OSMBuilding):
#                 outline.add_part(self)
#             return


#         # If no relation is found find the building which uses all/most of the nodes in the part 
        
#         shared_by = {} # Dictionary of building Ids and how many nodes are encompassed by it

#         free_node = None
#         for node in self._nodes:
#             referenced_by = node.get_referenced_from()[OSMBuilding]
#             if node._id in referenced_by:
#                 for ref in referenced_by:
#                     shared_by[ref] = shared_by.get(ref, 0) + 1
#             else:
#                 if free_node is None:
#                     free_node = node._id
        
#         # Find the building parts with the most candidates
#         max_references = max((len(s) for s in shared_by.values()), default = None)
#         candidates = [b for b in shared_by.keys() if len(shared_by[b])==max_references] if max_references else []

#         # If not all nodes are withing a building check if the parts is ray cast within a building
#         if free_node:
#             if len(candidates) == 1:
#                 # To save time we won't check if all free OSM nodes of <part>
#                 # are located inside the building
#                 self._library.get_element_by_id(candidates[0]).add_part(self)
#             else:
#                 # Take the first encountered free node <freeNode> and
#                 # calculated if it is located inside any building from <self.buildings>
#                 bvhTree = self._library.bvh_tree
#                 coords = next(n for n in self._nodes if n._id == free_node)
#                 # Cast a ray from the point with horizontal coords equal to <coords> and
#                 # z = -1. in the direction of <zAxis>
#                 buildingIndex = bvhTree.ray_cast((coords[0], coords[1], -1.), zAxis)[2]
#                 if not buildingIndex is None:
#                     # we consider that <part> is located inside <buildings[buildingIndex]>
#                     self._library.get_element_by_id(buildingIndex).add_part(self)
#         else:
#             # all OSM nodes of <part> are used by one or more buildings from <self.buildings>
#             # the case numCandidates > 1 probably means some weird configuration, so skip that <part>
#             if len(candidates) == 1:
#                 self._library.get_element_by_id(candidates[0]).add_part(self)


class OSMMultiPolygonBuildingRelation(OSMMultiPolygonRelation, OSMBuildingRelation):
    ''' A polygon relation which defines a building
    '''

    _osm_relation_type: ClassVar[str] = 'multipolygon'
    _osm_relation_tags: ClassVar["list['str']"] = ['building']
    detail_level: ClassVar[int] = 3

    polygons: list[OSMMultipolygon]
    _parts: list[OSMMultiPolygonRelation]

    def __str__(self):
        return f"OSMMultipolygonBuildingRelation with id: {self._id}, made up of {len(self.members)} members(s) and tags:\n{pprint.pformat(self._tags)}"
    
    def __init__(self, **kwargs) -> None:
        self.polygons = []
        self._parts = []
        super(OSMMultiPolygonBuildingRelation, self).__init__(**kwargs)

        return

    @classmethod
    def is_valid(cls, element) -> bool:
        return super(OSMMultiPolygonBuildingRelation, cls).is_valid(element)

    @classmethod
    def is_valid_xml(cls, xml_element: Element) -> bool:
        return super(OSMMultiPolygonBuildingRelation, cls).is_valid_xml(xml_element) and any(
            child.get('k', None) == 'type'
            and child.get('v', None) == cls._osm_relation_type
            for child in xml_element.findall('tag'))

    @classmethod
    def is_valid_json(cls, json_element) -> bool:
        return super(OSMMultiPolygonBuildingRelation,
                     cls).is_valid_json(json_element) and json_element.get(
                         'tags', {}).get('type', None) == cls._osm_relation_type

    
    def build_instance(self, geoscn, reproject, ray_caster = None, build_parameters:dict = {}) -> bpy.types.Object|None:

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
        edges =[e for polygon in self.polygons for e in polygon.get_edges(bm, geoscn=geoscn,reproject=reproject, ray_caster=ray_caster)]

        fill = bmesh.ops.triangle_fill(bm, use_beauty=True, use_dissolve=True, edges=edges)

        
        offset = None
        if "height" in self._tags:
                htag = self._tags["height"]
                htag.replace(',', '.')
                try:
                    offset = int(htag)
                except:
                    try:
                        offset = float(htag)
                    except:
                        for i, c in enumerate(htag):
                            if not c.isdigit():
                                try:
                                    offset, unit = float(htag[:i]), htag[i:].strip()
                                    #todo : parse unit  25, 25m, 25 ft, etc.
                                except:
                                    offset = None
        elif "building:levels" in self._tags:
            try:
                offset = int(self._tags["building:levels"]) * build_parameters.get('level_height',3)
            except ValueError as e:
                offset = None

        if offset is None:
            minH = build_parameters.get('default_height', 30) - build_parameters.get('random_height_threshold', 15)
            if minH < 0 :
                minH = 0
            maxH = build_parameters.get('default_height', 30) + build_parameters.get('random_height_threshold', 15)
            offset = random.randint(minH, maxH)

        #Extrude
        
        # if build_parameters.get('extrusion_axis', 'Z') == 'NORMAL':
        #     normal = face.normal
        #     vect = normal * offset
        # elif build_parameters.get('extrusion_axis', 'Z') == 'Z':
        vect = (0, 0, offset)

        # for f in geom["geom"]:
        #     if isinstance(f, bmesh.types.BMFace):
        extruded = bmesh.ops.extrude_face_region(bm, geom=fill['geom']) #return {'faces': [BMFace]}
        verts = [v for v in extruded['geom'] if isinstance(v, bmesh.types.BMVert)]
        if ray_caster:
            #Making flat roof
            z = max([v.co.z for v in verts]) + offset #get max z coord
            for v in verts:
                v.co.z = z
        else:
            bmesh.ops.translate(bm, verts=verts, vec=vect)

        return bm
    
    
class OSMMultiPolygonBuildingPartRelation(OSMMultiPolygonRelation, OSMBuildingPart):
    ''' A polygon relation which defines a building part
    Notes: the relation can have a building:levels, building:min_level tag
    '''

    _osm_relation_type: ClassVar[str] = 'multipolygon'
    _osm_relation_tags: ClassVar["list['str']"] = ['building:part']
    detail_level: ClassVar[int] = 3


    polygons: list[OSMMultipolygon]
    _parts: list[OSMMultiPolygonRelation]

    def __str__(self):
        return f"OSMMultiPolygonBuildingPartRelation with id: {self._id}, made up of {len(self.members)} members(s) and tags:\n{pprint.pformat(self._tags)}"
    
    def __init__(self, **kwargs) -> None:
        self.polygons = []
        super(OSMMultiPolygonBuildingPartRelation, self).__init__(**kwargs)
        return

    @classmethod
    def is_valid(cls, element) -> bool:
        return super(OSMMultiPolygonBuildingPartRelation, cls).is_valid(element)

    @classmethod
    def is_valid_xml(cls, xml_element: Element) -> bool:
        return super(OSMMultiPolygonBuildingPartRelation, cls).is_valid_xml(xml_element) and any(
            child.get('k', None) == 'type'
            and child.get('v', None) == cls._osm_relation_type
            for child in xml_element.findall('tag'))

    @classmethod
    def is_valid_json(cls, json_element) -> bool:
        return super(OSMMultiPolygonBuildingPartRelation,
                     cls).is_valid_json(json_element) and json_element.get(
                         'tags', {}).get('type', None) == cls._osm_relation_type

    def preprocess_instance(self):
        """Preprocess the relationship. Does the following in order:
        - Find and store a reference to all members part of the relationship
        - Find the outline way
        - Assign the other parts to the outline (Might be useless as already done by individual parts)
        """
        if self._is_preprocessed:
            return
        
        member_elements = self._library.get_elements_by_ids(list(self.members.keys()))
        for member_element in member_elements:
            member_element.preprocess_instance()
            self.members[member_element._id].element = member_element
        
        for node in self.get_nodes():
            node.add_referenced_from(self.__class__, self._id)

        self.generate_polygons()

        self.assign_to_parent()

        self._is_preprocessed = True

    def assign_to_parent(self):
        #Find if it is already referenced by a relation
        from .relation import OSMRelation
        relation = next((r for r in self._library.get_elements(OSMRelation).values() if self._id in r.members), None)
        if relation:
            outline_id = relation.outline
            outline = self._library.get_element_by_id(outline_id)
            if isinstance(outline, OSMBuilding):
                outline.add_part(self)
            return


        # If no relation is found find the building which uses all/most of the nodes in the part 
        
        shared_by = {} # Dictionary of building Ids and how many nodes are encompassed by it

        free_node = None
        nodes = [n for outline in self.outer.values() for n in outline.get_nodes()]
        for node in nodes:
            referenced_by = node.get_referenced_from().get(OSMBuilding, set())
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

    def build_instance(self, geoscn, reproject, ray_caster: DropToGround = None, build_parameters: dict = ...) -> bpy.types.Object|None:
        return
    
    def _build_instance(self, bm, geoscn, reproject, ray_caster:DropToGround = None, build_parameters:dict={})->bmesh:
        edges =[e for polygon in self.polygons for e in polygon.get_edges(bm, geoscn=geoscn,reproject=reproject, ray_caster=ray_caster)]

        fill = bmesh.ops.triangle_fill(bm, use_beauty=True, use_dissolve=True, edges=edges)

        
        offset = None
        if "height" in self._tags:
                htag = self._tags["height"]
                htag.replace(',', '.')
                try:
                    offset = int(htag)
                except:
                    try:
                        offset = float(htag)
                    except:
                        for i, c in enumerate(htag):
                            if not c.isdigit():
                                try:
                                    offset, unit = float(htag[:i]), htag[i:].strip()
                                    #todo : parse unit  25, 25m, 25 ft, etc.
                                except:
                                    offset = None
        elif "building:levels" in self._tags:
            try:
                offset = int(self._tags["building:levels"]) * build_parameters.get('level_height',3)
            except ValueError as e:
                offset = None

        if offset is None:
            minH = build_parameters.get('default_height', 30) - build_parameters.get('random_height_threshold', 15)
            if minH < 0 :
                minH = 0
            maxH = build_parameters.get('default_height', 30) + build_parameters.get('random_height_threshold', 15)
            offset = random.randint(minH, maxH)

        #Extrude
        
        # if build_parameters.get('extrusion_axis', 'Z') == 'NORMAL':
        #     normal = face.normal
        #     vect = normal * offset
        # elif build_parameters.get('extrusion_axis', 'Z') == 'Z':
        vect = (0, 0, offset)

        # for f in geom["geom"]:
        #     if isinstance(f, bmesh.types.BMFace):
        extruded = bmesh.ops.extrude_face_region(bm, geom=fill['geom']) #return {'faces': [BMFace]}
        verts = [v for v in extruded['geom'] if isinstance(v, bmesh.types.BMVert)]
        if ray_caster:
            #Making flat roof
            z = max([v.co.z for v in verts]) + offset #get max z coord
            for v in verts:
                v.co.z = z
        else:
            bmesh.ops.translate(bm, verts=verts, vec=vect)

        return bm
    
   