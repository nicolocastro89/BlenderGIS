from __future__ import annotations
import math
from numbers import Number
import pprint
from typing import ClassVar, TypeVar
from xml.etree.ElementTree import Element
from .....utils.bgis_utils import DropToGround, remove_straight_angles

from .element import OSMElement
from .node import OSMNode

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..OSMLibrary import OSMLibrary

T = TypeVar('T', bound='OSMWay')
from mathutils import Vector

class OSMWay(OSMElement):
    ''' A way is one of the fundamental elements of the map. In everyday language, it is a line.
    A way normally represents a linear feature on the ground (such as a road, wall, or river).
    Technically a way is an ordered list of nodes which normally also has at least one tags or is
    included within a relations. A way can have between 2 and 2,000 nodes, although it's possible 
    that faulty ways with zero or a single node exist. A way can be open or closed.
    '''
    blender_mesh_name: ClassVar[str] = "HighWayway"
    _osm_name: ClassVar[str] = 'way'
    detail_level: ClassVar[int] = 1

    _node_ids: "list['int']" = []
    _nodes: "list['OSMNode']" = []

    def __str__(self):
        return f"OSMWay with id: {self._id}, made up of {len(self._node_ids)} nodes(s) and tags:\n{pprint.pformat(self._tags)}"
    
    def __init__(self, **kwargs):
        super(OSMWay, self).__init__(**kwargs)
        self._node_ids = [int(n) for n in kwargs.get('nodes',[])]
        self.add_reference_to_nodes()
        return

    def __getitem__(self, position:int)->OSMNode:
         return self.nodes[position]
    
    def is_closed(self) -> bool:
        return self.closed
    
    @property
    def closed(self) -> bool:
        return self._node_ids[0] == self._node_ids[-1]
    
    @property
    def previous_ways(self) -> list[T]:
        previous = []
        if not self.is_closed:
            node = self.nodes[0]
            referenced_ways  = node._referenced_by[T]
            previous = [w for w in referenced_ways if w._id != self._id]

        return previous
    
    @property
    def following_ways(self) -> list[T]:
        next = []
        if not self.is_closed:
            node = self.nodes[-1]
            referenced_ways  = node._referenced_by[T]
            next = [w for w in referenced_ways if w._id != self._id]
        return next

    @classmethod
    def is_valid_data(cls, element) -> bool:
        return super(OSMWay, cls).is_valid_data(element)

    @classmethod
    def is_valid_xml(cls, xml_element) -> bool:
        return super(OSMWay, cls).is_valid_xml(xml_element)

    @classmethod
    def is_valid_json(cls, json_element) -> bool:
        return super(OSMWay, cls).is_valid_json(json_element)

    @classmethod
    def _create_init_dict_from_xml(cls, xml_element: Element) -> dict:
        base_dict = super(
            OSMWay, cls)._create_init_dict_from_xml(xml_element=xml_element)
        base_dict['nodes'] = [
            int(node.attrib['ref']) for node in xml_element.iter('nd')
        ]
        return base_dict

    @classmethod
    def _create_init_dict_from_json(cls, json_element: dict) -> dict:
        return super(
            OSMWay, cls)._create_init_dict_from_json(json_element=json_element)

    @property
    def nodes(self)->list[OSMNode]:
        if len(self._nodes) != len(self._node_ids):
            try:
                self._nodes = list(self._library.get_elements(OSMNode, self._node_ids).values())
            except Exception as e:
                print(f'Failed to retrieve nodes for {self}.\n{e}')
                
        return self._nodes
    
    def get_nodes(self) -> "list['OSMNode']":
        if len(self._nodes) != len(self._node_ids):
            try:
                self._nodes = list(self._library.get_elements(OSMNode, self._node_ids).values())
            except Exception as e:
                print(f'Failed to retrieve nodes for {self}.\n{e}')
        return self._nodes

    def end_points(self)->tuple[OSMNode]:
        nodes = self.nodes
        if len(nodes)==0:
            print('Cannot retrieve end points for {self} as it contains 0 nodes')
            raise IndexError
        return (self._nodes[0], self._nodes[-1])

    def preprocess_instance(self, geoscn, ray_caster:DropToGround):
        """Preprocess the way by doing the following in order:
        - Adding a reference to the way in all nodes referenced
        """
        if self._is_preprocessed:
            return
        #self.add_reference_to_nodes()
        self._is_preprocessed = True
        return

    def add_reference_to_nodes(self):
        for id in self._node_ids:
            node = self._library.get_element_by_id(id)
            if node is None:
                node = OSMNode(id = id, library = self._library)
            node.add_reference(self.__class__, self._id)
        # for node in self.nodes:
        #     node.add_reference(self.__class__, self._id)

    def get_points(self, geoscn= None, reproject=None, ray_caster: DropToGround = None)->"list":
        hits = [node.ray_cast_hit for node in self.nodes] 
        if not all(h.hit for h in hits) and any(h.hit for h in hits):
            zs = [p.loc.z for p in hits if p.hit]
            meanZ = sum(zs) / len(zs)
            for v in hits:
                if not v.hit:
                    v.loc.z = meanZ
        pts = [pt.loc for pt in hits]
        return pts 
    
    def get_vertices(self, bm, geoscn=None, reproject=None, ray_caster: DropToGround = None, subdivision_size: Number = None, straight_line_toll = None)->"list":
        pts = self.get_points(geoscn=geoscn, reproject=reproject, ray_caster=ray_caster)
        if straight_line_toll:
            pts = remove_straight_angles(pts, straight_line_toll)
        if subdivision_size:
            pts = self.subdivide_way(pts, subdivision_size)

        return [bm.verts.new(pt) for pt in pts]                      

    def subdivide_way(self, points: list(tuple(float,float,float)), subdivision_size: Number)->list[tuple[float,float,float]]:
        subdivided=[]
        subdivided.append(points[0])
        for first, second in zip(points, points[1:]):
            number_steps, vec = self.get_subdivision_params(first, second, subdivision_size)
            subdivided.extend((first+step*vec for step in range(1,number_steps)))

        subdivided.append(points[-1])

        return subdivided

    def get_subdivision_params(self, preceding_point: tuple[float,float,float], current_point:tuple[float,float,float], subdivision_size: Number)->tuple[int, Vector]:
        vec = Vector(current_point) - Vector(preceding_point)
        if subdivision_size is None:
            return 1, vec
        number_steps = max(math.ceil(vec.length/subdivision_size),1)
        return number_steps, vec/number_steps
    
    def calculate_length(self, geoscn = None, reproject = None,  ray_caster: DropToGround = None):
        pts = pts = [(float(node._lon), float(node._lat), 0) for node in self.nodes]
        if geoscn is not None and reproject is not None:
            pts = self.get_points(geoscn=geoscn, reproject=reproject, ray_caster=ray_caster)
        return sum(((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)**0.5 for p1, p2 in zip(pts[:-1], pts[1:]))