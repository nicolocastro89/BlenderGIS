from __future__ import annotations
import pprint
from typing import ClassVar, TypeVar
from xml.etree.ElementTree import Element
from .....utils.bgis_utils import DropToGround

from .element import OSMElement
from .node import OSMNode

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..OSMLibrary import OSMLibrary

T = TypeVar('T', bound='OSMWay')


class OSMWay(OSMElement):
    ''' A way is one of the fundamental elements of the map. In everyday language, it is a line.
    A way normally represents a linear feature on the ground (such as a road, wall, or river).
    Technically a way is an ordered list of nodes which normally also has at least one tags or is
    included within a relations. A way can have between 2 and 2,000 nodes, although it's possible 
    that faulty ways with zero or a single node exist. A way can be open or closed.
    '''

    _osm_name: ClassVar[str] = 'way'
    detail_level: ClassVar[int] = 1

    _node_ids: "list['int']" = []
    _nodes: "list['OSMNode']" = []

    def __str__(self):
        return f"OSMWay with id: {self._id}, made up of {len(self._node_ids)} nodes(s) and tags:\n{pprint.pformat(self._tags)}"
    
    def __init__(self, **kwargs):
        super(OSMWay, self).__init__(**kwargs)
        self._node_ids = [int(n) for n in kwargs.get('nodes',[])]
        return

    def __getitem__(self, position:int)->OSMNode:
         return self.get_nodes()[position]
    
    def is_closed(self) -> bool:
        return self._node_ids[0] == self._node_ids[-1]

    @classmethod
    def is_valid(cls, element) -> bool:
        return super(OSMWay, cls).is_valid(element)

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

    def preprocess_instance(self):
        """Preprocess the way by doing the following in order:
        - Adding a reference to the way in all nodes referenced
        """
        if self._is_preprocessed:
            return
        self.add_reference_to_nodes()
        self._is_preprocessed = True
        return

    def add_reference_to_nodes(self):
        for node in self.nodes:
            node.add_referenced_from(self.__class__, self._id)

    def get_points(self, geoscn, reproject, ray_caster: DropToGround = None)->"list":
        pts = [(float(node._lon), float(node._lat)) for node in self.nodes]
        pts = reproject.pts(pts)
        dx, dy = geoscn.crsx, geoscn.crsy

        if ray_caster:
            pts = [ray_caster.rayCast(v[0]-dx, v[1]-dy) for v in pts]
            hits = [pt.hit for pt in pts]
            if not all(hits) and any(hits):
                zs = [p.loc.z for p in pts if p.hit]
                meanZ = sum(zs) / len(zs)
                for v in pts:
                    if not v.hit:
                        v.loc.z = meanZ
            pts = [pt.loc for pt in pts]
        else:
            pts = [ (v[0]-dx, v[1]-dy, 0) for v in pts]
                                
        return pts
    
    def get_vertices(self, bm, geoscn, reproject, ray_caster: DropToGround = None)->"list":
        pts = self.get_points(geoscn=geoscn, reproject=reproject, ray_caster=ray_caster)
        
        return [bm.verts.new(pt) for pt in pts]                      

    def calculate_length(self, geoscn = None, reproject = None,  ray_caster: DropToGround = None):
        pts = pts = [(float(node._lon), float(node._lat), 0) for node in self.nodes]
        if geoscn is not None and reproject is not None:
            pts = self.get_points(geoscn=geoscn, reproject=reproject, ray_caster=ray_caster)
        return sum(((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)**0.5 for p1, p2 in zip(pts[:-1], pts[1:]))