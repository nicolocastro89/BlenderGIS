from __future__ import annotations
from ast import Num
import pprint
from typing import ClassVar, TypeVar

from .element import OSMElement
from xml.etree.ElementTree import Element
from ......core.proj import Reproj
from typing import TYPE_CHECKING
from .....utils.bgis_utils import DropToGround, RayCastHit
if TYPE_CHECKING:
    from ..OSMLibrary import OSMLibrary


T = TypeVar('T', bound='OSMNode')


class OSMNode(OSMElement):
    ''' A node is one of the core elements in the OpenStreetMap data model. 
    It consists of a single point in space defined by its latitude, longitude and node id.
    '''
    blender_mesh_name: ClassVar[str] = "Node"
    _osm_name: ClassVar[str] = 'node'
    detail_level: ClassVar[int] = 1

    
    _lat:Num
    _lon:Num
    _ele:Num

    ray_cast_hit:RayCastHit

    def __str__(self):
        return f"OSMNode with id: {self._id} located at Lon:{self._lon}, Lat{self._lat} and tags:\n{pprint.pformat(self._tags)}"
    
    def __init__(self, **kwargs):
        super(OSMNode,self).__init__(**kwargs)
        self._lat = kwargs.get('lat', None)
        self._lon = kwargs.get('lon', None)
        self._ele = kwargs.get('ele', None)
        self._referenced_by = {}
        return 
    
    def merge(self, node: OSMNode):
        super(OSMNode,self).merge(node)
        self._lat = node._lat
        self._lon = node._lon
        self._ele = node._ele
        self._referenced_by.update(node.reference)

    @classmethod
    def is_valid_data(cls, element) -> bool:
        return super(OSMNode, cls).is_valid_data(element)

    @classmethod
    def is_valid_xml(cls, xml_element: Element) -> bool:
        return super(OSMNode, cls).is_valid_xml(xml_element)

    @classmethod
    def is_valid_json(cls, json_element) -> bool:
        return super(OSMNode, cls).is_valid_json(json_element)

    @classmethod
    def load_from_xml(cls, library:OSMLibrary, xml_element) -> T:
        kwargs = cls._create_init_dict_from_xml(xml_element = xml_element)
        existing_node = library.get_element_by_id(kwargs['id'])
        node = OSMNode(library=library, **kwargs)
        if existing_node:
            existing_node.merge(node)
            return existing_node
        return node
        

    @classmethod
    def load_from_json(cls, library:OSMLibrary, json_element) -> T:
        kwargs = cls._create_init_dict_from_json(json_element = json_element)
        existing_node = library.get_element_by_id(kwargs['id'])
        node = OSMNode(library=library, **kwargs)
        if existing_node:
            existing_node.merge(node)
            return existing_node
        return node

    @classmethod
    def _create_init_dict_from_xml(cls, xml_element: Element)->dict:
        base_dict = super(OSMNode, cls)._create_init_dict_from_xml(xml_element=xml_element)
        base_dict['lat'] = float(xml_element.attrib['lat'])
        base_dict['lon'] = float(xml_element.attrib['lon'])
        if 'ele' in xml_element.attrib:
            base_dict['ele'] = float(xml_element.attrib['ele'])

        return base_dict

    def get_nodes(self) -> "list['OSMNode']":
        return [self]

    @property
    def nodes(self)->list[OSMNode]:
        return [self]
    
    def preprocess_instance(self, geoscn, ray_caster:DropToGround):
        """Preprocess the way by doing the following in order:
        - Adding a reference to the way in all nodes referenced
        """
        if self._is_preprocessed:
            return
        
        pt = self._library.reprojector.pt(float(self._lon), float(self._lat))
        dx, dy = geoscn.crsx, geoscn.crsy

        if ray_caster:
            self.ray_cast_hit = ray_caster.rayCast(pt[0]-dx, pt[1]-dy) 
            
        else:
            self.ray_cast_hit = RayCastHit(loc = (pt[0]-dx, pt[1]-dy, 0))
                                
        
    @classmethod
    def _create_init_dict_from_json(cls, json_element: dict)->dict:
        return super(OSMNode, cls)._create_init_dict_from_json(json_element=json_element)

    
    
    


