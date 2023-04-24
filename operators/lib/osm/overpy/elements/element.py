from __future__ import annotations
import pprint
from typing import ClassVar, TypeVar
from abc import ABC, abstractclassmethod, abstractmethod
from xml.etree.ElementInclude import include
from xml.etree.ElementTree import Element
from bpy.types import bpy_struct 
from typing import TYPE_CHECKING

from .....utils.bgis_utils import DropToGround

if TYPE_CHECKING:
    from ..OSMLibrary import OSMLibrary

T = TypeVar('T', bound='OSMElement')


class OSMElement(ABC):
    ''' Base class for all OSM elements.
    Elements are the basic components of OpenStreetMap's conceptual data model of the physical world.
    '''
    _osm_name: ClassVar[str] = 'element'
    detail_level: ClassVar[int] = 0

    _id: int
    _tags: dict[str, str]
    _library: OSMLibrary

    _is_valid: bool = True

    _is_preprocessed: bool = False
    _is_built: bool = False

    _referenced_by: dict[type, set[int]]

    @property
    def references(self):
        return set().union(*self._referenced_by.values()) 
    
    def add_reference(self, type, id):
        self._referenced_by.setdefault(type,set()).add(id)

    _blender_objects: list[bpy_struct] = []

    @property
    def blender_objects(self)->list[bpy_struct]:
        return self._blender_objects
    
    def add_blender_object_reference(self, obj:bpy_struct):
        self._blender_objects.append(obj)
    
    @property
    def is_built(self):
        return self._is_built
    
    def __str__(self):
        return f"Generic OSMElement with id: {self._id} and tags:\n{pprint.pformat(self._tags)}"
    
    def __init__(self, library: OSMLibrary, **kwargs) -> None:
        self._library = library
        self._id = int(kwargs.get("id", None))
        self._tags = kwargs.get("tags", {})
        return

    @classmethod
    def is_valid(cls, element) -> bool:
        if isinstance(element, dict) and cls.is_valid_json(element):
            return cls.is_valid_json(element)
        if isinstance(element, Element) and cls.is_valid_xml(element):
            return cls.is_valid_xml(element)

    @classmethod
    def is_valid_xml(cls, xml_element: Element) -> bool:
        return xml_element.tag.lower() == cls._osm_name

    @classmethod
    def is_valid_json(cls, json_element: dict) -> bool:
        return json_element['type'].lower() == cls._osm_name

    @classmethod
    def load(cls, library: OSMLibrary, element) -> T:
        kwargs = cls._create_init_dict(element=element)
        return cls(library=library, **kwargs)

    @classmethod
    def load_from_xml(cls, library: OSMLibrary, xml_element) -> T:
        kwargs = cls._create_init_dict_from_xml(xml_element=xml_element)
        return cls(library=library, **kwargs)

    @classmethod
    def load_from_json(cls, library: OSMLibrary, json_element) -> T:
        kwargs = cls._create_init_dict_from_json(json_element=json_element)
        return cls(library=library, **kwargs)

    @classmethod
    def _create_init_dict(cls, element) -> dict:
        if isinstance(element, dict) and cls.is_valid_json(element):
            return cls._create_init_dict_from_json(element)
        if isinstance(element, Element) and cls.is_valid_xml(element):
            return cls._create_init_dict_from_xml(element)

    @classmethod
    def _create_init_dict_from_xml(cls, xml_element: Element) -> dict:
        return {
            'id': xml_element.attrib['id'],
            'tags': {
                node.attrib['k']: node.attrib['v']
                for node in xml_element if node.tag.lower() == 'tag'
            }
        }

    @classmethod
    def _create_init_dict_from_json(cls, json_element: dict) -> dict:
        return json_element

    @classmethod
    def preprocess(cls, library: OSMLibrary):
        for part in library.get_elements(cls).values():
            part.preprocess_instance()
    
    def preprocess_instance(self):
        """Preprocess the element. Empty method"""
        self._is_preprocessed = True
        return

    @classmethod
    def build(cls, library: OSMLibrary, geoscn, reproject, ray_caster:DropToGround=None, build_parameters:dict={}) -> None:
        for part in library.get_elements(cls).values():
            if part.is_built:
                continue
            part.build_instance(geoscn=geoscn, reproject=reproject, ray_caster=ray_caster, build_parameters=build_parameters)
        return
    
    def build_instance(self, geoscn, reproject, ray_caster:DropToGround=None, build_parameters:dict={}) -> bpy.types.Object|None:
        return



