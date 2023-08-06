from __future__ import annotations
import cProfile, pstats, io
from collections import Counter
from pstats import SortKey
import pprint
from typing import ClassVar, TypeVar
from abc import ABC, abstractclassmethod, abstractmethod
from xml.etree.ElementInclude import include
from xml.etree.ElementTree import Element
from bpy.types import bpy_struct 
from typing import TYPE_CHECKING

from mathutils import Vector
from mathutils.kdtree import KDTree

from .....utils.bgis_utils import DropToGround

if TYPE_CHECKING:
    from ..OSMLibrary import OSMLibrary

T = TypeVar('T', bound='OSMElement')


class OSMElement(ABC):
    ''' Base class for all OSM elements.
    Elements are the basic components of OpenStreetMap's conceptual data model of the physical world.
    '''

    xAxis = Vector((1., 0., 0.))
    yAxis = Vector((0., 1., 0.))
    zAxis = Vector((0., 0., 1.))

    _osm_name: ClassVar[str] = 'element'
    detail_level: ClassVar[int] = 0

    _id: int
    _tags: dict[str, str]
    @property
    def tags(self):
        return self._tags
    
    _library: OSMLibrary

    _is_valid: bool = True

    _is_preprocessed: bool = False
    _is_built: bool = False

    _referenced_by: dict[type, set[int]] = {}

    _blender_element = None

    def __hash__(self):
        return self._id

    def __eq__(self, other):
        return type(self)==type(other) and self._id==other._id 
    
    @property
    def nodes(self)->list:
        return []
    
    @property
    def references(self):
        return set().union(*self._referenced_by.values()) 
    
    def add_reference(self, type, id):
        self._referenced_by.setdefault(type,set()).add(id)
    
    def get_referenced_from(self, referencing_type: OSMElement, include_sub_classes = False)->set(int):
        references = set(self._referenced_by.get(referencing_type,set()))
        if include_sub_classes:
            for sub_type in referencing_type.__subclasses__():
                references.union(set(self._referenced_by.get(sub_type,set())))
        return references
    
    @property
    def is_built(self)->bool:
        return self._is_built
    
    @is_built.setter
    def is_built(self, value:bool):
        self._is_built = value
    
    def __str__(self):
        return f"Generic OSMElement with id: {self._id} and tags:\n{pprint.pformat(self._tags)}"
    
    def __init__(self, library: OSMLibrary, **kwargs) -> None:
        self._library = library
        self._id = int(kwargs.get("id", None))
        self._tags = kwargs.get("tags", {})
        self._blender_element = None
        self._library.add_element(self)
        return

    def merge(self, original: OSMElement):
        self._tags.update(original.tags)
        self._blender_element = original._blender_element
        
    @classmethod
    def is_valid_data(cls, element) -> bool:
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
    def preprocess(cls, library: OSMLibrary, ray_caster:DropToGround):
        for part in library.get_elements(cls).values():
            part.preprocess_instance(geoscn = library.geo_scene, ray_caster=ray_caster)
    
    def preprocess_instance(self, geoscn, ray_caster:DropToGround):
        """Preprocess the element. Empty method"""
        self._is_preprocessed = True
        return

    @classmethod
    def build(cls, library: OSMLibrary, geoscn, reproject, ray_caster:DropToGround=None, build_parameters:dict={}) -> None:
        pr = cProfile.Profile()
        pr.enable()
        for part in library.get_elements(cls).values():
            if part.is_built or not part._is_valid:
                continue
            part.build_instance(geoscn=geoscn, reproject=reproject, ray_caster=ray_caster, build_parameters=build_parameters)
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(f'Profile of {cls} building')
        print(s.getvalue())
        return
    
    def build_instance(self, geoscn, reproject, ray_caster:DropToGround=None, build_parameters:dict={}) -> bpy.types.Object|None:
        return
    
    def assign_to_best_overlap(self, container_type, allow_multiple=False):
        for candidate in  self.get_best_overlap(container_type=container_type, allow_multiple=allow_multiple):
            self._library.get_element_by_id(candidate).add_reference(type(self),self._id)

    def get_best_overlap(self, container_type, allow_multiple=False)->list(int):
        shared_by = Counter() # Counter of Ids and how many nodes are encompassed by it

        free_nodes = []
        for node in self.nodes:
            referenced_by = node.get_referenced_from(container_type)
            if referenced_by:
                shared_by.update(referenced_by)
            else:
                free_nodes.append(node._id)
        
        candidates = []
        if shared_by.total()>0:
            max_references = shared_by.most_common(1)[0][1]
            candidates = [c[0] for c in shared_by.most_common() if c[1]==max_references]
        


        # If not all nodes are withing a building check if the parts is ray cast within a building
        if len(free_nodes):
            if len(candidates) == 1 or (len(candidates)>1 and allow_multiple):
                # To save time we won't check if all free OSM nodes of <part>
                # are located inside the building
                return candidates
            else:
                # Take the first encountered free node <freeNode> and
                # calculated if it is located inside any building from <self.buildings>
                (bvhTree,bvh_tree_index) = self._library.get_bvh_tree(container_type)
                shared_by = Counter()
                for node in self.nodes:
                    elementIndex = bvhTree.ray_cast((node._lat, node._lon, -1.), self.zAxis)[2]
                    if elementIndex is not None:
                        shared_by.update([bvh_tree_index[elementIndex]])
                    
                
                if shared_by.total()>0:
                    max_references = shared_by.most_common(1)[0][1]
                    candidates = [c[0] for c in shared_by.most_common() if c[1]==max_references]
                    
        if not allow_multiple:
            #decide that we assign the element to the first candidate
            if len(candidates)==0:
                return []
            return [candidates[0]]
        else:
            return [c for c in candidates]
            


