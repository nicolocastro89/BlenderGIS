from __future__ import annotations
from collections import OrderedDict
from itertools import groupby
from numbers import Number
import json
import os
import xml.etree.ElementTree as ET
from .....core.proj.reproj import Reproj, reprojBbox, reprojPt, UTM
from .....geoscene import GeoScene
import logging
from .elements import ElementFactory
from typing import TYPE_CHECKING, Type
if TYPE_CHECKING:
    from .elements import T, OSMNode, OSMElement, OSMRelation, OSMWay, OSMBuilding

log = logging.getLogger(__name__)

import bpy

from ....utils.bgis_utils import DropToGround

PKG, SUBPKG = __package__.split('.', maxsplit=1)


class BoundingBox(object):
    _south: float
    _west: float
    _north: float
    _east: float
    def __init__(self, south:Number, west:Number, north:Number, east:Number):
        self._south = float(south)
        self._west = float(west)
        self._north = float(north)
        self._east = float(east)

    @property
    def south(self):
        return self._south
    
    @property
    def minlat(self):
        return self._south

    @property
    def minlon(self):
        return self._west

    @property
    def maxlon(self):
        return self._north

    @property
    def maxlat(self):
        return self._east
    


class OSMLibrary(object):
    """
    Class to store the overpass response
    """

    _class_collection_map: dict[T, OrderedDict[int, OSMElement]] = {}
    _bounds: BoundingBox

    _bvh_trees: dict[str,'BVHTree'] = {}
    _bvh_trees_index: dict[str,'list["int"]'] = {}
    _bvh_tree: 'BVHTree' = None
    _bvh_tree_index: 'list["int"]' = None

    _processed: bool
    reprojector: Reproj = None

    def get_bvh_tree(self, element_type)->tuple['BVHTree',list[int]]:
        #currently can't support mergeing trees
        if isinstance(element_type, list):
             (tree, tree_index) = self.create_bvh_tree(element_type=element_type)
             return (tree, tree_index)
        tree = self._bvh_trees.get(element_type, None)
        tree_index = self._bvh_trees_index.get(element_type, None)
        if tree is None:
            (tree, tree_index) = self.create_bvh_tree(element_type=element_type)
            self._bvh_trees[element_type] = tree
            self._bvh_trees_index[element_type] = tree_index

        return (tree, tree_index)
    
    @property
    def bvh_tree(self)->'BVHTree':
        if self._bvh_tree is None:
           from .elements import T, OSMNode, OSMElement, OSMRelation, OSMWay, OSMBuilding
           self._bvh_tree,  self._bvh_tree_index = self.create_bvh_tree(self.reprojector, OSMBuilding)
        return self._bvh_tree

    @bvh_tree.setter
    def bvh_tree(self, value:'BVHTree'):
        self._bvh_tree=value
    
    @property
    def bvh_tree_index(self)->'list["int"]':
        if self._bvh_tree_index is None:
           from .elements import T, OSMNode, OSMElement, OSMRelation, OSMWay, OSMBuilding
           self.create_bvh_tree(self.reprojector, OSMBuilding)
        return self._bvh_tree_index

    @bvh_tree_index.setter
    def bvh_tree_index(self, value:'list["int"]'):
        self._bvh_tree_index=value

    def __init__(self, elements=None, api=None, bounds=None) -> None:
        """

        :param List elements:
        :param api:
        :type api: overpy.Overpass
        """
        if elements is None:
            elements = []

        self.extend_elements(collection=elements)

        self.api = api

        self._bounds = {}
        if isinstance(bounds, tuple):
            self._bounds = BoundingBox(*bounds)
        elif isinstance(bounds, BoundingBox):
            self._bounds = bounds

        self._processed = True

    def is_valid_type(self, element, cls) -> bool:
        """
        Test if an element is of a given type.

        :param Element() element: The element instance to test
        :param Element cls: The element class to test
        :return: False or True
        :rtype: Boolean
        """
        return isinstance(element, cls) and element.id is not None

    def extend_elements(self, collection: list[OSMElement]):
        for element in collection:
            self.add_element(element)

    def add_element(self, element: OSMElement):
        if element is None:
            return
        el_type = type(element)
        if el_type in self._class_collection_map:
            self._class_collection_map[el_type][element._id] = element
        else:
            self._class_collection_map[el_type] = OrderedDict([(element._id, element)])

    def get_element_by_id(self, elem_id: int) -> OSMElement | None:
        return self.get_elements(elem_id=[elem_id]).get(elem_id, None)

    def get_elements_by_ids(self, elem_ids: list[int]) -> list[OSMElement]:
        return list(self.get_elements(elem_id=elem_ids).values())

    def get_elements(self,
                     filter_cls: T|list(T) = None,
                     elem_id: int | list('int') = None) -> OrderedDict[int, T]:
        """
        Get a list of elements from the result and filter the element type by a class.

        :param filter_cls:
        :param elem_id: ID of the object
        :type elem_id: Integer
        :return: List of available elements
        :rtype: List
        """
        result = OrderedDict()
        if filter_cls is not None and not isinstance(filter_cls, list):
            filter_cls = [filter_cls]

        if filter_cls is not None and all(filter not in self._class_collection_map for filter in filter_cls):
            return result
    
        classes = filter_cls if filter_cls is not None else self._class_collection_map.keys()
        for cls in classes:
            if elem_id is not None:
                elem_id = [elem_id] if isinstance(elem_id, int) else elem_id
                for id in elem_id:
                    elem_value = self._class_collection_map.get(cls, {}).get(
                        id, None)
                    if elem_value is not None:
                        result[id] = elem_value
            else:
                result.update(self._class_collection_map.get(cls,{}))
        return result

    def get_ids(self, filter_cls):
        """

        :param filter_cls:
        :return:
        """
        return list(self._class_collection_map[filter_cls].keys())

    def get_node_ids(self):
        return self.get_ids(filter_cls=OSMNode)

    def get_way_ids(self):
        return self.get_ids(filter_cls=OSMWay)

    def get_relation_ids(self):
        return self.get_ids(filter_cls=OSMRelation)

    def preprocess(self, ray_caster:DropToGround):
        for element_types in self._class_collection_map.keys():
            element_types.preprocess(self, ray_caster = ray_caster)
        self.bvh_tree = None

        return

    def create_bvh_tree(self, reprojector: Reproj = None, element_type=None)->tuple['BVHTree',list[int]]:
        from mathutils.bvhtree import BVHTree
        from .elements import OSMBuilding
        element_type = element_type if element_type is not None else OSMBuilding
        vertices = []
        polygons = []
        polygon_start = 0
        polygon_end = 0
        bvh_index = []
        for idx, building in self.get_elements(element_type).items():
            if reprojector:
                nodes = [(p[0],p[1],0) for p in reprojector.pts([[n._lat, n._lon]
                                    for n in building.nodes[:-1]])]
            else:
                nodes =[(n._lat, n._lon, 0) for n in building.nodes[:-1]]
            # In the case of a multipolygon we consider the only outer linestring that defines the outline
            # of the polygon
            if not nodes:
                # no outer linestring, so skip it
                continue
            vertices.extend(nodes)
            polygon_end = len(vertices)
            polygons.append(tuple(range(polygon_start, polygon_end)))
            polygon_start = polygon_end
            bvh_index.append(idx)
        return BVHTree.FromPolygons(vertices, polygons), bvh_index
        # self.bvh_tree = BVHTree.FromPolygons(vertices, polygons)
        # self.bvh_tree_index = bvh_index

    def build(self, context, ray_caster:DropToGround, separate:bool, build_parameters: dict = {})->list[bpy.types.Object]:
        prefs = context.preferences.addons[PKG].preferences
        scn = context.scene
        geoscn = GeoScene(scn)
        scale = geoscn.scale  #TODO


        layer = bpy.data.collections.new('OSM')
        context.scene.collection.children.link(layer)
        osm_built = []
        for element_type in self._class_collection_map.keys():
            built_objects = element_type.build(self, 
                               geoscn = geoscn, 
                               reproject = self.reprojector, 
                               ray_caster = ray_caster, 
                               build_parameters = build_parameters)
            if built_objects is None:
                print(f'BUULD OBJECTS IS NONE AFTER {element_type}')
            if not separate and len(built_objects)>0:
                base_object = bpy.context.scene.objects.get(element_type.blender_mesh_name, next(iter(built_objects)))

                with bpy.context.temp_override(active_object=base_object, selected_objects=list(built_objects.union([base_object])), selected_editable_objects=list(built_objects.union([base_object]))):
                    bpy.ops.object.join()
                    base_object.name = element_type.blender_mesh_name
               
                built_objects = [base_object]

            osm_built.extend(built_objects)

        return osm_built

    @classmethod
    def from_json(cls, data):
        try:
            isFile = os.path.exists(data)
        except:
            isFile = False

        if isFile:
            with open(data, 'r', encoding='utf-8') as f:
                data = json.load(data)

        library = cls()

        elements = [
            ElementFactory(library, e, data_type='json')
            for e in data.get("elements", [])
        ]

        library.extend_elements(elements=elements)
        return library

    @classmethod
    def from_xml(cls, data):
        try:
            isFile = os.path.exists(data)
        except:
            isFile = False

        if isFile:
            with open(data, 'r', encoding='utf-8') as f:
                data = f.read()  #all file in memory
        root = ET.fromstring(data)

        bounds_node = root.find('bounds') 

        library = cls(bounds= BoundingBox(bounds_node.get('minlat'), bounds_node.get('minlon'), bounds_node.get('maxlat'), bounds_node.get('maxlon')))

        elements = [ElementFactory(library, e, data_type='xml') for e in root]

        # library.extend_elements(collection=elements)
        return library


