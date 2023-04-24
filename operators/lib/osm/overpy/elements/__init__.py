from __future__ import annotations

from .man_made import OSMBridge
from .highway import OSMHighway
from .building import OSMBuilding, OSMBuildingPart
from .relation import OSMMultiPolygonBuildingPartRelation, OSMMultiPolygonBuildingRelation, OSMRelation, OSMBuildingRelation, OSMMultiPolygonRelation
from .element import OSMElement
from .node import OSMNode
from .way import OSMWay

from typing import TypeVar, Literal
from .. import OSMLibrary

T = TypeVar('T', bound='OSMElement')

_existing_elements:list[T] = [
    OSMNode, 
    OSMRelation, 
    OSMWay, 
    OSMHighway,
    OSMBridge,
    OSMBuilding, 
    OSMBuildingPart, 
    OSMBuildingRelation, 
    OSMMultiPolygonRelation, 
    OSMMultiPolygonBuildingRelation, 
    OSMMultiPolygonBuildingPartRelation]

def ElementFactory(library: OSMLibrary.OSMLibrary, element_definition, data_type:Literal['xml', 'json'] = 'xml')-> OSMElement:
    validation_method_name = f'is_valid_{data_type}'
    creation_method_name = f'load_from_{data_type}'
    valid_types:list[T] = []
    for possibility in _existing_elements:
        is_valid = getattr(possibility, validation_method_name)
        if is_valid(element_definition):
            valid_types.append(possibility)
    valid_types.sort(key=lambda x: x.detail_level, reverse=True)
    if len(valid_types)>0:
        return getattr(valid_types[0], creation_method_name)(library, element_definition)
