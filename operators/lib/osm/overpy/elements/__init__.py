from __future__ import annotations

from .man_made import OSMBridge
from .highway import OSMHighway
from .building import OSMBuilding, OSMBuildingPart
from .relation import OSMMultiPolygonBuildingPartRelation, OSMMultiPolygonBuildingRelation, OSMRelation, OSMBuildingRelation, OSMMultiPolygonRelation
from .element import OSMElement
from .node import OSMNode
from .way import OSMWay
from .railway import OSMRailway
from .amenity import OSMAmenity
from .barrier import OSMBarrier
from .liesure import OSMLeisure

from typing import TypeVar, Literal
from .. import OSMLibrary
from .....utils.bgis_utils import all_subclasses
T = TypeVar('T', bound='OSMElement')

_existing_elements:list[T] = [
    OSMNode, 
    OSMRelation, 
    OSMWay, 
    OSMHighway,
    OSMRailway,
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
    for possibility in (sc for sc in all_subclasses(OSMElement) if sc.detail_level>=0):
        is_valid = getattr(possibility, validation_method_name)
        if is_valid(element_definition):
            valid_types.append(possibility)
    
    if len(valid_types)>0:
        valid_types.sort(key=lambda x: x.detail_level, reverse=True)
        return getattr(valid_types[0], creation_method_name)(library, element_definition)
