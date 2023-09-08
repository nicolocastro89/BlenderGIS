from typing import ClassVar, TypeVar

from .highway import OSMHighway


T = TypeVar('T', bound='OSMHighway')



class OSMRailway(OSMHighway):
    '''A Highway is any kind of road, street or path. This is a base class which should never be directly
    assigned to any element.
    '''
    blender_mesh_name: ClassVar[str] = "Railway"
    _osm_sub_name: ClassVar[str] = 'railway'

    @property
    def highway_type(self):
        return self._tags['railway']
    
    @property
    def highway_base_type(self):
        return self._tags['railway'].removesuffix('_link')
    
    @property
    def bevel_name(self):
        return f'profile_railways'
    
    @classmethod
    def _bevel_name(cls, highway_type):
        return f'profile_railways'
    