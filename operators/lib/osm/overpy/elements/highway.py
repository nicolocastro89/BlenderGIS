from __future__ import annotations
from dataclasses import dataclass
from itertools import groupby
import math
from numbers import Number
import pprint
import random
from typing import ClassVar, TypeVar
from xml.etree.ElementTree import Element

from .....utils.bgis_utils import DropToGround, RayCastHit
from .....utils.blender import appendObjectsFromAssets, createCollection, almost_overlapping
from .node import OSMNode
from .way import OSMWay
from mathutils import Vector
from bpy.types import Spline

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..OSMLibrary import OSMLibrary
    from . import OSMMultiPolygonRelation

import bpy
import bmesh
from bpy.types import Operator, Panel, AddonPreferences
from bpy.props import StringProperty, IntProperty, FloatProperty, BoolProperty, EnumProperty, FloatVectorProperty
from bpy import context
from bpy.ops import curve as BOC
xAxis = Vector((1., 0., 0.))
yAxis = Vector((0., 1., 0.))
zAxis = Vector((0., 0., 1.))
#import bmesh


T = TypeVar('T', bound='OSMHighway')

@dataclass
class HighwayContinuation:
    before: OSMHighway|None
    after: OSMHighway|None

class OSMHighway(OSMWay):
    '''A Highway is any kind of road, street or path. This is a base class which should never be directly
    assigned to any element.
    '''

    _osm_sub_name: ClassVar[str] = 'highway'
    _osm_highway_type: str = None
    _highways_to_level:list[str] = ["bridge", 'tunnel']
    _highways_roads:list[str] = ["motorway", 'truck', 'primary', 'secondary', 'tertiary', 'unclassified', 'residential', 'service', 'pedestrian', 'track', 'other']
    _highways_paths:list[str] = ["footway", 'steps', 'cycleway']
    detail_level: ClassVar[int] = 2

    # Blender file with way profiles
    assetFile: ClassVar[str] = "way_profiles.blend"
    collectionName: ClassVar[str] = "way_profiles"

    @property
    def highway_type(self):
        return self._tags['highway']
    
    @property
    def bevel_name(self):
        profile_subtype = self.highway_type
        if profile_subtype == 'motorway_link':
            profile_subtype = 'motorway'
        profile_type= "roads" if profile_subtype in self._highways_roads else  "paths"
        return f'profile_roads_{self.highway_type}'
    
    def __str__(self):
        return f"OSMWay of road with id: {self._id}, made up of {len(self._node_ids)} nodes(s) and tags:\n{pprint.pformat(self._tags)}"

    def __init__(self, **kwargs):
        super(OSMHighway,self).__init__(**kwargs)
    
    @classmethod
    def is_valid_xml(cls, xml_element:Element) -> bool:
        #If a relation subtype (e.g. multipolygon) is present will check it
        # highway_type = any(
        #     child.get('k', None) == 'highway'
        #     and child.get('v', None) == cls._osm_highway_type
        #     for child in xml_element.findall('tag')) if cls._osm_highway_type else True
        

        return super(OSMHighway, cls).is_valid_xml(xml_element) and any(
                c.attrib['k'] == cls._osm_sub_name for c in xml_element.iter('tag'))
    # and highway_type

    @classmethod
    def is_valid_json(cls, json_element:dict) -> bool:
        return super(OSMHighway, cls).is_valid_json(json_element) and cls._osm_sub_name in json_element.get('tags',{})

    def get_ray_cast_hit_points(self, geoscn, reproject, ray_caster: DropToGround = None, subdivision_size:Number=None)->list[RayCastHit]:

        dx, dy = geoscn.crsx, geoscn.crsy

        delta_xy = Vector(dx,dy)
        pts = [(float(node._lon), float(node._lat)) for node in self.nodes]
        pts = [{'point':Vector(xy)-delta_xy, 'nodes':[self.nodes[idx]]} for idx, xy in enumerate(reproject.pts(pts))]
        if subdivision_size:
            subdivided=[]
            subdivided.append(first)
            for first, second in zip(pts, pts[1:]):
                number_steps, vector = self.getSubdivisionParams(first['xy'], second['xy'], subdivision_size)
                subdivided.extend(({'point': first['xy']+step*vector, 'nodes':[]} for step in range(1,number_steps)))
            subdivided.append(pts[-1])
            pts=subdivided

        
        rc_hits = []
        if ray_caster:
            for v in pts:
                rc_hit = ray_caster.rayCast(v['point'].x, v['point'].y)
                rc_hit.originating_node_ids = v['nodes']
                rc_hits.append(rc_hit)

            hits = [pt.hit for pt in rc_hits]
            if not all(hits) and any(hits):
                zs = [p.loc.z for p in pts if p.hit]
                meanZ = sum(zs) / len(zs)
                for v in pts:
                    if not v.hit:
                        v.loc.z = meanZ
        else:
            pts = [ RayCastHit(loc=(v['point'].x, v['point'].y, 0), hit = True, originates_from= v['nodes']) for v in pts]

        # pts = [(float(node._lon), float(node._lat)) for node in self.nodes]
        # pts = reproject.pts(pts)
        # if subdivision_size:
        #     subdivided = []
        #     for first, second in zip(pts, pts[1:]):
        #         number_steps, vector = self.getSubdivisionParams(first, second, subdivision_size)
        #         subdivided.extend((Vector(first)+(step*vector)).to_tuple() for step in range(number_steps))
        #     subdivided.append(pts[-1])
        #     pts=subdivided
        # dx, dy = geoscn.crsx, geoscn.crsy

        # if ray_caster:
        #     pts = [ray_caster.rayCast(v[0]-dx, v[1]-dy) for v in pts]
        #     hits = [pt.hit for pt in pts]
        #     if not all(hits) and any(hits):
        #         zs = [p.loc.z for p in pts if p.hit]
        #         meanZ = sum(zs) / len(zs)
        #         for v in pts:
        #             if not v.hit:
        #                 v.loc.z = meanZ
        # else:
        #     pts = [ RayCastHit(loc=(v[0]-dx, v[1]-dy, 0), hit = True) for v in pts]
                                
        # return pts
            
    @classmethod
    def build(cls, library: OSMLibrary, geoscn, reproject, ray_caster:DropToGround=None, build_parameters:dict={}) -> None:

        cls.load_highway_profiles()
        highways = library.get_elements(cls)

        for _id, highway in highways.items():
            highway = highway.build_instance(geoscn=geoscn, reproject=reproject, ray_caster=ray_caster, build_parameters=build_parameters)
            
        
        return
    
    @classmethod
    def load_highway_profiles(cls):
        if cls.collectionName not in bpy.data.collections:
            collection = createCollection(
                        cls.collectionName,
                        hide_viewport=True,
                        hide_select=True,
                        hide_render=True
            )
        else:
            collection = bpy.data.collections.get(cls.collectionName)

        bevelObjs = appendObjectsFromAssets(cls.assetFile, collection, 'profile.*')
        for bevelObj in bevelObjs:
            bevelObj.hide_viewport = True
            bevelObj.hide_select = True
            bevelObj.hide_render = True
            
    def get_highway_continuation(self)->HighwayContinuation:
        first, last = self.end_points()
        
        first_referenced = self._library.get_elements_by_ids(list(first.get_referenced_from().get(OSMHighway, set())))
        first_referenced = [h for h in first_referenced if h._id!=self._id and h.highway_type == self.highway_type and first in h.end_points()]
        first_referenced = first_referenced[0] if len(first_referenced)==1 else None

        last_referenced = self._library.get_elements_by_ids(list(last.get_referenced_from().get(OSMHighway, set())))
        last_referenced = [h for h in last_referenced if h._id!=self._id and h.highway_type == self.highway_type and last in h.end_points()]
        last_referenced = last_referenced[0] if len(last_referenced)==1 else None

        return HighwayContinuation(first_referenced, last_referenced)

    def build_instance(self, geoscn, reproject, ray_caster:DropToGround = None, build_parameters:dict = {}, continue_from:bpy.types.Object|None = None) -> bpy.types.Object|None:
        if self._is_built:
            return
        #Create a new bmesh
        if continue_from is not None:
            obj = continue_from
        else:
            curve = bpy.data.curves.new(f'{self._id}', 'CURVE')
            curve.fill_mode = 'NONE'
            if hasattr(curve, "bevel_mode"):
                curve.bevel_mode = 'OBJECT'
            obj = bpy.data.objects.new(f'{self._id}', curve)

        if obj.vertex_groups.get(f'{self._id}',None) is None:
            obj.vertex_groups.add(f'{self._id}')
        self._build_instance(obj, geoscn=geoscn, reproject=reproject, ray_caster=ray_caster, build_parameters = build_parameters)
        if continue_from is None:
            geoscn.scn.collection.objects.link(obj)
        self._is_built =True
        continuation = self.get_highway_continuation()
        if continuation.after is not None:
            continuation.after.build_instance(geoscn=geoscn, reproject=reproject, ray_caster=ray_caster, build_parameters = build_parameters, continue_from=obj)
        if continuation.before is not None:
            continuation.before.build_instance(geoscn=geoscn, reproject=reproject, ray_caster=ray_caster, build_parameters = build_parameters, continue_from=obj)
        obj.data.bevel_object = bpy.data.objects.get(self.bevel_name)
        return obj

    
    def _build_instance(self, obj, geoscn, reproject, ray_caster:DropToGround = None, build_parameters:dict={})->bmesh:
        
        points = self.get_ray_cast_hit_points(geoscn=geoscn, reproject=reproject, ray_caster=ray_caster, subdivision_size=build_parameters.get('highway_subdivision_size', None))
        ends = [points[0].loc,points[-1].loc]
        full_extension=[]
        if any(t in self._tags for t in self._highways_to_level):
            continuation = self.get_highway_continuation()
            full_after = []
            full_before = []
            subtype = next(t for t in self._highways_to_level if t in self._tags )
            if continuation.after and subtype in continuation.after._tags:
                full_after = [continuation.after]
                full_after.extend(continuation.after.get_all_highway_parts(self, True, possible_subtypes=[subtype]))
            if continuation.before and subtype in continuation.before._tags:
                full_before = [self, continuation.before]
                full_before.extend(continuation.after.get_all_highway_parts(self, True, possible_subtypes=[subtype]))
            full_extension = full_before[::-1]
            full_extension.append(self)
            full_extension.extend(full_after)
            points = self.combine_highway_points(full_extension, geoscn, reproject, ray_caster, build_parameters)
            self.level_points(points)

        splines_points = [list(g) for k, g in groupby(points, key= lambda p: p.hit) if k]
        for spline_points in splines_points:
            #override = context.copy()
            _spline = next((s for s in obj.data.splines if any(almost_overlapping(s.points[0].co, e) for e in ends) or any(almost_overlapping(s.points[-1].co, e) for e in ends)), None)
            start_at=0
            if _spline is None:
                _spline = obj.data.splines.new('POLY')
                _spline.use_endpoint_u = True
                _spline.points[start_at+idx].co=(*spline_points[0].loc,1.0)
                #_spline.points.add(len(spline_points)-1)
            else:
                #if the end of the spline matches the end of the new points or vice versa reverse the points
                if almost_overlapping(_spline.points[-1].co, ends[-1]) or  almost_overlapping(_spline.points[0].co, ends[0]):
                    spline_points.reverse()

                #If the end of the spline matches one of the new endpoints start at the end
                if _spline.points[-1].co.xyz in ends:
                    start_at=len(_spline.points)-1
                    _spline.points.add(len(spline_points)-1)
                else: #shift the spline 
                    self.shift_spline_by(_spline, len(spline_points)-1)
            
            BOC.select_all(action='DESELECT')
            select_spline_point(_spline.points[start_at]) # the new point will be connected here
            BOC.vertex_add(location=location)
            for idx, p in enumerate(spline_points):
                _spline.points[start_at+idx].co=(*p.loc,1.0)
                for n in p.originating_node_ids:
                    group = obj.vertex_groups.get(f'{n}', None)
                    if group is None:
                        group = obj.vertex_groups.new(f'{n}')

                    group.add(_spline.points[start_at+idx].index, 1, 'ADD')
                        

            _spline = None
        for part in full_extension:
            part._is_built = True

    def select_spline_point(point):
        if isinstance(point, bpy.types.BezierSplinePoint):
            point.select_control_point = True
            point.select_left_handle = True
            point.select_right_handle = True
        else:
            point.select = True

    def combine_highway_points(self, highway_parts: list[OSMHighway], geoscn, reproject, ray_caster:DropToGround = None, build_parameters:dict={}):
        points = highway_parts[0].get_ray_cast_hit_points(geoscn=geoscn, reproject=reproject, ray_caster=ray_caster, subdivision_size=build_parameters.get('highway_subdivision_size', None))
        for idx in range(1,len(highway_parts)):
            current_part = highway_parts[idx]
            previous_part = highway_parts[idx-1]

            part_points = current_part.get_ray_cast_hit_points(geoscn=geoscn, reproject=reproject, ray_caster=ray_caster, subdivision_size=build_parameters.get('highway_subdivision_size', None))
            if current_part.nodes[-1] == previous_part.nodes[-1]:
                part_points.reverse()

            points.extend(part_points[1:])
        return points
       
    def shift_spline_by(self, spline, by:int):
        spline.points.add(by)
        for i in reversed(range(len(spline.points))):
            if i>=by:
                spline.points[i].co = [*spline.points[i-by].co]
            else:
                spline.points[i].co = (0,0,0,1.0)

    def level_points(self, points: list[RayCastHit]):
        full_length = sum(((p1.loc[0]-p2.loc[0])**2 + (p1.loc[1]-p2.loc[1])**2)**0.5 for p1, p2 in zip(points[:-1], points[1:]))
        
        starting_height = points[0].loc[2] if points[0].hit else None
        ending_height = points[-1].loc[2] if points[-1].hit else None

        delta_z=0
        if starting_height is None and ending_height is None:
            first_initial_node = next((p for p in points if p.hit), None)
            if first_initial_node:
                starting_height = first_initial_node.loc[2]
            first_ending_node = next((p for p in points[::-1] if p.hit), None)
            if first_ending_node:
                ending_height = first_ending_node.loc[2]
            starting_height = ending_height = max(starting_height, ending_height)

        if starting_height is not None and ending_height is None:
            ending_height = starting_height
        elif starting_height is None and ending_height is not None:
            starting_height = ending_height
            
        delta_z = (ending_height-starting_height)/full_length

        for i in range(1,len(points)):
            xy_dist = ((points[i-1].loc[0]-points[i].loc[0])**2 + (points[i-1].loc[1]-points[i].loc[1])**2)**0.5
            points[i].loc[2] = points[i-1].loc[2] + delta_z*xy_dist

    def get_all_highway_parts(self, previous: OSMHighway, stop_at_type_change = True, possible_subtypes = [])->list[OSMHighway]:
        conditions = {
            'different_ids': lambda p, n: n is not None and n._id != p._id,
            'same_type': lambda p, n: n is not None and p.highway_type == n.highway_type,
            'subtype_in': lambda p, n: n is not None and (any(n._tags.get(t, None)== 'yes' for t in possible_subtypes) or len(possible_subtypes)==0)
        }
        all_parts = []
        filter_algorithm = lambda p, n: conditions['different_ids'](p,n)
        if stop_at_type_change and len(possible_subtypes)>0:
            filter_algorithm = lambda p, n: conditions['different_ids'](p,n) and conditions['same_type'](p,n) and conditions['subtype_in'](p,n)
        elif stop_at_type_change:
            filter_algorithm = lambda p, n: conditions['different_ids'](p,n) and conditions['same_type'](p,n)
        elif len(possible_subtypes)>0:
            filter_algorithm = lambda p, n: conditions['different_ids'](p,n) and conditions['subtype_in'](p,n)

        continuation = self.get_highway_continuation()
        next = None
        if filter_algorithm(previous, continuation.after):
            next = continuation.after
        elif filter_algorithm(previous, continuation.before):
            next = continuation.before
        
        if next is not None:
            all_parts.append(next)
            all_parts.extend(next.get_all_highway_parts(self, stop_at_type_change))

        return all_parts
        

    def isPointOnTerrain(self, point, ray_caster:DropToGround = None) -> bool:
        return ray_caster(point, -zAxis)[0] != None

    def getSubdivisionParams(self, preceding_point: tuple[float,float], current_point:tuple[float,float], subdivision_size: Number)->tuple[int, Vector]:
        vec = Vector(current_point) - Vector(preceding_point)
        number_steps = math.ceil(vec.length/subdivision_size)
        return number_steps, vec/number_steps
    
    def setSplinePoint(self, spline,  index: int, point):
        self.spline.points[index].co = (point[0], point[1], point[2], 1.)
        self.pointIndex += 1

    def processOnTerrainOnTerrain(self, spline, preceding_point, current_point, number_steps: int, vec, closed):
        """
        Create spline if both points are on the terrain just create the intermediate points
        """
        
        index = len(spline.points)
        spline.points.add(number_steps)
        p = preceding_point
        for i in range(number_steps-1):
            p = p + vec
            spline.points[index+i].co = (*p ,1.0)
            self.setSplinePoint(index+i, p)
        self.setSplinePoint(len(spline.points)-1, current_point)
    
    def processNoTerrainOnTerrain(self, spline, preceding_point, current_point, number_steps, vec, ray_caster:DropToGround = None):
        """
        Start the spline at the first sub division over the terrain. Start halfway and 
        keep adjusting by half.
        """
        # By iteration find the first point on the terrain
        firstTerrainPointIndex = number_steps
        bound1 = 0
        bound2 = number_steps
        # index of subdivision points starting from 1
        pointIndex = math.ceil((bound1 + bound2)/2)
        while True:
            # coordinates of the subdivision point with the index <pointIndex>
            if self.isPointOnTerrain(preceding_point + pointIndex * vec, ray_caster):
                firstTerrainPointIndex = pointIndex
                if pointIndex == bound1+1: #If it's the first point then stop
                    break
                bound2 = pointIndex
                pointIndex = math.ceil((bound1 + bound2)/2)
            else:
                if pointIndex==bound2-1: #Last possible index, found point is the first
                    break
                bound1 = pointIndex
                pointIndex = math.ceil((bound1 + bound2)/2)

        spline.points.add(number_steps - firstTerrainPointIndex)
        for pointIndex in range(firstTerrainPointIndex, number_steps):
            p = preceding_point + pointIndex * vec
            spline.points[pointIndex-firstTerrainPointIndex].co = (*p ,1.0)
            self.setSplinePoint(pointIndex-firstTerrainPointIndex, p)
        self.setSplinePoint(len(spline.points)-1, current_point)
    
    def processOnTerrainNoTerrain(self, spline, preceding_point, current_point, number_steps, vec, ray_caster:DropToGround = None):
        """
        Find the last point to plot as the spline goes off terrain
        """
        lastTerrainPointIndex = 0
        bound1 = 0
        bound2 = number_steps
        # index of subdivision points starting from 1
        pointIndex = math.ceil((bound1 + bound2)/2)
        while True:
            # coordinates of the subdivision point with the index <pointIndex>
            if self.isPointOnTerrain(preceding_point + pointIndex * vec, ray_caster=ray_caster):
                lastTerrainPointIndex = pointIndex
                if pointIndex == bound2-1:
                    break
                bound1 = pointIndex
                pointIndex = math.ceil((bound1 + bound2)/2)
            else:
                if pointIndex==bound1+1:
                    break
                bound2 = pointIndex
                pointIndex = math.floor((bound1 + bound2)/2)
        if lastTerrainPointIndex:
            
            spline.points[0].co = (*preceding_point ,1.0)
            spline.points.add(lastTerrainPointIndex)
            for pointIndex in range(1, lastTerrainPointIndex+1):
                p = preceding_point + pointIndex * vec
                spline.points[pointIndex].co = (*p ,1.0)
        self.spline = None
    

class OSMMotorway(OSMHighway):
    """
    A restricted access major divided highway, normally with 2 or more running lanes plus emergency hard shoulder. Equivalent to the Freeway, Autobahn, etc..
    """
    _osm_highway_type: str = 'motorway'
    detail_level: ClassVar[int] = 3

class OSMTrunk(OSMHighway):
    """
    The most important roads in a country's system that aren't motorways. (Need not necessarily be a divided highway.)
    """
    _osm_highway_type: str = 'trunk'
    detail_level: ClassVar[int] = 3

class OSMPrimary(OSMHighway):
    """
    The next most important roads in a country's system. (Often link larger towns.)	
    """
    _osm_highway_type: str = 'primary'
    detail_level: ClassVar[int] = 3

class OSMSecondary(OSMHighway):
    """
    The next most important roads in a country's system. (Often link towns.)		
    """
    _osm_highway_type: str = 'secondary'
    detail_level: ClassVar[int] = 3

class OSMTertiary(OSMHighway):
    """
    The next most important roads in a country's system. (Often link smaller towns and villages)			
    """
    _osm_highway_type: str = 'tertiary'
    detail_level: ClassVar[int] = 3

class OSMUnclassified(OSMHighway):
    """
    The least important through roads in a country's system – i.e. minor roads of a lower classification than tertiary, but which serve a purpose other than access to properties. (Often link villages and hamlets.)
The word 'unclassified' is a historical artefact of the UK road system and does not mean that the classification is unknown; you can use highway=road for that.			
    """
    _osm_highway_type: str = 'unclassified'
    detail_level: ClassVar[int] = 3