from __future__ import annotations
from abc import abstractmethod
from collections import OrderedDict
import collections
import random
import itertools
import math
from numbers import Number
import pprint
import random
import sys
import traceback
from typing import ClassVar, Type, TypeVar
from xml.etree.ElementTree import Element

from .....utils.bgis_utils import DropToGround, all_subclasses, find_geometric_center, parse_measurement, remove_straight_angles, find_longest_direction
from .....utils.blender import polish_mesh, polish_object
from .....utils.straight_skeleton import straightSkeletonOfPolygon
from .node import OSMNode
from .way import OSMWay
from .element import OSMElement
from mathutils import Vector, Quaternion

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..OSMLibrary import OSMLibrary
    from . import OSMMultiPolygonRelation

import bpy
import bmesh
from bpy.types import Operator, Panel, AddonPreferences
from bpy.props import StringProperty, IntProperty, FloatProperty, BoolProperty, EnumProperty, FloatVectorProperty

xAxis = Vector((1., 0., 0.))
yAxis = Vector((0., 1., 0.))
zAxis = Vector((0., 0., 1.))
#import bmesh


T = TypeVar('T', bound='OSMBuilding')



class OSMBuilding(OSMWay):
    '''A building is a man-made structure with a roof, standing more or less permanently in one place
    '''
    blender_mesh_name: ClassVar[str] = "Buildings"
    _osm_sub_name: ClassVar[str] = 'building'
    detail_level: ClassVar[int] = 2

    _parts: list[OSMBuildingPart | OSMMultiPolygonRelation]
    
    @property
    def _is_valid(self)-> bool:
        return len(self._node_ids)>3
    
    def __str__(self):
        return f"OSMWay of type building with id: {self._id}, made up of {len(self._node_ids)} nodes(s) and tags:\n{pprint.pformat(self._tags)}"

    def __init__(self, **kwargs):
        super(OSMBuilding,self).__init__(**kwargs)
        self._parts=[]

    def get_height(self, build_parameters)->int:
        building_height=0
        if "height" in self._tags:
                building_height = parse_measurement(self._tags["height"])
                
                roof_height = self._tags.get('roof:height', None)
                if roof_height:
                    roof_height = parse_measurement(roof_height)
                    building_height - roof_height

        elif "building:levels" in self._tags:
            try:
                building_height = int(self._tags["building:levels"]) * build_parameters.get('level_height',3)
            except ValueError as e:
                building_height = None

        else:
            minH = build_parameters.get('default_height', 30) - build_parameters.get('random_height_threshold', 15)
            if minH < 0 :
                minH = 0
            maxH = build_parameters.get('default_height', 30) + build_parameters.get('random_height_threshold', 15)
            building_height = self._id%(maxH-minH)+minH
            
        return building_height
    
    @classmethod
    def is_valid_xml(cls, xml_element:Element) -> bool:
        # for c in xml_element.iter('tag'):
        return super(OSMBuilding, cls).is_valid_xml(xml_element) and any(c.attrib['k'] == cls._osm_sub_name for c in xml_element.iter('tag'))

    @classmethod
    def is_valid_json(cls, json_element:dict) -> bool:
        return super(OSMBuilding, cls).is_valid_json(json_element) and cls._osm_sub_name in json_element.get('tags',{})

    def preprocess_instance(self, geoscn, ray_caster:DropToGround):
        """Preprocess the building  by doing the following in order:
        - Adding a reference to the way in all nodes referenced
        """
        super(OSMBuilding,self).preprocess_instance(geoscn,ray_caster)
        _is_valid = len(self._node_ids)>3 and self.is_closed()
        return
    
    def build_instance(self, geoscn, reproject, ray_caster:DropToGround = None, build_parameters:dict = {}) -> bpy.types.Object|None:
        #Create a new bmesh
        bm = bmesh.new()
        if len(self._parts)>0:
            for part in self._parts:
                part._build_instance(bm, geoscn=geoscn, reproject = reproject, ray_caster = ray_caster, build_parameters = build_parameters)
        else:
            self._build_instance(bm, geoscn=geoscn, reproject=reproject, ray_caster=ray_caster, build_parameters = build_parameters)

        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.01)
        mesh = bpy.data.meshes.new(f"{self._id}")
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()#calc_edges=True)
        mesh.validate()
        obj = bpy.data.objects.new(f"{self._id}", mesh)
        
        geoscn.scn.collection.objects.link(obj)
        obj.select_set(True)
        polish_object(obj)
        return obj
            
    def _build_instance(self, bm, geoscn, reproject, ray_caster:DropToGround = None, build_parameters:dict={})->bmesh:
        print(f'building {self._id}')
        plant_verts = self.get_vertices(bm, geoscn=geoscn, reproject=reproject, ray_caster=ray_caster, straight_line_toll=4.5)

        #This should only be useful for building parts
        min_height = float(self._tags['min_height']) if 'min_height' in self._tags else float(self._tags.get('min_level',0))*build_parameters.get('level_height',3)
        bmesh.ops.translate(bm, verts=plant_verts, vec=(0, 0, min_height))
        
        #plant edges
        shifted_vert = itertools.cycle(plant_verts)
        try:
            next(shifted_vert)
        except Exception as e:
            print(self._id)
            print(e)
            raise e
            
        edges=[]
        for v in zip(plant_verts,shifted_vert):
            if v[1] not in [x for y in [a.verts for a in v[0].link_edges] for x in y if x != v[0]]:
                edges.append(bm.edges.new(v))
        try:
            face = bm.faces.new(plant_verts)
        except:
            pass
            #print(f'Face error {self._id}')
        
        #ensure face is up (anticlockwise order)
        #because in OSM there is no particular order for closed ways
        
        face.normal_update()
        if face.normal.z > 0:
            face.normal_flip()

        building_height = self.get_height(build_parameters=build_parameters)
        

        building_height -=min_height

        #Extrude
        
        if build_parameters.get('extrusion_axis', 'Z') == 'NORMAL':
            normal = face.normal
            vect = normal * building_height
        elif build_parameters.get('extrusion_axis', 'Z') == 'Z':
            vect = (0, 0, building_height)

        # extrusion = bmesh.ops.extrude_edge_only(bm, edges = edges)
        extrusion = bmesh.ops.extrude_face_region(bm, geom=[face]) #return {'faces': [BMFace]} extrude_edge_only

        faces = [v for v in extrusion['geom'] if isinstance(v,bmesh.types.BMFace)]
        verts = [v for v in faces[0].verts]
        edges = [e for e in faces[0].edges]

        # extrusion = bmesh.ops.extrude_edge_only(bm, edges = edges)
        # # bmesh.ops.extrude_face_region(bm, faces=[face]) #return {'faces': [BMFace]} extrude_edge_only
        # verts = [v for v in extrusion['geom'] if isinstance(v,bmesh.types.BMVert)]
        # edges = [v for v in extrusion['geom'] if isinstance(v,bmesh.types.BMEdge)]
        
        if ray_caster:
            #Making flat roof
            z = max([v.co.z for v in verts]) + building_height #get max z coord
            for v in verts:
                v.co.z = z
        else:
            bmesh.ops.translate(bm, verts=verts, vec=vect)

        self.build_roof(bm, verts, edges, build_parameters)
        
        return bm
    
    def build_roof(self, bm:BMesh, roof_verts:list[BMVert], roof_edges:list[BMEdge], build_parameters:dict):
        roof_builder_type = OSMRoof.get_roof_builder(self) or OSMFlatRoof
        roof_builder = roof_builder_type(self)
        try:
            roof_builder.build_roof(bm = bm, roof_verts = roof_verts, roof_edges = roof_edges, build_parameters=build_parameters)
        except Exception as e:
            print(f'Failed to build the desired roof {roof_builder_type} becuase of {e}.')
            print(traceback.format_exc())
            print('Falling back on FlatRoof')
            roof_builder = OSMFlatRoof(self)
            roof_builder.build_roof(bm = bm, roof_verts = roof_verts, roof_edges = roof_edges, build_parameters=build_parameters)

    def add_part(self, part: OSMBuildingPart):
        if part not in self._parts:
            self._parts.append(part)

    
class OSMBuildingPart(OSMBuilding):

    _osm_sub_name: ClassVar[str] = 'building:part'
    detail_level: ClassVar[int] = 3
    part_of: OSMBuilding
     
    def __str__(self):
        return f"OSMWay of type  Building Part with id: {self._id}, made up of {len(self._node_ids)} nodes(s) and tags:\n{pprint.pformat(self._tags)}"

    def __init__(self, **kwargs):
        super(OSMBuilding,self).__init__(**kwargs)


    @classmethod
    def is_valid_xml(cls, xml_element) -> bool:
        return super(OSMBuildingPart, cls).is_valid_xml(xml_element)

    @classmethod
    def is_valid_json(cls, json_element) -> bool:
        return super(OSMBuildingPart, cls).is_valid_json(json_element) and cls._osm_sub_name in json_element.get('tags',{})

    def preprocess_instance(self, geoscn, ray_caster:DropToGround):
        """Preprocess the building part. Does the following in order:
        - Adding a reference to the way in all nodes referenced
        - Find the parent element of the building part
        """
        if self._is_preprocessed:
            return
        
        self.add_reference_to_nodes()
            
        self.assign_to_parent()
        self._is_preprocessed = True
        

    def assign_to_parent(self):
        """Find the parent of the building part and assign the part to the parent
        """
        #Find which building the part belongs to
        from .relation import OSMRelation
        relation = next((r for r in self._library.get_elements(OSMRelation).values() if self._id in r.members), None)
        if relation:
            outline_id = relation.outline
            outline = self._library.get_element_by_id(outline_id)
            if isinstance(outline, OSMBuilding):
                self.part_of = outline
                outline.add_part(self)
            return

        candidates = self.get_best_overlap(OSMBuilding, False)
        for candidate in candidates:
            outline = self._library.get_element_by_id(candidate)
            outline.add_part(self)
            self.part_of=outline
        # If no relation is found find the building which uses all/most of the nodes in the part 
        
        # shared_by = {} # Dictionary of building Ids and how many nodes are encompassed by it

        # free_node = None
        # for node in self._nodes:
        #     referenced_by = node.get_referenced_from(OSMBuilding)
        #     if node._id in referenced_by:
        #         for ref in referenced_by:
        #             shared_by[ref] = shared_by.get(ref, 0) + 1
        #     else:
        #         if free_node is None:
        #             free_node = node
        
        # # Find the building parts with the most candidates
        # max_references = max((len(s) for s in shared_by.values()), default = None)
        # candidates = [b for b in shared_by.keys() if len(shared_by[b])==max_references] if max_references else []

        # # If not all nodes are withing a building check if the parts is ray cast within a building
        # if free_node:
        #     if len(candidates) == 1:
        #         # To save time we won't check if all free OSM nodes of <part>
        #         # are located inside the building
        #         self._library.get_element_by_id(candidates[0]).add_part(self)
        #     else:
        #         # Take the first encountered free node <freeNode> and
        #         # calculated if it is located inside any building from <self.buildings>
        #         bvhTree = self._library.bvh_tree
        #         p = self._library.reprojector.pt(node._lat, node._lon)
        #         # Cast a ray from the point with horizontal coords equal to <coords> and
        #         # z = -1. in the direction of <zAxis>
        #         buildingIndex = bvhTree.ray_cast((p[0], p[1], -1.), zAxis)[2]
        #         if buildingIndex is not None:
        #             # we consider that <part> is located inside <buildings[buildingIndex]>
        #             self._library.get_element_by_id(self._library.bvh_tree_index[buildingIndex]).add_part(self)
        # else:
        #     # all OSM nodes of <part> are used by one or more buildings from <self.buildings>
        #     # the case numCandidates > 1 probably means some weird configuration, so skip that <part>
        #     if len(candidates) == 1:
        #         self._library.get_element_by_id(candidates[0]).add_part(self)
            

    def build(self, geoscn, reproject, ray_caster = None, build_parameters:dict = {}) -> set:
        return set()
    
    def build_instance(self, bm, geoscn, reproject, ray_caster:DropToGround = None, build_parameters:dict = {}) -> None:
        return

    def _build_instance(self, bm, geoscn, reproject, ray_caster:DropToGround = None, build_parameters:dict={})->bmesh:
        outline_points = self.part_of.get_points()
        min_z = min(p[2] for p in outline_points)
        plant_verts = self.get_vertices(bm, geoscn=geoscn, reproject=reproject, ray_caster=ray_caster)
        # min_z = min(p.co[2] for p in plant_verts)
        #This should only be useful for building parts
        min_height = float(self._tags['min_height']) if 'min_height' in self._tags else float(self._tags.get('min_level',0))*build_parameters.get('level_height',3)
        for p in plant_verts:
            p.co.z=min_z+min_height
        #bmesh.ops.translate(bm, verts=plant_verts, vec=(0, 0, min_z+min_height))
        
        #plant edges
        shifted_vert = itertools.cycle(plant_verts)
        next(shifted_vert)
        edges = [
                bm.edges.new( v )
                for v in zip(plant_verts,shifted_vert)]
        
        face = bm.faces.new(plant_verts)
        
        #ensure face is up (anticlockwise order)
        #because in OSM there is no particular order for closed ways
        
        face.normal_update()
        if face.normal.z > 0:
            face.normal_flip()

        building_height = self.get_height(build_parameters=build_parameters)

        
        building_height -=min_height

        #Extrude
        
        if build_parameters.get('extrusion_axis', 'Z') == 'NORMAL':
            normal = face.normal
            vect = normal * building_height
        elif build_parameters.get('extrusion_axis', 'Z') == 'Z':
            vect = (0, 0, building_height)

        # extrusion = bmesh.ops.extrude_edge_only(bm, edges = edges)
        # # bmesh.ops.extrude_face_region(bm, faces=[face]) #return {'faces': [BMFace]} extrude_edge_only
        # verts = [v for v in extrusion['geom'] if isinstance(v,bmesh.types.BMVert)]
        # edges = [v for v in extrusion['geom'] if isinstance(v,bmesh.types.BMEdge)]
    
        extrusion = bmesh.ops.extrude_face_region(bm, geom=[face]) #return {'faces': [BMFace]} extrude_edge_only

        faces = [v for v in extrusion['geom'] if isinstance(v,bmesh.types.BMFace)]
        verts = [v for v in faces[0].verts]
        edges = [e for e in faces[0].edges]
        
        if ray_caster:
            #Making flat roof
            z = max([v.co.z for v in verts]) + building_height #get max z coord
            for v in verts:
                v.co.z = z
        else:
            bmesh.ops.translate(bm, verts=verts, vec=vect)

        
        self.build_roof(bm, verts, edges, build_parameters)
        return bm

class OSMRoof():
    
    directions:ClassVar[Vector] = {
        'N': Vector((0., 1., 0.)),
        'NNE': Vector((0.38268, 0.92388, 0.)),
        'NE': Vector((0.70711, 0.70711, 0.)),
        'ENE': Vector((0.92388, 0.38268, 0.)),
        'E': Vector((1., 0., 0.)),
        'ESE': Vector((0.92388, -0.38268, 0.)),
        'SE': Vector((0.70711, -0.70711, 0.)),
        'SSE': Vector((0.38268, -0.92388, 0.)),
        'S': Vector((0., -1., 0.)),
        'SSW': Vector((-0.38268, -0.92388, 0.)),
        'SW': Vector((-0.70711, -0.70711, 0.)),
        'WSW': Vector((-0.92388, -0.38268, 0.)),
        'W': Vector((-1., 0., 0.)),
        'WNW': Vector((-0.92388, 0.38268, 0.)),
        'NW': Vector((-0.70711, 0.70711, 0.)),
        'NNW': Vector((-0.38268, 0.92388, 0.))
    }
    roof_shape:ClassVar[str]=''
    height_tags: ClassVar[list[str]] = ['roof:height', 'roof:angle', 'roof:levels']

    _element: OSMElement = None

    def get_height(self, build_parameters:dict ={}, **kwargs):
        height_tag = next((k for k in self.height_tags if k in kwargs),None)
        if height_tag == 'roof:levels':
            return self.kwargs[height_tag]*build_parameters.get('level_height',3)
        if height_tag == 'roof:height':
            return self.kwargs[height_tag]
        return build_parameters.get('default_roof_height', 5)
    
    @property
    def orientation(self)->str:
        return self._element._tags.get('roof:orientation', 'along')

    @property
    def direction(self)->Vector|None:
        direction = self._element._tags.get('roof:direction', None)
        if direction is None:
            return direction
        if direction in self.directions:
            return self.directions[direction]
        direction = parse_measurement(direction)
        if direction:
            d = math.radians(d)
            d = Vector((math.sin(d), math.cos(d), 0.))
        return None
    
    def __init__(self, element:OSMElement) -> None:
        self._element = element

    @classmethod
    def get_roof_builder(cls, element:OSMElement)->Type[OSMRoof]:
        return next((r for r in all_subclasses(cls) if r.is_valid(element=element)),None)
    
    @classmethod
    def is_valid(cls, element:OSMElement)->bool:
        return element._tags.get('roof:shape',None) == cls.roof_shape

    @abstractmethod
    def build_roof(self, bm, roof_verts:list[bmesh.types.BMVert], roof_edges:list[bmesh.types.BMEdge], build_parameters:dict={}, **kwargs):
        pass

class RidgedRoof(OSMRoof):
    
    _ridges:ClassVar[OrderedDict] = OrderedDict()

    _fixed_heights: ClassVar[list[tuple[float,float]]] = []

    # used during build
    _ridge_segments: list[list] =[]

    _direction: Vector = None

    _width: Number = None

    def _clear_build_variables(self):
        self._ridge_segments = []
        self._direction= None
        self._width = None

    def get_direction(self, roof_verts: list[Vector], force_recalculate = False)->Vector:
        """
        Returns the main direction of the way. If already calculated returns the previously calculated value
        
        :param list[Vector] roof_verts: List of the coordinates for all vertices in the roof
        :param bool force_recalculate: Recalculated the direction even if a previous calculation is already found
        :return: The main direction of the way
        :rtyp: Vector

        """
        if self._direction and not force_recalculate:
            return self._direction
        
        d = self.direction
        if d is None:
            orientation = self.orientation
            d = find_longest_direction([v for v in roof_verts])
            if orientation == 'along':
                d.rotate(Quaternion((0.0, 0.0, 1.0), math.radians(90.0)))
        d.normalize()
        self._direction = d
        return d
    
    def calculate_relative_position(self, vertex: Vector, direction: Vector) -> Number:
        """
        Returns the relative position of the given point along the provided direction
        
        :param Vector vertex: vertex for which to find the relative position
        :param Vector direction: direction along which to find the position
        :return: the position along the given direction of the point
        """
        return direction[0]*vertex[0] + direction[1]*vertex[1] 
    
    def calculate_relative_positions(self, vertices: list[Vector]):
        """
        Returns the relative position of the given points along the main direction of the points
        
        :param list[Vector] vertices: vertices for which to find the relative position
        :return: the position along the main direction of the vertices
        """
        direction = self.get_direction(remove_straight_angles([v for v in vertices]))
        return [self.calculate_relative_position(v, direction) for v in vertices]
    
    def get_relative_height(self, normalized_position: Number):
        """
        Returns the relative height (from 0 to 1) based on the normalized position along the direction
        :param Number normalized_position: Position along the x axis between 0 and 1
        :return: The relative position of the vector compared to the total height of the roof, between 0 and 1
        :rtype: Number
        """
        previous_fixed=None
        next_fixed=None
        for xy in self._fixed_heights:
            if previous_fixed is None and normalized_position>=xy[0]:
                previous_fixed = xy
            if normalized_position <= xy[0]:
                next_fixed = xy
        return normalized_position*(next_fixed[1]-previous_fixed[1])/(next_fixed[0]-previous_fixed[0])

    def _create_projected_vertex(self, bm, plant_position: Vector, normalized_position:Number, full_height:Number)-> bmesh.types.BMVert:
        """
        Create a new vertex at the specified point, at the correct roof height for the position. Also assigned the vertex to the correct ridge segment
        
        :param BMesh bm: Bmesh with the roof
        :param Vector plant_position: vector with the xy coordinates of the point to project
        :param Number normalized_position: Normalized position across the roof profile (Between 0 and 1)
        :return: Created Vertex
        :rtype: BMVert
        """
        delta_z = self.get_relative_height(normalized_position = normalized_position)
        if delta_z>0.0005:
            coords = Vector(plant_position)
            coords.z = coords.z+delta_z*full_height
            
            vertex = bm.verts.new(coords)
            self._assign_to_ridges(vertex=vertex, normalized_position=normalized_position)
        return None


    def build_roof(self, bm, roof_verts:list[bmesh.types.BMVert], roof_edges:list[bmesh.types.BMEdge], build_parameters:dict={}, **kwargs):
        height = self.get_height(build_parameters=build_parameters, **kwargs)

        relative_positions = self.calculate_relative_positions([v.co for v in roof_verts])

        #get the full width of the plant and normalize the relative x-position
        min_position = min(relative_positions)
        max_position = max(relative_positions)
        self._width = max_position- min_position
        normalized_positions = [(p-min_position)/self._width for p in relative_positions]
        

        first_relative_position = previous_relative_position = normalized_positions[0]
        first_vertex = previous_vertex = roof_verts[0]
        first_roof_vertex = previous_roof_vertex = self._create_projected_vertex(bm, previous_vertex.co, previous_relative_position, height)
        current_vertex = None
        current_roof_vertex = None

        self._ridge_segments = [[] for _ in range(len(self._fixed_heights)-1)]

        for current_relative_position, current_vertex in zip(normalized_positions[1:], roof_verts[1:]):

            current_roof_vertex = self._create_projected_vertex(bm, current_vertex, current_relative_position, height)

            self._build_roof_segment(bm, 
                                     current_vertex, current_roof_vertex,current_relative_position,
                                       previous_vertex, previous_roof_vertex, previous_relative_position,
                                         height)
            
            previous_roof_vertex = current_roof_vertex
            previous_vertex=current_vertex
            previous_relative_position = current_relative_position

            current_roof_vertex = None
            current_vertex=None
            current_relative_position=None
        
        self._build_roof_segment(bm, first_vertex, first_roof_vertex, first_relative_position,
                                  previous_vertex, previous_roof_vertex, previous_relative_position,
                                  height)

        for r in self._ridge_segments:
            
            if len(r)>=3:
                bm.faces.new(r)
        
        return {'verts':[],
                'edges':[],
                'faces':[]}
    

    def _build_roof_segment(self, bm, 
                            current_vertex: bmesh.types.BMVert, 
                            current_roof_vertex: bmesh.types.BMVert| None, 
                            current_normalized_position: Number,
                            previous_vertex: bmesh.types.BMVert, 
                            previous_roof_vertex: bmesh.types.BMVert | None,
                            previous_normalized_position: Number,
                            roof_height: Number):
        wall_extension_vertices=[current_vertex, previous_vertex]
        if previous_roof_vertex:
            wall_extension_vertices.append(previous_roof_vertex)
        

        crossed_fixed_points = self._get_intermediate_ridge_points(previous_normalized_position, current_normalized_position)
        edge_vector = current_vertex.co - previous_vertex.co
        for crossed_fixed_point in crossed_fixed_points:
            ratio = (previous_normalized_position-crossed_fixed_point[0])/(previous_normalized_position-current_normalized_position)
            coords = previous_vertex.co + edge_vector*ratio
            coords.z = coords.z+crossed_fixed_point[1]*roof_height
            midway_vertex = bm.verts.new(coords)
            wall_extension_vertices.append( midway_vertex)
                #self._create_projected_vertex(midway_vertex)
                # bm, 
                #                               plant_position=coords, 
                #                               normalized_position = crossed_fixed_point[0], 
                #                               full_height= roof_height))
            #    

            self._assign_to_ridges(midway_vertex, crossed_fixed_point[0])
        
        if current_roof_vertex:
            wall_extension_vertices.append(current_roof_vertex)
            #self._assign_to_ridges(current_roof_vertex, current_normalized_position)
        else:
            self._assign_to_ridges(current_vertex, current_normalized_position)

        if len(wall_extension_vertices)>=3:
            return bm.faces.new(wall_extension_vertices)
        
        return None    
        


    def _get_intermediate_ridge_points(self,previous_normalized_position:Number, current_normalized_position:Number):
        crossed = lambda p1, p2, x: (p1-x)*(p2-x)<0
        crossed_fixed_points = [fp for fp in self._fixed_heights if crossed(previous_normalized_position, current_normalized_position, fp[0])]
        if previous_normalized_position>current_normalized_position:
            crossed_fixed_points.reverse()
        return crossed_fixed_points
    
    def _get_ridge_endpoints(self, i:Number):
        return (self._fixed_heights[i], self._fixed_heights[i+1])
    
    def _assign_to_ridges(self, vertex, normalized_position):
        for i in range(len(self._fixed_heights)-1):
            endpoints = [pt[0] for pt in self._get_ridge_endpoints(i)]
            endpoints.sort()
            if endpoints[0] <= normalized_position <= endpoints[1]:
                self._ridge_segments[i].append(vertex)

class OSMFlatRoof(OSMRoof):
    roof_shape:ClassVar[str]='flat'

    def __init__(self, element:OSMElement) -> None:
        super().__init__(element)

    def build_roof(self, bm, roof_verts:list[bmesh.types.BMVert], roof_edges:list[bmesh.types.BMEdge], build_parameters:dict={}, **kwargs):
        fill = bmesh.ops.triangle_fill(bm, use_beauty=True, use_dissolve=True, edges=roof_edges)
        return fill

class OSMPyramidalRoof(OSMRoof):
    roof_shape:ClassVar[str]='pyramidal'
    def __init__(self, element:OSMElement) -> None:
        super().__init__(element)

    def build_roof(self, bm, roof_verts:list[bmesh.types.BMVert], roof_edges:list[bmesh.types.BMEdge], build_parameters:dict={}, **kwargs):

        height = self.get_height(build_parameters=build_parameters, **kwargs)
        minimal_points = remove_straight_angles([v.co for v in roof_verts])

        center = find_geometric_center(minimal_points)
        center.z += height

        center_v = bm.verts.new(center)

        
        edges = [
                bm.edges.new( (v, center_v) )
                for v in roof_verts]
        
        shifted_vert = itertools.cycle(roof_verts)
        next(shifted_vert)
        faces = [bm.faces.new(verts)
                 for verts in zip(roof_verts, shifted_vert, itertools.repeat(center_v))]
        
        return {'verts':[center_v],
                'edges':edges,
                'faces':faces}
            
class OSMGabledRoof(RidgedRoof):
    roof_shape:ClassVar[str]='gabled'

    _fixed_heights = [
         (0.0, 0.0),
         (0.5, 1.0),
         (1.0, 0.0),
         ]
    
        
    def __init__(self, element:OSMElement) -> None:
        super().__init__(element)

class OSMGambreledRoof(RidgedRoof):
    roof_shape:ClassVar[str]='gambrel'

    _fixed_heights = [
         (0.0, 0.0),
         (0.25, 0.7),
         (0.5, 1.0),
         (0.75, 0.7),
         (1.0, 0.0),
         ]
    
        
    def __init__(self, element:OSMElement) -> None:
        super().__init__(element)

class OSMSkillionRoof(RidgedRoof):
    _fixed_heights = [
         (0.0, 1.0),
         (1.0, 0.0),
         ]
    def __init__(self, element:OSMElement) -> None:
        super().__init__(element)

class OSMRoundRoof(OSMRoof):
    roof_shape:ClassVar[str]='round'
    _fixed_heights = [(round(1-math.cos((math.radians(theta))),3),round(math.sin(math.radians(theta)),3)) for theta in range(0,190,10)]
    def __init__(self, element:OSMElement) -> None:
        super().__init__(element)

class OSMHippedRoof(OSMRoof):
    roof_shape:ClassVar[str]='hipped'

    def __init__(self, element:OSMElement) -> None:
        super().__init__(element)

    def build_roof(self, bm, roof_verts:list[bmesh.types.BMVert], roof_edges:list[bmesh.types.BMEdge], build_parameters:dict={}, **kwargs):
        dest_mesh = bpy.data.meshes.new("dest_mesh")
        
        height = self.get_height(build_parameters=build_parameters, **kwargs)

        shift_vertex = find_geometric_center([v.co for v in roof_verts])
        plane_matrix = straightSkeletonOfPolygon([v.co-shift_vertex for v in roof_verts], dest_mesh, height=height, tollerance=0.0001)
        dest_mesh.transform(plane_matrix)
        bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=0.01)
        print(type(dest_mesh))
        for v in dest_mesh.vertices:
            v.co += shift_vertex
        # bmesh.ops.translate(dest_mesh, verts=dest_mesh.vertices, vec=shift_vertex)
        # roof_level = min(v.co.z for v in roof_verts)
        # if min(v.co.z for v in dest_mesh.vertices) < roof_level:
        #     roof_level = min(v.co.z for v in roof_verts)
        #     for v in (vert for vert in dest_mesh.vertices if vert.co[2]<roof_level):
        #         v.co.z = roof_level + (roof_level-v.co.z)
            
        bm.from_mesh(dest_mesh)

        # for f in bm.faces:
        #     f.normal_update()


class OSMDomedRoof(OSMRoof):
    roof_shape:ClassVar[str]='dome'

    number_of_segments: Number = 10
    
    _steps = None

    @property
    def steps(self):
        if self._steps is None:
            step = (math.pi/2)/self.number_of_segments
            self._steps = [(math.cos(idx*step), math.sin(idx*step)) for idx in range(1, self.number_of_segments) ]
            
        return self._steps

    # def get_height(self, build_parameters: dict = {}, **kwargs):
    #     height_tag = next((k for k in self.height_tags if k in kwargs),None)
    #     if height_tag is not None:
    #         return super().get_height(build_parameters, **kwargs)
    #     return abs((center-roof_verts[0].co).length)
    
    def __init__(self, element:OSMElement) -> None:
        super().__init__(element)

    def build_roof(self, bm, roof_verts:list[bmesh.types.BMVert], roof_edges:list[bmesh.types.BMEdge], build_parameters:dict={}, **kwargs):
        
        minimal_points = remove_straight_angles([v.co for v in roof_verts])

        center = find_geometric_center(minimal_points)
        
        height_tag = next((k for k in self.height_tags if k in kwargs),None)
        if height_tag is not None:
            height = super().get_height(build_parameters, **kwargs)
        else:
            height = abs((center-roof_verts[0].co).length)

        center.z += height

        center_v = bm.verts.new(center)
        
        first_vertices, edges = self._build_ridge(bm, roof_verts[0], center_v)
        previous_vertices = first_vertices
        faces = []
        for vertex in roof_verts[1:]:
            vertices, edges = self._build_ridge(bm, vertex, center_v)
            faces.extend([bm.faces.new(vs) for vs in zip(previous_vertices[:-1], vertices[:-1], vertices[1:], previous_vertices[1:])])
            faces.append(bm.faces.new((previous_vertices[-1], vertices[-1], center_v)))
            previous_vertices = vertices
            vertices = []
        
        faces.extend([bm.faces.new(vs) for vs in zip(previous_vertices[:-1], first_vertices[:-1], first_vertices[1:], previous_vertices[1:])])
        faces.append(bm.faces.new((previous_vertices[-1], first_vertices[-1], center_v)))

        return {'verts':[center_v],
                'edges':edges,
                'faces':faces}
    
    def _build_ridge(self,bm, first_vertex, center):
        direction = first_vertex.co - center.co
        direction.z = - direction.z
        vertices = [first_vertex]
        height = center.co.z-first_vertex.co.z
        vertices.extend(bm.verts.new(center.co+Vector((direction.x*s[0],direction.y*s[0],direction.z*s[1]-height))) for s in self.steps)

        edges = [bm.edges.new(v) for v in zip(vertices[:-1],vertices[1:])]
        edges.append(bm.edges.new((vertices[-1], center)))
        return vertices, edges


    def _build_segment(self, first_vertex, second_vertex, center):
        first_vertex_direction = center-first_vertex
        second_vertex_direction = center-second_vertex
        
        step = (math.pi/2)/self.number_of_segments
        for theta in range(0,(math.pi/2), step):
            first_coord = (first_vertex_direction.co) * math.cos(theta)
            second_coord = (second_vertex_direction.co) * math.cos(theta)
            

class OSMOnionRoof(OSMRoof):
    def __init__(self) -> None:
        super().__init__()