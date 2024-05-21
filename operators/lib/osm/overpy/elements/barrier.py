from __future__ import annotations
import cProfile, pstats, io
from operator import itemgetter
from collections import OrderedDict, deque
from pstats import SortKey
from dataclasses import dataclass
from itertools import groupby, takewhile
import itertools
import math

from numbers import Number
import pprint
import random
from typing import ClassVar, TypeVar
from xml.etree.ElementTree import Element

from .....utils.bgis_utils import DropToGround, RayCastHit, parse_measurement
from .....utils.blender import appendObjectsFromAssets, convert_obj_to_curve,convert_curve_to_obj, createCollection, almost_overlapping, merge_splines
from .node import OSMNode
from .way import OSMWay

from mathutils import Vector
from mathutils.kdtree import KDTree
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


T = TypeVar('T', bound='OSMBarrier')

@dataclass
class BarrierContinuation:
    before: list[OSMBarrier]
    after: list[OSMBarrier]

class OSMBarrier(OSMWay):
    '''A barrier is a physical structure which blocks or impedes movement. The barrier tag only covers on-the-ground barriers. 
    It does not cover typical waterway barriers (dams, waterfalls, etc.). However, barriers that are normally found on land (such as fences) can also be found (and thus tagged) in water.
    '''
    blender_mesh_name: ClassVar[str] = "Barrier"
    _osm_sub_name: ClassVar[str] = 'barrier'
    _osm_barrier_type: str = None
    detail_level: ClassVar[int] = -1 # barrier should be an abstract class and should never be used not subclassed

    _height = 1
    _width = 1
    
    def get_height(self, build_parameters)->int:
        barrier_height=0
        if "height" in self._tags:
                barrier_height = parse_measurement(self._tags["height"])
        else:
            barrier_height = self._height
            
        return barrier_height
    
    def get_width(self, build_parameters)->int:
        barrier_width=0
        if "width" in self._tags:
                barrier_width = parse_measurement(self._tags["width"])
        else:
            barrier_width = self._width
            
        return barrier_width
    

    # Blender file with way profiles
    assetFile: ClassVar[str] = "barrier_profiles.blend"
    collectionName: ClassVar[str] = "barrier_profiles"

    @property
    def layer(self):
        return int(self._tags.get('layer',0))
    
    @property
    def barrier_type(self):
        return self._tags['barrier']
    
    @property
    def barrier_base_type(self):
        return self._tags['barrier'].removesuffix('_link')
    
    @property
    def bevel_name(self):
        return f'profile_{self.barrier_type}'
    
    @classmethod
    def _bevel_name(cls, barrier_type):
        return f'profile_barrier'
    
    def __str__(self):
        return f"OSMWay of barrier with id: {self._id}, made up of {len(self._node_ids)} nodes(s) and tags:\n{pprint.pformat(self._tags)}"

    def __init__(self, **kwargs):
        super(OSMBarrier,self).__init__(**kwargs)
    
    @classmethod
    def is_valid_xml(cls, xml_element:Element) -> bool:
        return super(OSMBarrier, cls).is_valid_xml(xml_element) and any(
                c.attrib['k'] == cls._osm_sub_name for c in xml_element.iter('tag'))

    @classmethod
    def is_valid_json(cls, json_element:dict) -> bool:
        return super(OSMBarrier, cls).is_valid_json(json_element) and cls._osm_sub_name in json_element.get('tags',{})

    @classmethod
    def load_barrier_profiles(cls):
        if cls.collectionName not in bpy.data.collections:
            collection = createCollection(
                        cls.collectionName,
                        hide_viewport=True,
                        hide_select=True,
                        hide_render=True
            )
        else:
            collection = bpy.data.collections.get(cls.collectionName)

        bevelObjs = appendObjectsFromAssets(cls.assetFile, collection, 'profile.*') #\.[0-9]{3}')
        for bevelObj in bevelObjs:
            bevelObj.hide_viewport = True
            bevelObj.hide_select = True
            bevelObj.hide_render = True
            
    @classmethod
    def preprocess(cls, library: OSMLibrary, ray_caster:DropToGround):
        barriers = library.get_elements(cls).values()
        for part in barriers:
            part.preprocess_instance(geoscn = library.geo_scene, ray_caster=ray_caster)
        

    def preprocess_instance(self, geoscn, ray_caster:DropToGround):#, bridges: tuple['BVHTree',list[int]]
        """Preprocess the barrier by doing the following in order:
        - Adding a reference to the barrier in all nodes referenced
        - add references to before and after barriers"""
        #from .man_made import OSMBridge
        self._is_preprocessed = False
        super(OSMBarrier,self).preprocess_instance(geoscn, ray_caster)
        # if 'bridge' in self._tags:
        #     print(f'Looking for bridges for {self._id}')
        #     self.find_assigned_bridges()
        
        self._is_preprocessed = True
        return

    @classmethod
    def build(cls, library: OSMLibrary, geoscn, reproject, ray_caster:DropToGround=None, build_parameters:dict=None) -> set[bpy.types.Object]:
        from .man_made import OSMBridge
        build_parameters = build_parameters or {}
        cls.load_barrier_profiles()
        meshes = dict()
        barriers = library.get_elements(cls).values()
        pr = cProfile.Profile()
        pr.enable()
        built_objects = []
        for part in barriers:
            if part.is_built or not part._is_valid:
                continue

            bm = bmesh.new()
            bm.verts.layers.string.new('node')
            built_objects.append(part._build_barrier_with_graph(bm, geoscn, reproject, ray_caster, build_parameters))

        
        return set(built_objects)
        # barrier_objects = set()
        # for barrier_type,barrier_bmesh in meshes.items():
        #     if len(barrier_bmesh.verts)== 0:
        #         barrier_bmesh.free()
        #         continue
        #     barrier_type_object = bpy.context.scene.objects.get(f"OSMBarriers_{barrier_type}")
        #     if barrier_type_object is None:
        #         barrier_type_object = bpy.data.objects.new(barrier_type,bpy.data.meshes.new(barrier_type))
        #         geoscn.scn.collection.objects.link(barrier_type_object)

        #     # Assign the mesh to the new object, convert to curve and then back to object
        #     bmesh.ops.remove_doubles(barrier_bmesh, verts=barrier_bmesh.verts, dist=0.0001)
        #     barrier_bmesh.to_mesh(barrier_type_object.data)
        #     barrier_bmesh.free()
        #     convert_obj_to_curve(barrier_type_object)
        #     print(f'HIGHWAY IS NOW TYPE {type(barrier_type_object)}')
        #     barrier_type_object.select_set(True)
        #     # bpy.ops.object.convert(target='CURVE', thickness = 1)

        #     barrier_type_object.data.bevel_object = bpy.data.objects.get(cls._bevel_name(barrier_type))
        #     if hasattr(barrier_type_object.data, "bevel_mode"):
        #         barrier_type_object.data.bevel_mode = 'OBJECT'
        #     if hasattr(barrier_type_object.data,"use_fill_caps"):
        #         barrier_type_object.data.use_fill_caps = True
        #     if hasattr(barrier_type_object.data,"twist_mode"):
        #         barrier_type_object.data.twist_mode = 'Z_UP'
        #     convert_curve_to_obj(barrier_type_object)
        #     barrier_objects.add(barrier_type_object)
        
        # pr.disable()
        # s = io.StringIO()
        # sortby = SortKey.CUMULATIVE
        # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        # ps.print_stats()

    
    def _get_full_barrier_pieces(self, current_piece: OSMBarrier, barrier_pieces: set(OSMBarrier)):
        # get all un built barriers starting from the given one which are not already in the provided set and which 
        # share a node with the given one
        filter_barriers = lambda h: not h.is_built and h.barrier_base_type == current_piece.barrier_base_type


        adjacent_pieces = set([e for node in current_piece.nodes 
                               for e in self._library.get_elements_by_ids(list(node.get_referenced_from(type(current_piece)))) 
                               if filter_barriers(e)
                               ])
        unsearched = adjacent_pieces.difference(barrier_pieces)
        barrier_pieces.update(unsearched)
        for new_piece in unsearched:
            self._get_full_barrier_pieces(new_piece, barrier_pieces)

    def _build_barrier_with_graph(self, bm, geoscn, reproject, ray_caster:DropToGround = None, build_parameters:dict=None)->bmesh:
        #Get all barriers pieces which are either of the same type of links
        build_parameters = build_parameters or {}
        built_barriers = set([self])
        self._get_full_barrier_pieces(self, built_barriers)
        
        #Generate a graph from all the pieces
        graph = self._generate_graph(bm, barriers = built_barriers)

        # start off by building all the main part of the barriers
        # #start wth the first non built node with edges of the base type
        # breadth first build (flood fill) only the base type
        # check if there are other base type un built edges, and restart flood filling
        # from that node     
        # fill in the links, where starting and ending node have to either hit the ground or be connected ot another base barrier

        barrier_base_type = self.barrier_base_type
        built = set()
        for starting_node in takewhile(lambda x: x is not None , graph.get_unbuilt_nodes([barrier_base_type])):
            self._breadth_first_build_graph(bm, ray_caster=ray_caster, graph=graph, start = starting_node, built = built, valid_edges = [barrier_base_type], build_parameters=build_parameters)
        

        built = set()

        for barrier in built_barriers:
            barrier.is_built = True
            if barrier.closed:
                starting_vertex = next((bm.verts[i] for i in range(len(bm.verts)) if bm.verts[i][bm.verts.layers.string.get("node")].decode() == str(barrier._node_ids[0])), None)
                ending_vertex = next((bm.verts[i] for i in range(len(bm.verts)) if bm.verts[i][bm.verts.layers.string.get("node")].decode() == str(barrier._node_ids[-2])), None)
                if starting_vertex is not None and ending_vertex is not None:
                    try:
                        bm.edges.new((starting_vertex, ending_vertex))
                    except:
                        print('Edge already exists')
        
        #bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.01)
        #mesh = bpy.data.meshes.new(f"{self._id}")
        #bm.to_mesh(mesh)
        #bm.free()
        #mesh.update()#calc_edges=True)
        #mesh.validate()
        #obj = bpy.data.objects.new(f"{self._id}", mesh)
        
        #geoscn.scn.collection.objects.link(obj)
        #obj.select_set(True)
        #return obj
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0001)
        barrier_temp_object = bpy.data.objects.new(f"{self._id}", bpy.data.meshes.new(f"{self._id}"))
        geoscn.scn.collection.objects.link(barrier_temp_object)

        bm.to_mesh(barrier_temp_object.data)
        bm.free()
        convert_obj_to_curve(barrier_temp_object)
        barrier_temp_object.select_set(True)

        bevel_object = bpy.data.objects.get(self._bevel_name(self.barrier_type)).copy()
        bevel_object.scale=Vector((self._width, self._height,1.0))
        barrier_temp_object.data.bevel_object = bevel_object
        if hasattr(barrier_temp_object.data, "bevel_mode"):
            barrier_temp_object.data.bevel_mode = 'OBJECT'
        if hasattr(barrier_temp_object.data,"use_fill_caps"):
            barrier_temp_object.data.use_fill_caps = True
        if hasattr(barrier_temp_object.data,"twist_mode"):
            barrier_temp_object.data.twist_mode = 'Z_UP'
        convert_curve_to_obj(barrier_temp_object)
        return barrier_temp_object
        


    def _breadth_first_build_graph(self, bm:BMesh, ray_caster: DropToGround, graph: BarrierGraph, start:BarrierGraphNode, built:set = None, valid_edges:list(str)=None, build_parameters:dict=None):

        print(f'Setting off breadth first building, starting node is {start.node}')
        built = built or set()
        valid_edges = valid_edges or []
        build_parameters = build_parameters or {}
        queue = deque()
        start.build(bm)
        
        built.add(start)
        queue.append(start)
        
        while queue:
             
            current_node = queue.popleft()
            for (neighbor, edge) in ( (n,e) for (n,e) in graph.get_valid_neighbors(current_node, valid_edges) if n not in built):

                neighbor.build(bm)
                
                self._build_barrier_graph_edge(start= current_node, end=neighbor, edge = edge, bm=bm, ray_caster=ray_caster, build_parameters=build_parameters)
                built.add(neighbor)
                queue.append(neighbor)
    
    def _build_barrier_graph_edge(self, start: BarrierGraphNode, end:BarrierGraphNode, edge: BarrierGraphEdge, bm:BMesh, ray_caster: DropToGround, build_parameters:dict=None)->list[BMVert]:
        # print(f'building edge between {start.id} and {end.id}')
        build_parameters = build_parameters or {}
        intermediate_points = self.subdivide_way([start.node.ray_cast_hit.loc,end.node.ray_cast_hit.loc], build_parameters.get('barrier_subdivision_size', None))[1:-1]
        built_vertices = []
        previous_vertex = start.vertex
        for i,v in enumerate(intermediate_points):
            rc_hit = ray_caster.rayCast(v[0], v[1])
            if not rc_hit.hit:
                previous_vertex = None
                continue

            next_vertex = bm.verts.new(rc_hit.loc)
            built_vertices.append(next_vertex)
            next_vertex[bm.verts.layers.string.get("node")] = str.encode(str(start.node._id)) if i<=len(intermediate_points)/2 else str.encode(str(end.node._id))
            if previous_vertex:
                bm.edges.new((previous_vertex, next_vertex))
            previous_vertex = next_vertex

        if end.vertex and previous_vertex:
            try:
                bm.edges.new((previous_vertex, end.vertex))
            except:
                print('Edge already exists')
        return built_vertices
            
    def _generate_graph(self, bm: bmesh, barriers:list())->BarrierGraph:
        graph = BarrierGraph(bm,self.barrier_type)
        
        for barrier in barriers:
            for (v1,v2) in zip(barrier.nodes[:-1],barrier.nodes[1:]):
                graph.add_edge(v1=v1, v2=v2, edge_tags=barrier._tags)
        return graph
        
    def get_subdivision_params(self, preceding_point: tuple[float,float], current_point:tuple[float,float], subdivision_size: Number)->tuple[int, Vector]:
        vec = Vector(current_point) - Vector(preceding_point)
        if subdivision_size is None:
            return 1, vec
        number_steps = max(math.ceil(vec.length/subdivision_size),1)
        return number_steps, vec/number_steps

class OSMBarrierSubType(OSMBarrier):
    """
    A restricted access major divided highway, normally with 2 or more running lanes plus emergency hard shoulder. Equivalent to the Freeway, Autobahn, etc..
    """
    _osm_barrier_type: str = 'city_wall'
    detail_level: ClassVar[int] = 3

    @classmethod
    def is_valid_xml(cls, xml_element:Element) -> bool:
        return super(OSMBarrier, cls).is_valid_xml(xml_element) and any(
                c.attrib['k'] == cls._osm_sub_name and c.attrib['v'] == cls._osm_barrier_type for c in xml_element.iter('tag'))

    @classmethod
    def is_valid_json(cls, json_element:dict) -> bool:
        return super(OSMBarrier, cls).is_valid_json(json_element) and json_element.get('tags',{}).get(cls._osm_sub_name,None)==cls._osm_barrier_type



class OSMCityWall(OSMBarrier):
    """
    A restricted access major divided barrier, normally with 2 or more running lanes plus emergency hard shoulder. Equivalent to the Freeway, Autobahn, etc..
    """
    _osm_barrier_type: str = 'city_wall'
    detail_level: ClassVar[int] = 3
    _height = 6
    _width = 4

class BarrierGraphNode():
    _node: OSMNode
    _vertex:BMVert = None
    
    @property
    def vertex(self)->BMVert:
        return self._vertex
    
    @vertex.setter
    def vertex(self, value):
        self._vertex=value
        self._node._blender_element = value

    is_built = False
    distance_to_closest_end:float

    @property
    def id(self):
        return self._node._id
    
    @property
    def node(self):
        return self._node
    
    @node.setter
    def node(self,value:OSMNode):
        self._node = value
        self.x = value.ray_cast_hit.loc.x
        self.y = value.ray_cast_hit.loc.y
        self.z = value.ray_cast_hit.loc.z if value.ray_cast_hit.hit else None

    _x = None
    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self,value:Number):
        self._x = value

    _y = None
    @property
    def y(self):
        return self._y
    
    @y.setter
    def y(self,value:Number):
        self._y = value

    _z = None
    @property
    def z(self):
        return self._z
    
    @z.setter
    def z(self,value:Number):
        self._z = value

    def __init__(self, node) -> None:
        self.node=node
        self.is_built=False
        self.distance_to_closest_end=-1


    def __hash__(self):
        return self.id

    def __eq__(self, other):
        if isinstance(other, int):
            return self.id==other 
        return self.id==other.id 
    
    def distance(self, other: BarrierGraphNode)->float:
        return ((self.x-other.x)**2+(self.y-other.y)**2+(self.z-other.z)**2)**0.5

    def distance2D(self, other: BarrierGraphNode)->float:
        return ((self.x-other.x)**2+(self.y-other.y)**2)**0.5
    
    def build(self, bm: BMesh)->BVert:
        self.is_built = True
        
        if self.vertex is None and self.node.ray_cast_hit.hit:
            self.vertex = bm.verts.new(self.node.ray_cast_hit.loc)
            bm.verts.ensure_lookup_table()
            self.vertex[bm.verts.layers.string.get("node")] = str.encode(str(self.node._id))

        return self.vertex

class BarrierGraphEdge():
    length:float
    tags:dict
    types=set()
    is_built = False

    vertices:list['BVert'] = []

    def vertices_from(self, starting_node:OSMNode):
        if starting_node.vertex is None:
            return None
        for edge in starting_node.vertex.link_edges():
            other_vertex = edge.other_vert(starting_node.vertex)
            if other_vertex==self.vertices[0]:
                return self.vertices
            elif other_vertex==self.vertices[-1]:
                return reversed(self.vertices)
        return None

    def is_continuation(self, required_tags:list)->bool:
        return all(t in self.types for t in required_tags)

    def __init__(self, length, tags) -> None:
        self.length = length
        self.tags = tags
        self.types = set()
        barrier_type = self.tags.get('barrier',None) or self.tags.get('railway',None)
        self.types.add(barrier_type)
        if self.tags.get('bridge',None) is not None:
            self.types.add('bridge')
        if self.tags.get('tunnel',None) is not None:
            self.types.add('tunnel')


class BarrierGraph():
    bm: BMesh = None
    barrier_type:str
    barrier_base_type:str
    adj_list:dict[BarrierGraphNode, list[tuple[BarrierGraphNode,BarrierGraphEdge]]]

    def __getitem__(self, key):
        return next((n for n in self.adj_list.keys() if str(n.id) == key), None)
    
    def __init__(self, bm: BMesh, barrier_type:str) -> None:
        self.bm = bm
        self.barrier_type = barrier_type
        self.barrier_type = barrier_type.removesuffix('_link')
        self.adj_list = {}

    def add_edge(self, v1:OSMNode, v2:OSMNode, edge_tags:dict):
        
        node_1 = next((n for n in self.adj_list.keys() if n.id==v1._id), None) or BarrierGraphNode(v1)
        node_2 = next((n for n in self.adj_list.keys() if n.id==v2._id), None) or BarrierGraphNode(v2)
        edge = BarrierGraphEdge(self._node_distance(node_1,node_2), edge_tags)
        if not any(n == node_2 for n,_ in self.adj_list.get(node_1,[])):
            self.adj_list.setdefault(node_1,[]).append((node_2,edge))
        if not any(n == node_1 for n,_ in self.adj_list.get(node_2,[])):
            self.adj_list.setdefault(node_2,[]).append((node_1,edge))

    def get_edge(self, v1:OSMNode, v2:OSMNode)->BarrierGraphEdge:
        return next((e for n,e in self.adj_list.get(v1) if n==v2), None)
    
    def _node_distance(self, v1:BarrierGraphNode, v2:BarrierGraphNode)->float:
        return math.sqrt((v1.x-v2.x)**2 + (v1.y-v2.y)**2)
    
    def get_node_by_id(self, id:str)->BarrierGraphNode:
        return next((n for n in self.adj_list.values() if n.id==id),None)
    
    def get_valid_neighbors(self, node:BarrierGraphNode, valid_edges:list[str] =[])->list[tuple[BarrierGraphNode,BarrierGraphEdge]]:
        return [(n,e) for (n,e) in self.adj_list[node] if e.is_continuation(valid_edges)]

    def dfs(self, start, target = None, path = [], visited = set(), valid_edges:list[str]=[]):
        path.append(start)
        visited.add(start)
        if target and start == target:
            return path
        for (neighbor, edge) in self.adj_list[start]:
            if neighbor not in visited and edge.is_continuation(valid_edges):
                result = self.dfs(neighbor, target, path, visited, valid_edges=valid_edges)
                if result:
                    return result
        path.pop()
        return None
    
    def bfs_explore(self, start_node, valid_edges = [])->dict[str,float]:
        """
        Get list of all possible paths by length. 
        """
        print(f'BFS explore starting from {start_node.id} and searching for edges of type {";".join(valid_edges)}')
        #visited = set()
        queue = deque()
        queue.append([str(start_node.id)])
        #visited.add(start_node)
        distances = {
            str(start_node.id):0
        }
        while queue:
            current_path = queue.popleft()
            current_node = self.__getitem__(current_path[-1])
            found_extension = False
            for (neighbor,edge) in self.adj_list[current_node]:
                # print(f'Looking at neighbor {neighbor.id} which has and edge of type {";".join(edge.types)}')
                if edge.is_continuation(valid_edges):
                    if str(neighbor.id) not in current_path:
                        found_extension = True
                        next_path = current_path+[str(neighbor.id)]
                        queue.append(next_path)
                        distances[','.join(next_path)] = distances[','.join(current_path)]+edge.length
            if found_extension:
                distances.pop(','.join(current_path),None)


        # all_paths = sorted(list(distances.keys()))
        # for i,path in enumerate(all_paths):
        #     if any(p.startswith(path) for p in all_paths[i+1:]):
        #         distances.pop(path)
        return distances



    def get_unbuilt_nodes(self, barrier_types:list = []):
        for node,edges in sorted(self.adj_list.items(), key=lambda x: len(x[1])):
            if any(not n.is_built and e.is_continuation(barrier_types) for n,e in edges): #not node.is_built and 
                yield node
        yield None