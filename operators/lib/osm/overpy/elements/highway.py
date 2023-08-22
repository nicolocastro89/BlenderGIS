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

from .....utils.bgis_utils import DropToGround, RayCastHit
from .....utils.blender import appendObjectsFromAssets, convert_obj_to_curve, createCollection, almost_overlapping, merge_splines
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


T = TypeVar('T', bound='OSMHighway')

@dataclass
class HighwayContinuation:
    before: list[OSMHighway]
    after: list[OSMHighway]

class OSMHighway(OSMWay):
    '''A Highway is any kind of road, street or path. This is a base class which should never be directly
    assigned to any element.
    '''

    _osm_sub_name: ClassVar[str] = 'highway'
    _osm_highway_type: str = None
    _highways_to_level:list[str] = ["bridge", 'tunnel']
    _highways_roads:ClassVar[list[str]] = ["motorway", 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified', 'residential', 'service', 'pedestrian', 'track', 'other']
    _highways_paths:list[str] = ["footway", 'steps', 'cycleway']
    detail_level: ClassVar[int] = 2

    # Blender file with way profiles
    assetFile: ClassVar[str] = "way_profiles.blend"
    collectionName: ClassVar[str] = "way_profiles"

    @property
    def layer(self):
        return int(self._tags.get('layer',0))
    
    @property
    def highway_type(self):
        return self._tags['highway']
    
    @property
    def highway_base_type(self):
        return self._tags['highway'].removesuffix('_link')
    
    @property
    def bevel_name(self):
        profile_subtype = self.highway_type
        if profile_subtype == 'motorway_link':
            profile_subtype = 'motorway'
        
        profile_type= "roads" if profile_subtype in self._highways_roads else  "paths"
        if profile_type=='paths':
            profile_subtype = 'footway'
        return f'profile_{profile_type}_{profile_subtype}'
    
    @classmethod
    def _bevel_name(cls, highway_type):
        profile_subtype = highway_type
        if profile_subtype == 'motorway_link':
            profile_subtype = 'motorway'
        
        profile_type= "roads" if profile_subtype in cls._highways_roads else  "paths"
        if profile_type=='paths':
            profile_subtype = 'footway'
        return f'profile_{profile_type}_{profile_subtype}'
    
    def __str__(self):
        return f"OSMWay of road with id: {self._id}, made up of {len(self._node_ids)} nodes(s) and tags:\n{pprint.pformat(self._tags)}"

    def __init__(self, **kwargs):
        super(OSMHighway,self).__init__(**kwargs)
    
    @classmethod
    def is_valid_xml(cls, xml_element:Element) -> bool:
        return super(OSMHighway, cls).is_valid_xml(xml_element) and any(
                c.attrib['k'] == cls._osm_sub_name for c in xml_element.iter('tag'))

    @classmethod
    def is_valid_json(cls, json_element:dict) -> bool:
        return super(OSMHighway, cls).is_valid_json(json_element) and cls._osm_sub_name in json_element.get('tags',{})

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
            
    @classmethod
    def preprocess(cls, library: OSMLibrary, ray_caster:DropToGround):
        highways = library.get_elements(cls).values()
        for part in highways:
            part.preprocess_instance(geoscn = library.geo_scene, ray_caster=ray_caster)

        from .man_made import OSMBridge
        bridges_bvh_tree, bridges_bvh_tree_indices = library.get_bvh_tree(OSMBridge)

        for part in (h for h in highways if 'bridge' in h._tags):
            bridge_indices = [bridges_bvh_tree.ray_cast((n._lat, n._lon, -1.), zAxis)[2] for n in part.nodes]
            bridge_ids = [bridges_bvh_tree_indices[idx] for idx in bridge_indices if idx is not None]
            for id in bridge_ids:
                bridge = library.get_element_by_id(id)
                print(f'Found bridge {id}')
                if part.layer==bridge.layer:
                    print(f'Found highway with correct layer for {id}: {part._id}')
                    bridge.add_reference(OSMHighway, part._id)
                    part.add_reference(OSMBridge, id)
        

    def preprocess_instance(self, geoscn, ray_caster:DropToGround):#, bridges: tuple['BVHTree',list[int]]
        """Preprocess the highway by doing the following in order:
        - Adding a reference to the highway in all nodes referenced
        - add references to before and after highways"""
        #from .man_made import OSMBridge
        self._is_preprocessed = False
        super(OSMHighway,self).preprocess_instance(geoscn, ray_caster)
        # if 'bridge' in self._tags:
        #     print(f'Looking for bridges for {self._id}')
        #     self.find_assigned_bridges()
        
        self._is_preprocessed = True
        return

    def find_assigned_bridges(self):
        from .man_made import OSMBridge
        bridges_bvh_tree, bridges_bvh_tree_indices = self._library.get_bvh_tree(OSMBridge)

        bridge_indices = [bridges_bvh_tree.ray_cast((n._lat, n._lon, -1.), zAxis)[2] for n in self._nodes]
        bridge_ids = [bridges_bvh_tree_indices[idx] for idx in bridge_indices if idx is not None]
        for id in bridge_ids:
            bridge = self._library.get_element_by_id(id)
            print(f'Found bridge {id}')
            if self.layer==bridge.layer:
                print(f'Found highway with correct layer for {id}: {self._id}')
                bridge.add_reference(OSMHighway, self._id)
                self.add_reference(OSMBridge, id)

    @classmethod
    def build(cls, library: OSMLibrary, geoscn, reproject, ray_caster:DropToGround=None, build_parameters:dict=None) -> None:
        from .man_made import OSMBridge
        build_parameters = build_parameters or {}
        cls.load_highway_profiles()
        meshes = dict()
        highways = library.get_elements(cls).values()
        pr = cProfile.Profile()
        pr.enable()
        for part in highways:
            if part.is_built or not part._is_valid:
                continue

            bm = meshes.setdefault(part.highway_base_type, bmesh.new())
            if bm.verts.layers.string.get('node', None) is None:
                bm.verts.layers.string.new('node')
            part._build_highway_with_graph(bm, geoscn, reproject, ray_caster, build_parameters)

        for part in (p for p in highways if 'bridge' in p._tags):
            bridges = set(library.get_elements_by_ids([ b for b in part.get_referenced_from(OSMBridge)]))
                        # part._referenced_by.get(OSMBridge, []))
            if any(not b.is_built for b in bridges):
                bridge_nodes = []
                bridge_vertices = []
                for p in (h for h in highways if h.tags.get('bridge',False) is not False):
                    bridge_nodes.extend((n._id for n in p.nodes))
                
                for bm in meshes.values():
                    layer = bm.verts.layers.string.get('node')
                    for v in bm.verts:
                        if int(v[layer].decode()) in bridge_nodes:
                            bridge_vertices.append(v)

                size = len(bridge_vertices)
                kd = KDTree(size)

                for i, v in enumerate(bridge_vertices):
                    kd.insert(Vector((v.co[0], v.co[1], 0)), i)
                kd.balance()
                for bridge in [b for b in bridges if not b.is_built]:
                    bridge_bm = bmesh.new()
                    bridge_bm = bridge._build_instance(
                                            bridge_bm, 
                                            geoscn=geoscn, 
                                            reproject=reproject, 
                                            ray_caster=ray_caster, 
                                            supported_highway_nodes = bridge_vertices, 
                                            kd_tree = kd,
                                            build_parameters=build_parameters)
                    
                    bmesh.ops.remove_doubles(bridge_bm, verts=bridge_bm.verts, dist=0.0001)
                    bridge_object = bpy.data.objects.new(str(bridge._id),bpy.data.meshes.new(str(bridge._id)))
                    geoscn.scn.collection.objects.link(bridge_object)
                    bridge._is_built = True

                    bridge_bm.to_mesh(bridge_object.data)
                    bridge_bm.free()

        
        for highway_type,highway_bmesh in meshes.items():
            if len(highway_bmesh.verts)== 0:
                highway_bmesh.free()
                continue
            obj = bpy.context.scene.objects.get(f"OSMHighways_{highway_type}")
            if obj is None:
                obj = bpy.data.objects.new(highway_type,bpy.data.meshes.new(highway_type))
                geoscn.scn.collection.objects.link(obj)

            bmesh.ops.remove_doubles(highway_bmesh, verts=highway_bmesh.verts, dist=0.0001)
            highway_bmesh.to_mesh(obj.data)
            highway_bmesh.free()
            convert_obj_to_curve(obj)
            obj.select_set(True)
            # bpy.ops.object.convert(target='CURVE', thickness = 1)
            obj.data.bevel_object = bpy.data.objects.get(cls._bevel_name(highway_type))
            if hasattr(obj.data, "bevel_mode"):
                obj.data.bevel_mode = 'OBJECT'
            if hasattr(obj.data,"use_fill_caps"):
                obj.data.use_fill_caps = True
            # if hasattr(obj.data,"dimensions"):
            #     obj.data.dimensions = '2D'
        
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(f'Profile of highway building')
        print(s.getvalue())
        return
    
    def _get_full_highway_pieces(self, current_piece: OSMHighway, highway_pieces: set(OSMHighway)):
        # get all un built highways starting from the given one which are not already in the provided set and which 
        # share a node with the given one
        filter_highways = lambda h: not h.is_built and h.highway_base_type == current_piece.highway_base_type


        adjacent_pieces = set([e for node in current_piece.nodes 
                               for e in self._library.get_elements_by_ids(list(node.get_referenced_from(type(current_piece)))) 
                               if filter_highways(e)
                               ])
        unsearched = adjacent_pieces.difference(highway_pieces)
        highway_pieces.update(unsearched)
        for new_piece in unsearched:
            self._get_full_highway_pieces(new_piece, highway_pieces)


    def _build_highway_with_graph(self, bm, geoscn, reproject, ray_caster:DropToGround = None, build_parameters:dict=None)->bmesh:
        #Get all highways pieces which are either of the same type of links
        build_parameters = build_parameters or {}
        built_highways = set([self])
        self._get_full_highway_pieces(self, built_highways)
        
        #Generate a graph from all the pieces
        graph = self._generate_graph(bm, highways = built_highways)

        # start off by building all the main part of the highways
        # #start wth the first non built node with edges of the base type
        # breadth first build (flood fill) only the base type
        # check if there are other base type un built edges, and restart flood filling
        # from that node     
        # fill in the links, where starting and ending node have to either hit the ground or be connected ot another base highway

        highway_base_type = self.highway_base_type
        built = set()
        for starting_node in takewhile(lambda x: x is not None , graph.get_unbuilt_nodes([highway_base_type])):
            self._breadth_first_build_graph(bm, ray_caster=ray_caster, graph=graph, start = starting_node, built = built, valid_edges = [highway_base_type], build_parameters=build_parameters)
        
        built = set()

        for starting_node in takewhile(lambda x: x is not None , graph.get_unbuilt_nodes([highway_base_type+'_link'])):
            self._breadth_first_build_graph(bm, ray_caster=ray_caster, graph=graph, start = starting_node, built = built, valid_edges = [highway_base_type+'_link'], build_parameters=build_parameters)
        
        for highway in built_highways:
            highway.is_built = True

    def _breadth_first_build_graph(self, bm:BMesh, ray_caster: DropToGround, graph: HighwayGraph, start:HighwayGraphNode, built:set = None, valid_edges:list(str)=None, build_parameters:dict=None):

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
                levelled=next((l for l in self._highways_to_level if l in edge.types), None)
                if levelled is not None:
                    neighbors = self._leveled_depth_first_build_graph(bm = bm,
                                                                      ray_caster=ray_caster,
                                                                      graph = graph,
                                                                      start_node=current_node,
                                                                      built = built,
                                                                      valid_edges=valid_edges+[levelled],
                                                                      build_parameters=build_parameters)
                    queue.extend(neighbors)
                else:
                    neighbor.build(bm)
                    
                    self._build_highway_graph_edge(start= current_node, end=neighbor, edge = edge, bm=bm, ray_caster=ray_caster, build_parameters=build_parameters)
                    built.add(neighbor)
                    queue.append(neighbor)
    
    def _leveled_depth_first_build_graph(self, bm:BMesh, ray_caster: DropToGround, graph: HighwayGraph, start_node:HighwayGraphNode, built = None, valid_edges:list[str]=None, build_parameters:dict=None):
        """
        Used to build leveled (bridge or tunnel) highways and their links
        """
        print(f'Building leveled bridge starting from {start_node.id}')
        built = built or set()
        valid_edges = valid_edges or None
        build_parameters = build_parameters or None

        built_paths = []

        # Get all the possible paths and sort them by length
        bfs_explore_result = graph.bfs_explore(start_node, valid_edges = valid_edges)
        # split the path and cast each id to int meanwhile order by length desc
        path_distances = [([int(id) for id in p.split(',')], d ) for p,d in sorted(bfs_explore_result.items(), key=itemgetter(1), reverse=True)]
        # print(f'Found {len(path_distances)} paths')
        
        for node in graph.adj_list.keys():
            for path,_ in path_distances:
                try:
                    path[path.index(node.id)] = node
                except:
                    continue
        
        #Build the longest path first
        #longest_distance = max(path_distances.values())
        #longest_path = next(p for p,d in longest_distance.items() if d == longest_distance)
        
        for longest_path, longest_distance in path_distances:
            print(f'Analysing path from {longest_path[0].id} to {longest_path[-1].id}')
            actual_path = []
            longest_to_build = 0

            # Find the longest path to build and start there
            for built_path in built_paths:
                overlap_truth_list = [n1 == n2 for n1,n2 in zip(longest_path, built_path)]
                try:
                    overlap_truth_index = overlap_truth_list.index(False)
                    
                    overlap_length = sum(built_path[i-1].distance2D(built_path[i]) for i in range(1,overlap_truth_index))
                    to_build_distance = longest_distance-overlap_length
                    if to_build_distance > longest_to_build:
                        longest_to_build = to_build_distance
                        # take the first built node
                        actual_path = longest_path[overlap_truth_index-1:]
                except ValueError:
                    continue
            if not actual_path:
                actual_path = longest_path
                longest_to_build = longest_distance
            if longest_to_build == 0:
                continue
            
            actual_path[0].build(bm)
            
            if any(graph.get_edge(p1, p2).tags.get('highway','').endswith('_link') for p1, p2 in zip(path[:-1],path[1:])):
                self._build_link(path=actual_path,
                                    path_length=longest_to_build,
                                    graph = graph,
                                    built= built,
                                    bm=bm,
                                    ray_caster=ray_caster,
                                    build_parameters=build_parameters)
            else:
                self._build_leveled_path(path=actual_path,
                                    path_length=longest_to_build,
                                    graph = graph,
                                    built= built,
                                    bm=bm,
                                    ray_caster=ray_caster,
                                    build_parameters=build_parameters)
            
            built_paths.append(longest_path)
        
        
        neighbors = set() 
        for p in built:
            for (n,_) in graph.get_valid_neighbors(p, [t for t in valid_edges if t not in self._highways_to_level]):
                if n not in built:
                    neighbors.add(p)

        return neighbors        
    
    def _build_leveled_path(self, path: list[HighwayGraphNode], path_length: float, graph: HighwayGraph, built: set[HighwayGraphNode], bm:BMesh, ray_caster: DropToGround, build_parameters:dict=None):
        print('\t'+f'building path starting at {path[0].id} and ending at {path[-1].id}')
        
        build_parameters = build_parameters or {}

        # Find the height of the first and last node to actually hit the ground
        starting_height = next((p.node.ray_cast_hit.loc[2] for p in path if p.node.ray_cast_hit.hit), 0)
        ending_height = next((p.node.ray_cast_hit.loc[2] for p in reversed(path) if p.node.ray_cast_hit.hit), 0)

        
        slope = (ending_height-starting_height)/path_length
        total_traveled_distance = 0
        layer_access_ramp_length = build_parameters.get('layer_acccess_ramp_length',50.)
        # do a uniform slope from one endpoint to the other
        for previous_graph_node, current_graph_node in zip(path[:-1],path[1:]):
            # This part is done at the beginning to make sure the traveled path is kept up to date even if the node
            #falls outside the area
            traveled_distance = previous_graph_node.distance2D(current_graph_node)
            
            total_traveled_distance += traveled_distance

            if not current_graph_node.node.ray_cast_hit.hit:
                current_graph_node.vertex = None
                current_graph_node.is_built = True
                built.add(current_graph_node)
                continue

            edge = graph.get_edge(previous_graph_node, current_graph_node)
                
            current_graph_node.build(bm)
            built.add(current_graph_node)
            
            if current_graph_node.vertex is None:
                continue
            
            current_graph_node.vertex.co[2] = starting_height + slope*total_traveled_distance 

            built_vertices = self._build_highway_graph_edge(start= previous_graph_node, end=current_graph_node, edge = None, bm=bm, ray_caster=ray_caster, build_parameters=build_parameters)
            edge.vertices = built_vertices

            #Align the points along the previous and current vertex
            
            for i, built_vertex in enumerate(edge.vertices, 1):
                if previous_graph_node.vertex is None:
                    height_shift = 0
                else:
                    distance_from_end = current_graph_node.distance2D(previous_graph_node)

                    height_shift = (len(built_vertices)-i)/(len(built_vertices)+1) * (current_graph_node.vertex.co[2] - previous_graph_node.vertex.co[2])

                built_vertex.co[2] = current_graph_node.vertex.co[2] - height_shift

        points_below = sum((1 for p in path if p.vertex is not None and p.vertex.co[2]-p.node.ray_cast_hit.loc[2]<2))
        

        if points_below/len(path)>0.1:
            total_traveled_distance = 0
            
            for previous_graph_node, current_graph_node in zip(path[:-1],path[1:]):
                if current_graph_node.vertex is None:
                    continue
                edge = graph.get_edge(previous_graph_node,current_graph_node)
                
                layer_height = int(edge.tags.get('layer', 1)) * build_parameters.get('layer_default_height',10)

                traveled_distance = previous_graph_node.distance2D(current_graph_node)
                
                total_traveled_distance += traveled_distance
                
                distance_from_end = min([abs(total_traveled_distance-path_length),total_traveled_distance, layer_access_ramp_length])
                ramp_ratio = distance_from_end/layer_access_ramp_length

                minimal_ramp_height = ramp_ratio*layer_height
                actual_z = max(current_graph_node.vertex.co[2], current_graph_node.node.ray_cast_hit.loc[2] + minimal_ramp_height)

                current_graph_node.vertex.co[2] = actual_z


                for i, built_vertex in enumerate(edge.vertices, 1):
                    distance_from_end = (built_vertex.co.xy-current_graph_node.vertex.co.xy).length
                    if previous_graph_node.vertex is None:
                        z_shift = 0
                    else:
                        z_shift = (len(edge.vertices)-i)/(len(edge.vertices)+1) * (current_graph_node.vertex.co[2] - previous_graph_node.vertex.co[2])
                    
                    built_vertex.co[2] = current_graph_node.vertex.co[2] - z_shift

    def _build_link(self, path: list[HighwayGraphNode], path_length: float, graph: HighwayGraph, built: set[HighwayGraphNode], bm:BMesh, ray_caster: DropToGround, build_parameters:dict=None):
        build_parameters = build_parameters or {}

        # Find the height of the first and last node to actually hit the ground
        starting_height = next((p.vertex.co[2] if p.vertex else p.node.ray_cast_hit.loc[2] for p in path if p.node.ray_cast_hit.hit), 0)
        ending_height = next((p.vertex.co[2] if p.vertex else p.node.ray_cast_hit.loc[2] for p in reversed(path) if p.node.ray_cast_hit.hit), 0)

        slope = (ending_height-starting_height)/path_length
        total_traveled_distance = 0

        for previous_graph_node, current_graph_node in zip(path[:-1],path[1:]):
            # This part is done at the beginning to make sure the traveled path is kept up to date even if the node
            #falls outside the area
            traveled_distance = previous_graph_node.distance2D(current_graph_node)

            total_traveled_distance += traveled_distance

            if not current_graph_node.node.ray_cast_hit.hit:
                current_graph_node.vertex = None
                current_graph_node.is_built = True
                built.add(current_graph_node)
                continue

            edge = graph.get_edge(previous_graph_node, current_graph_node)
                
            current_graph_node.build(bm)
            built.add(current_graph_node)
            
            if current_graph_node.vertex is None:
                continue

            
            current_graph_node.vertex.co[2] = starting_height + slope*total_traveled_distance 

            built_vertices = self._build_highway_graph_edge(start= previous_graph_node, end=current_graph_node, edge = None, bm=bm, ray_caster=ray_caster, build_parameters=build_parameters)
            edge.vertices = built_vertices

            #Align the points along the previous and current vertex
            for i, built_vertex in enumerate(edge.vertices, 1):
                if previous_graph_node.vertex is None:
                    height_shift = 0
                else:
                    
                    height_shift = (len(built_vertices)-i)/(len(built_vertices)+1) * (current_graph_node.vertex.co[2] - previous_graph_node.vertex.co[2])
                    
                built_vertex.co[2] = current_graph_node.vertex.co[2] - height_shift


    def _build_highway_graph_edge(self, start: HighwayGraphNode, end:HighwayGraphNode, edge: HighwayGraphEdge, bm:BMesh, ray_caster: DropToGround, build_parameters:dict=None)->list[BMVert]:
        print(f'building edge between {start.id} and {end.id}')
        build_parameters = build_parameters or {}
        intermediate_points = self.subdivide_way([start.node.ray_cast_hit.loc,end.node.ray_cast_hit.loc], build_parameters.get('highway_subdivision_size', None))[1:-1]
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
            

    def _generate_graph(self, bm: bmesh, highways:list())->HighwayGraph:
        graph = HighwayGraph(bm,self.highway_type)
        
        for highway in highways:
            for (v1,v2) in zip(highway.nodes[:-1],highway.nodes[1:]):
                graph.add_edge(v1=v1, v2=v2, edge_tags=highway._tags)
        return graph
        
    def get_subdivision_params(self, preceding_point: tuple[float,float], current_point:tuple[float,float], subdivision_size: Number)->tuple[int, Vector]:
        vec = Vector(current_point) - Vector(preceding_point)
        number_steps = max(math.ceil(vec.length/subdivision_size),1)
        return number_steps, vec/number_steps
    
    
class OSMHighWaySubType():
    """
    A restricted access major divided highway, normally with 2 or more running lanes plus emergency hard shoulder. Equivalent to the Freeway, Autobahn, etc..
    """
    _osm_highway_type: str = ''
    detail_level: ClassVar[int] = 3

    @classmethod
    def is_valid_xml(cls, xml_element:Element) -> bool:
        return super(OSMHighway, cls).is_valid_xml(xml_element) and any(
                c.attrib['k'] == cls._osm_sub_name and c.attrib['v'] == cls._osm_highway_type for c in xml_element.iter('tag'))

    @classmethod
    def is_valid_json(cls, json_element:dict) -> bool:
        return super(OSMHighway, cls).is_valid_json(json_element) and json_element.get('tags',{}).get(cls._osm_sub_name,None)==cls._osm_highway_type


class OSMMotorwayLink(OSMHighWaySubType):
    """
    A restricted access major divided highway, normally with 2 or more running lanes plus emergency hard shoulder. Equivalent to the Freeway, Autobahn, etc..
    """
    _osm_highway_type: str = 'motorway_link'

class OSMMotorway(OSMHighWaySubType):
    """
    A restricted access major divided highway, normally with 2 or more running lanes plus emergency hard shoulder. Equivalent to the Freeway, Autobahn, etc..
    """
    _osm_highway_type: str = 'motorway'
    detail_level: ClassVar[int] = 3

class OSMTrunk(OSMHighWaySubType):
    """
    The most important roads in a country's system that aren't motorways. (Need not necessarily be a divided highway.)
    """
    _osm_highway_type: str = 'trunk'
    detail_level: ClassVar[int] = 3

# class OSMTrunkLink(OSMHighWaySubType):
#     """
#     The most important roads in a country's system that aren't motorways. (Need not necessarily be a divided highway.)
#     """
#     _osm_highway_type: str = 'trunk_link'
#     detail_level: ClassVar[int] = 3

class OSMPrimary(OSMHighWaySubType):
    """
    The next most important roads in a country's system. (Often link larger towns.)	
    """
    _osm_highway_type: str = 'primary'
    detail_level: ClassVar[int] = 3

# class OSMPrimaryLink(OSMHighWaySubType):
#     """
#     The next most important roads in a country's system. (Often link larger towns.)	
#     """
#     _osm_highway_type: str = 'primary_link'
#     detail_level: ClassVar[int] = 3

class OSMSecondary(OSMHighWaySubType):
    """
    The next most important roads in a country's system. (Often link towns.)		
    """
    _osm_highway_type: str = 'secondary'
    detail_level: ClassVar[int] = 3

# class OSMSecondaryLink(OSMHighWaySubType):
#     """
#     The next most important roads in a country's system. (Often link towns.)		
#     """
#     _osm_highway_type: str = 'secondary_link'
#     detail_level: ClassVar[int] = 3

class OSMTertiary(OSMHighWaySubType):
    """
    The next most important roads in a country's system. (Often link smaller towns and villages)			
    """
    _osm_highway_type: str = 'tertiary'
    detail_level: ClassVar[int] = 3

# class OSMTertiaryLink(OSMHighWaySubType):
#     """
#     The next most important roads in a country's system. (Often link smaller towns and villages)			
#     """
#     _osm_highway_type: str = 'tertiary_link'
#     detail_level: ClassVar[int] = 3

class OSMUnclassified(OSMHighWaySubType):
    """
    The least important through roads in a country's system – i.e. minor roads of a lower classification than tertiary, but which serve a purpose other than access to properties. (Often link villages and hamlets.)
The word 'unclassified' is a historical artefact of the UK road system and does not mean that the classification is unknown; you can use highway=road for that.			
    """
    _osm_highway_type: str = 'unclassified'
    detail_level: ClassVar[int] = 3

class OSMSteps(OSMHighWaySubType):
    """
    The least important through roads in a country's system – i.e. minor roads of a lower classification than tertiary, but which serve a purpose other than access to properties. (Often link villages and hamlets.)
The word 'unclassified' is a historical artefact of the UK road system and does not mean that the classification is unknown; you can use highway=road for that.			
    """
    _osm_highway_type: str = 'steps'
    detail_level: ClassVar[int] = 3

class OSMPedestrian(OSMHighWaySubType):
    """
    The least important through roads in a country's system – i.e. minor roads of a lower classification than tertiary, but which serve a purpose other than access to properties. (Often link villages and hamlets.)
The word 'unclassified' is a historical artefact of the UK road system and does not mean that the classification is unknown; you can use highway=road for that.			
    """
    _osm_highway_type: str = 'pedestrian'
    detail_level: ClassVar[int] = 3

class OSMFootway(OSMHighWaySubType):
    """
    The least important through roads in a country's system – i.e. minor roads of a lower classification than tertiary, but which serve a purpose other than access to properties. (Often link villages and hamlets.)
The word 'unclassified' is a historical artefact of the UK road system and does not mean that the classification is unknown; you can use highway=road for that.			
    """
    _osm_highway_type: str = 'footway'
    detail_level: ClassVar[int] = 3

class OSMResidential(OSMHighWaySubType):
    """
    The least important through roads in a country's system – i.e. minor roads of a lower classification than tertiary, but which serve a purpose other than access to properties. (Often link villages and hamlets.)
The word 'unclassified' is a historical artefact of the UK road system and does not mean that the classification is unknown; you can use highway=road for that.			
    """
    _osm_highway_type: str = 'residential'
    detail_level: ClassVar[int] = 3


class HighwayGraphNode():
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
    
    def distance(self, other: HighwayGraphNode)->float:
        return ((self.x-other.x)**2+(self.y-other.y)**2+(self.z-other.z)**2)**0.5

    def distance2D(self, other: HighwayGraphNode)->float:
        return ((self.x-other.x)**2+(self.y-other.y)**2)**0.5
    
    def build(self, bm: BMesh)->BVert:
        self.is_built = True
        
        if self.vertex is None and self.node.ray_cast_hit.hit:
            self.vertex = bm.verts.new(self.node.ray_cast_hit.loc)
            self.vertex[bm.verts.layers.string.get("node")] = str.encode(str(self.node._id))
        return self.vertex

class HighwayGraphEdge():
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
        highway_type = self.tags.get('highway',None) or self.tags.get('railway',None)
        self.types.add(highway_type)
        if self.tags.get('bridge',None) is not None:
            self.types.add('bridge')
        if self.tags.get('tunnel',None) is not None:
            self.types.add('tunnel')


class HighwayGraph():
    bm: BMesh = None
    highway_type:str
    highway_base_type:str
    adj_list:dict[HighwayGraphNode, list[tuple[HighwayGraphNode,HighwayGraphEdge]]]

    def __getitem__(self, key):
        return next((n for n in self.adj_list.keys() if str(n.id) == key), None)
    
    def __init__(self, bm: BMesh, highway_type:str) -> None:
        self.bm = bm
        self.highway_type = highway_type
        self.highway_type = highway_type.removesuffix('_link')
        self.adj_list = {}

    def add_edge(self, v1:OSMNode, v2:OSMNode, edge_tags:dict):
        
        node_1 = next((n for n in self.adj_list.keys() if n.id==v1._id), None) or HighwayGraphNode(v1)
        node_2 = next((n for n in self.adj_list.keys() if n.id==v2._id), None) or HighwayGraphNode(v2)
        edge = HighwayGraphEdge(self._node_distance(node_1,node_2), edge_tags)
        if not any(n == node_2 for n,_ in self.adj_list.get(node_1,[])):
            self.adj_list.setdefault(node_1,[]).append((node_2,edge))
        if not any(n == node_1 for n,_ in self.adj_list.get(node_2,[])):
            self.adj_list.setdefault(node_2,[]).append((node_1,edge))

    def get_edge(self, v1:OSMNode, v2:OSMNode)->HighwayGraphEdge:
        return next((e for n,e in self.adj_list.get(v1) if n==v2), None)
    
    def _node_distance(self, v1:HighwayGraphNode, v2:HighwayGraphNode)->float:
        return math.sqrt((v1.x-v2.x)**2 + (v1.y-v2.y)**2)
    
    def get_node_by_id(self, id:str)->HighwayGraphNode:
        return next((n for n in self.adj_list.values() if n.id==id),None)
    
    def get_valid_neighbors(self, node:HighwayGraphNode, valid_edges:list[str] =[])->list[tuple[HighwayGraphNode,HighwayGraphEdge]]:
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
            for (neighbor,edge) in self.adj_list[current_node]:
                # print(f'Looking at neighbor {neighbor.id} which has and edge of type {";".join(edge.types)}')
                if edge.is_continuation(valid_edges):
                    if str(neighbor.id) not in current_path:
                        next_path = current_path+[str(neighbor.id)]
                        queue.append(next_path)
                        distances[','.join(next_path)] = distances[','.join(current_path)]+edge.length

        all_paths = sorted(list(distances.keys()))
        for i,path in enumerate(all_paths):
            if any(p.startswith(path) for p in all_paths[i+1:]):
                distances.pop(path)
        return distances



    def get_unbuilt_nodes(self, highway_types:list = []):
        for node,edges in sorted(self.adj_list.items(), key=lambda x: len(x[1])):
            if any(not n.is_built and e.is_continuation(highway_types) for n,e in edges): #not node.is_built and 
                yield node
        yield None