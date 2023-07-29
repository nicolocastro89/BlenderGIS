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
    _highways_roads:ClassVar[list[str]] = ["motorway", 'truck', 'primary', 'secondary', 'tertiary', 'unclassified', 'residential', 'service', 'pedestrian', 'track', 'other']
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

    def get_ray_cast_hit_points(self, geoscn, reproject, ray_caster: DropToGround = None, subdivision_size:Number=None)->list[RayCastHit]:

        dx, dy = geoscn.crsx, geoscn.crsy

        delta_xy = Vector((dx,dy))
        pts = [(float(node._lon), float(node._lat)) for node in self.nodes]
        pts = [{'point':Vector(xy)-delta_xy, 'nodes':[self.nodes[idx]]} for idx, xy in enumerate(reproject.pts(pts))]
        if subdivision_size:
            subdivided=[]
            subdivided.append(pts[0])
            for first, second in zip(pts, pts[1:]):
                number_steps, vector = self.getSubdivisionParams(first['point'], second['point'], subdivision_size)
                subdivided.extend(({'point': first['point']+step*vector, 'nodes':[]} for step in range(1,number_steps)))
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
                zs = [p.loc.z for p in rc_hits if p.hit]
                meanZ = sum(zs) / len(zs)
                for v in rc_hits:
                    if not v.hit:
                        v.loc.z = meanZ
        else:
            rc_hits = [ RayCastHit(loc=(v['point'].x, v['point'].y, 0), hit = True, originates_from= v['nodes']) for v in pts]

        return rc_hits
    
   
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
            
    def get_highway_continuation(self, include_links=False)->HighwayContinuation:
        first, last = self.end_points()
        
        #Get all highway points that share the first endpoint
        first_referenced = self._library.get_elements_by_ids(list(first.get_referenced_from(type(self))))

        #only take highways that are not self, are of the same type and have the endpoint in an endpoint position
        filter_highway_type = lambda h: h.highway_type == self.highway_type if not include_links else h.highway_base_type == self.highway_base_type
        filter_highways = lambda h,endpoint: h._id!=self._id and filter_highway_type(h) and endpoint in h.end_points()
        first_referenced = [h for h in first_referenced if filter_highways(h, first)]
        #first_referenced = first_referenced[0] if len(first_referenced)==1 else None

        last_referenced = self._library.get_elements_by_ids(list(last.get_referenced_from(type(self))))
        last_referenced = [h for h in last_referenced if filter_highways(h, last)]
        #last_referenced = last_referenced[0] if len(last_referenced)==1 else None

        return HighwayContinuation(first_referenced, last_referenced)

    @classmethod
    def preprocess(cls, library: OSMLibrary, ray_caster:DropToGround):
        highways = library.get_elements(cls).values()
        for part in highways:
            part.preprocess_instance(geoscn = library.geo_scene, ray_caster=ray_caster)

    def preprocess_instance(self, geoscn, ray_caster:DropToGround):#, bridges: tuple['BVHTree',list[int]]
        """Preprocess the highway by doing the following in order:
        - Adding a reference to the highway in all nodes referenced
        - add references to before and after highways"""
        #from .man_made import OSMBridge
        self._is_preprocessed = False
        super(OSMHighway,self).preprocess_instance(geoscn, ray_caster)
        if 'bridge' in self._tags:
            print(f'Looking for bridges for {self._id}')
            self.find_assigned_bridges()
        
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
            #part._build_highway_in_mesh(bm, geoscn=geoscn, reproject=reproject, ray_caster=ray_caster, build_parameters=build_parameters)

        bridges = library.get_elements_by_ids(part._referenced_by.get(OSMBridge, []))
        if any(not b.is_built for b in bridges):
            bridge_nodes = []
            bridge_vertices = []
            for p in (h for h in highways if h.tags.get('bridge',False)):
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

        # for part in highways:
        #     bm = meshes.get(part.highway_type)
        #     if bm:
        #         bm.verts.ensure_lookup_table()
        #         bridges = library.get_elements_by_ids(part._referenced_by.get(OSMBridge, []))
        #         if any(not b.is_built for b in bridges):
        #             size = len(bm.verts)
        #             kd = KDTree(size)

        #             for i, v in enumerate(bm.verts):
        #                 kd.insert(Vector((v.co[0], v.co[1], 0)), i)
        #             kd.balance()

        #             for bridge in [b for b in bridges if not b.is_built]:
        #                 bridge_bm = bmesh.new()
        #                 bridge_bm = bridge._build_instance(
        #                                         bridge_bm, 
        #                                         geoscn=geoscn, 
        #                                         reproject=reproject, 
        #                                         ray_caster=ray_caster, 
        #                                         supported_highway_nodes = bm.verts, 
        #                                         kd_tree = kd,
        #                                         build_parameters=build_parameters)
                       
        #                 bmesh.ops.remove_doubles(bridge_bm, verts=bridge_bm.verts, dist=0.0001)
        #                 bridge_object = bpy.data.objects.new(str(bridge._id),bpy.data.meshes.new(str(bridge._id)))
        #                 geoscn.scn.collection.objects.link(bridge_object)
        #                 bridge._is_built = True

        #                 bridge_bm.to_mesh(bridge_object.data)
        #                 bridge_bm.free()
        
        
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


    def _build_highway_with_graph(self, bm, geoscn, reproject, ray_caster:DropToGround = None, build_parameters:dict={})->bmesh:
        #Get all highways pieces which are either of the same type of links
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

        for starting_node in takewhile(lambda x: x is not None , graph.get_unbuilt_nodes([highway_base_type])):
            self._breadth_first_build_graph(bm, ray_caster=ray_caster, graph=graph, start = starting_node, built = set(), valid_edges = [highway_base_type], build_parameters=build_parameters)
        
        for starting_node in takewhile(lambda x: x is not None , graph.get_unbuilt_nodes([highway_base_type+'_link'])):
            self._breadth_first_build_graph(bm, ray_caster=ray_caster, graph=graph, start = starting_node, built = set(), valid_edges = [highway_base_type+'_link'], build_parameters=build_parameters)
        

        # bridges = set([bridge for part in built_highways for bridge in self._library.get_elements_by_ids(part._referenced_by.get(OSMBridge, []))])
        # if any(not b.is_built for b in bridges):
        #     # bm.verts.ensure_lookup_table()
        #     bridge_vertices = [node.vertex for node,neighbors in graph.adj_list.items() if node.vertex is not None and any(edge.is_continuation(required_tags=['bridge']) for _,edge in neighbors)]
        #     size = len(bridge_vertices)
        #     kd = KDTree(size)
        #     for i, n in enumerate(bridge_vertices):
        #         kd.insert(Vector((n.co[0], n.co[1], 0)), i)
        #     kd.balance()

        #     for bridge in (b for b in bridges if not b.is_built):
        #         bridge_bm = bmesh.new()
        #         bridge_bm = bridge._build_instance(
        #                                 bridge_bm, 
        #                                 geoscn=geoscn, 
        #                                 reproject=reproject, 
        #                                 ray_caster=ray_caster, 
        #                                 supported_highway_nodes = bridge_vertices, 
        #                                 kd_tree = kd,
        #                                 build_parameters=build_parameters)
                
        #         bmesh.ops.remove_doubles(bridge_bm, verts=bridge_bm.verts, dist=0.0001)
        #         bridge_object = bpy.data.objects.new(str(bridge._id),bpy.data.meshes.new(str(bridge._id)))
        #         geoscn.scn.collection.objects.link(bridge_object)
        #         bridge._is_built = True

        #         bridge_bm.to_mesh(bridge_object.data)
        #         bridge_bm.free()
        
        
        # starting_node = next((n1 for n1,adj in graph.adj_list.items() if 
        #                       any(e.is_continuation([highway_base_type]) for _,e in adj)), None)
        
        # if starting_node is not None:
            
        
        # # build links
        # for starting_link in [node for (node, adj) in graph.adj_list.items() if 
        #                       (any(e.is_continuation([highway_base_type])) and any(e.is_continuation([highway_base_type+'_link'])) for _,e in adj)]:
        #     self._breadth_first_build_graph(bm, ray_caster=ray_caster, graph=graph, start = starting_link, built = set(), valid_edges = [highway_base_type+'_link'], build_parameters=build_parameters)


        for highway in built_highways:
            highway.is_built = True

    def _breadth_first_build_graph(self, bm:BMesh, ray_caster: DropToGround, graph: HighwayGraph, start:HighwayGraphNode, built:set = set(), valid_edges:list(str)=[], build_parameters:dict={}):

        print(f'Setting off breadth first building, starting node is {start.node}')

        queue = deque()
        start.is_built = True
        if start.node.ray_cast_hit.hit:
            start.vertex = bm.verts.new(start.node.ray_cast_hit.loc)
            start.vertex[bm.verts.layers.string.get("node")] = str.encode(str(start.node._id))
        
        built.add(start)
        queue.append(start)
        
        while queue:
             
            current_node = queue.popleft()
            for (neighbor, edge) in ( (n,e) for (n,e) in graph.get_valid_neighbors(current_node, valid_edges) if n not in built):
                # print(f'Looking at {current_node.id} neighbor {neighbor.id} linked by edge {";".join(edge.types)}')
                l=next((l for l in self._highways_to_level if l in edge.types), None)
                if l is not None:
                    neighbors = self._leveled_depth_first_build_graph(bm = bm,
                                                                      ray_caster=ray_caster,
                                                                      graph = graph,
                                                                      start_node=current_node,
                                                                      built = built,
                                                                      valid_edges=valid_edges+[l],
                                                                      build_parameters=build_parameters)
                    queue.extend(neighbors)
                else:
                    neighbor.is_built = True
                    if neighbor.node.ray_cast_hit.hit:
                      neighbor.vertex = bm.verts.new(neighbor.node.ray_cast_hit.loc)
                      neighbor.vertex[bm.verts.layers.string.get("node")] = str.encode(str(neighbor.node._id))
                    
                    # else:
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
        # split the path and cast each id to int
        path_distances = [([int(id) for id in p.split(',')], d ) for p,d in sorted(bfs_explore_result.items(), key=itemgetter(1), reverse=True)]
        print(f'Found {len(path_distances)} paths')
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
                    
                    overlap_length = sum(built_path[i].distance2D(built_path[i-1]) for i in range(1,overlap_truth_index+1))
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
            
            if not actual_path[0].vertex and actual_path[0].node.ray_cast_hit.hit:
                actual_path[0].vertex = bm.verts.new(actual_path[0].node.ray_cast_hit.loc)
            self._build_leveled_path(path=actual_path,
                                    path_length=longest_to_build,
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
    
    def _build_leveled_path(self, path: list[HighwayGraphNode], path_length: float, built: set[HighwayGraphNode], bm:BMesh, ray_caster: DropToGround, build_parameters:dict=None):
        #print('\t'+f'building path starting at {path[0].id} and ending at {path[-1].id}')
        build_parameters = build_parameters or {}
        max_delta = 0
        for p1,p2 in zip(path[:-1], path[1:]):
            traveled_distance = ((p1.node.ray_cast_hit.loc[0]-p2.node.ray_cast_hit.loc[0])**2 + 
                                  (p1.node.ray_cast_hit.loc[1]-p2.node.ray_cast_hit.loc[1])**2)**0.5
            vertical_distance = p1.node.ray_cast_hit.loc[2]-p2.node.ray_cast_hit.loc[2]
            max_delta = max(max_delta, abs(vertical_distance/traveled_distance))

        starting_height = next((p.node.ray_cast_hit.loc[2] for p in path if p.node.ray_cast_hit.hit), 0)
        ending_height = next((p.node.ray_cast_hit.loc[2] for p in reversed(path) if p.node.ray_cast_hit.hit), 0)

        if path[0].node.ray_cast_hit.hit and not path[-1].node.ray_cast_hit.hit:
            ending_height = starting_height
        if path[-1].node.ray_cast_hit.hit and not path[0].node.ray_cast_hit.hit:
            starting_height = ending_height

        delta_z = (ending_height-starting_height)/path_length
        total_traveled_distance = 0

        layer_height = self.layer * build_parameters.get('layer_default_height',10)
        layer_access_ramp_length = build_parameters.get('layer_acccess_ramp_length',50.)
        for previous_graph_node, current_graph_node in zip(path[:-1],path[1:]):
            current_graph_node.is_built=True
            built.add(current_graph_node)

            traveled_distance = ((previous_graph_node.node.ray_cast_hit.loc[0]-current_graph_node.node.ray_cast_hit.loc[0])**2 + 
                                  (previous_graph_node.node.ray_cast_hit.loc[1]-current_graph_node.node.ray_cast_hit.loc[1])**2)**0.5
            vertical_distance = previous_graph_node.node.ray_cast_hit.loc[2]-current_graph_node.node.ray_cast_hit.loc[2]

            total_traveled_distance += traveled_distance
            
            distance_from_end = min([abs(traveled_distance-path_length),path_length-traveled_distance, layer_access_ramp_length])
            ramp_ratio = distance_from_end/layer_access_ramp_length

            
            if max_delta>0.05:
                actual_z = starting_height + delta_z*total_traveled_distance
            else:
                minimal_ramp_height = ramp_ratio*layer_height
                actual_z = starting_height + max(delta_z*total_traveled_distance, minimal_ramp_height)
            
            if actual_z > 300:
                print('TOO HIGH')
                newline = "\n\t"
                print(f'Positioning point {current_graph_node.id} at z {actual_z}. Full path{newline}{newline.join(n.id for n in path)}')
            current_graph_node.node.ray_cast_hit.loc[2] = actual_z # + ramp_ratio*layer_height

            if not current_graph_node.node.ray_cast_hit.hit:
                current_graph_node.vertex = None
                continue

            current_graph_node.vertex = bm.verts.new(current_graph_node.node.ray_cast_hit.loc)
            current_graph_node.vertex[bm.verts.layers.string.get("node")] = str.encode(str(current_graph_node.node._id))

            built_vertices = self._build_highway_graph_edge(start= previous_graph_node, end=current_graph_node, edge = None, bm=bm, ray_caster=ray_caster, build_parameters=build_parameters)
            
            for i, built_vertex in enumerate(built_vertices, 1):
                distance_from_end = ((built_vertex.co[0]-current_graph_node.vertex.co[0])**2 + 
                                  (built_vertex.co[1]-current_graph_node.vertex.co[1])**2)**0.5
                if previous_graph_node.vertex is None:
                    z = current_graph_node.vertex.co[2]
                else:
                    z = i/(len(built_vertices)+1) *(current_graph_node.vertex.co[2] - previous_graph_node.vertex.co[2])
                    # print(f'Bridging between {current_graph_node.vertex.co[2]}-{previous_graph_node.vertex.co[2]} ration {i/(len(built_vertices)+1)} height {z} setting it to {starting_height + z}')
                #distance_from_end/traveled_distance * (current_graph_node.vertex.co[2] - previous_graph_node.vertex.co[2])
                #print(f'distance from end {distance_from_end} end nodes have height: {previous_graph_node.vertex.co[2]}-{current_graph_node.vertex.co[2]} ratio {distance_from_end/traveled_distance} height {z} from {starting_height}')
                built_vertex.co[2] = current_graph_node.vertex.co[2] - z
            

    def _build_highway_graph_edge(self, start: HighwayGraphNode, end:HighwayGraphNode, edge: HighwayGraphEdge, bm:BMesh, ray_caster: DropToGround, build_parameters:dict=None)->list[BMVert]:
        # print(f'building edge between {start.id} and {end.id}')
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
        

    def _build_highway_in_mesh(self, bm, geoscn, reproject, ray_caster:DropToGround = None, build_parameters:dict={})->bmesh:
        from .man_made import OSMBridge
        leveled_type = next((t for t in self._highways_to_level if t in self._tags), None)
        built_highways = [self]
        if leveled_type:
            continuation = self.get_highway_continuation()
            if len(continuation.after)>0 and leveled_type in continuation.after[0]._tags:
                built_highways.append(continuation.after[0])
                built_highways.extend(continuation.after[0].get_all_highway_parts(self, possible_subtypes = [leveled_type]))
            if len(continuation.before)>0 and leveled_type in continuation.before[0]._tags:
                built_highways.insert(0,continuation.before[0])
                built_highways[:0] = continuation.before[0].get_all_highway_parts(self, possible_subtypes = [leveled_type])
            points = self.combine_highway_points(built_highways, geoscn=geoscn, reproject=reproject, ray_caster = ray_caster, subdivision_size=None)
            self.level_points(points,build_parameters)
        else:
            points = self.get_ray_cast_hit_points(geoscn=geoscn, reproject=reproject, ray_caster=ray_caster, subdivision_size=build_parameters.get('highway_subdivision_size', None))
        
        hit_points = groupby(points, key= lambda p: p.hit)
        grouped_vertices = []
        for k,highway_points in hit_points:
            if k:
                grouped_vertices.append([bm.verts.new(pt.loc) for pt in highway_points])
                # points = self.combine_highway_points(list(highway_part), geoscn, reproject, ray_caster, build_parameters)

        for vertices in grouped_vertices:
            for v in zip(vertices[:-1], vertices[1:]):
                bm.edges.new(v)

        for highway_part in built_highways:
            highway_part._is_built = True

        # for highway_part in built_highways:
        #     bridges = self._library.get_elements_by_ids(highway_part._referenced_by.get(OSMBridge, []))

        #     if any(not b.is_built for b in bridges):
        #         all_vertices = [v for vertex_group in grouped_vertices for v in vertex_group]
        #         size = len(points)
        #         kd = KDTree(size)

        #         for i, v in enumerate(all_vertices):
        #             kd.insert(Vector((v.co[0], v.co[1], 0)), i)
        #         kd.balance()

        #         for bridge in [b for b in bridges if not b.is_built]:
        #             bridge_bm = bmesh.new()
        #             bridge_bm = bridge._build_instance(
        #                                     bridge_bm, 
        #                                     geoscn=geoscn, 
        #                                     reproject=reproject, 
        #                                     ray_caster=ray_caster, 
        #                                     supported_highway_nodes = all_vertices, 
        #                                     kd_tree = kd,
        #                                     build_parameters=build_parameters)
                    
        #             bmesh.ops.remove_doubles(bridge_bm, verts=bridge_bm.verts, dist=0.0001)
        #             bridge_object = bpy.data.objects.new(str(bridge._id),bpy.data.meshes.new(str(bridge._id)))
        #             geoscn.scn.collection.objects.link(bridge_object)
        #             bridge._is_built = True

        #             bridge_bm.to_mesh(bridge_object.data)
        #             bridge_bm.free()
        return bm     

    def combine_highway_points(self, highway_parts: list[OSMHighway], geoscn, reproject, ray_caster:DropToGround = None, subdivision_size:Number=None):
        points = highway_parts[0].get_ray_cast_hit_points(geoscn=geoscn, reproject=reproject, ray_caster=ray_caster, subdivision_size=subdivision_size)
        for idx in range(1,len(highway_parts)):
            current_part = highway_parts[idx]
            previous_part = highway_parts[idx-1]

            part_points = current_part.get_ray_cast_hit_points(geoscn=geoscn, reproject=reproject, ray_caster=ray_caster, subdivision_size = subdivision_size)
            if current_part.nodes[-1] == previous_part.nodes[-1]:
                part_points.reverse()
            if current_part.nodes[0] == previous_part.nodes[0]:
                points.reverse()

            points.extend(part_points[1:])
        return points
       
    def shift_spline_by(self, spline, by:int):
        spline.points.add(by)
        for i in reversed(range(len(spline.points))):
            if i>=by:
                spline.points[i].co = [*spline.points[i-by].co]
            else:
                spline.points[i].co = (0,0,0,1.0)

    def level_points(self, points: list[RayCastHit], build_parameters:dict={}):
        
        full_length = sum(((p1.loc[0]-p2.loc[0])**2 + (p1.loc[1]-p2.loc[1])**2)**0.5 for p1, p2 in zip(points[:-1], points[1:]))
        
        starting_height = points[0].loc[2] if points[0].hit else None
        ending_height = points[-1].loc[2] if points[-1].hit else None

        delta_z=0
        if starting_height is None and ending_height is None:
            first_initial_node = next((p for p in points if p.hit), None)
            starting_height = first_initial_node.loc[2] if first_initial_node else 0

            first_ending_node = next((p for p in points[::-1] if p.hit), None)
            ending_height = first_ending_node.loc[2] if first_ending_node else 0
            starting_height = ending_height = max(starting_height, ending_height)

        if starting_height is not None and ending_height is None:
            ending_height = starting_height
        elif starting_height is None and ending_height is not None:
            starting_height = ending_height
            
        delta_z = (ending_height-starting_height)/full_length
        full_distance = 0
        
        layer_height = self.layer * build_parameters.get('layer_default_height',10)
        layer_access_ramp_length = build_parameters.get('layer_acccess_ramp_length',50.)
        for i in range(1,len(points)):
            full_distance += ((points[i-1].loc[0]-points[i].loc[0])**2 + (points[i-1].loc[1]-points[i].loc[1])**2)**0.5
            distance_from_end = min([abs(full_distance-full_length),full_length-full_distance, layer_access_ramp_length])
            ramp_ratio = distance_from_end/layer_access_ramp_length
            points[i].loc[2] = starting_height + delta_z*full_distance + ramp_ratio*layer_height

    def get_all_highway_parts(self, previous: OSMHighway, stop_at_type_change = True, possible_subtypes = [])->list[OSMHighway]:
        print(f'Getting continuation for {self}')
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
        if len(continuation.after)>0 and filter_algorithm(previous, continuation.after[0]):
            next = continuation.after[0]
        elif len(continuation.before)>0 and filter_algorithm(previous, continuation.before[0]):
            next = continuation.before[0]
        
        if next is not None:
            all_parts.append(next)
            all_parts.extend(next.get_all_highway_parts(self, stop_at_type_change, possible_subtypes))

        return all_parts
        

    def isPointOnTerrain(self, point, ray_caster:DropToGround = None) -> bool:
        return ray_caster(point, -zAxis)[0] != None

    def getSubdivisionParams(self, preceding_point: tuple[float,float], current_point:tuple[float,float], subdivision_size: Number)->tuple[int, Vector]:
        vec = Vector(current_point) - Vector(preceding_point)
        number_steps = max(math.ceil(vec.length/subdivision_size),1)
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

class OSMTrunkLink(OSMHighWaySubType):
    """
    The most important roads in a country's system that aren't motorways. (Need not necessarily be a divided highway.)
    """
    _osm_highway_type: str = 'trunk_link'
    detail_level: ClassVar[int] = 3

class OSMPrimary(OSMHighWaySubType):
    """
    The next most important roads in a country's system. (Often link larger towns.)	
    """
    _osm_highway_type: str = 'primary'
    detail_level: ClassVar[int] = 3

class OSMPrimaryLink(OSMHighWaySubType):
    """
    The next most important roads in a country's system. (Often link larger towns.)	
    """
    _osm_highway_type: str = 'primary_link'
    detail_level: ClassVar[int] = 3

class OSMSecondary(OSMHighWaySubType):
    """
    The next most important roads in a country's system. (Often link towns.)		
    """
    _osm_highway_type: str = 'secondary'
    detail_level: ClassVar[int] = 3

class OSMSecondaryLink(OSMHighWaySubType):
    """
    The next most important roads in a country's system. (Often link towns.)		
    """
    _osm_highway_type: str = 'secondary_link'
    detail_level: ClassVar[int] = 3

class OSMTertiary(OSMHighWaySubType):
    """
    The next most important roads in a country's system. (Often link smaller towns and villages)			
    """
    _osm_highway_type: str = 'tertiary'
    detail_level: ClassVar[int] = 3

class OSMTertiaryLink(OSMHighWaySubType):
    """
    The next most important roads in a country's system. (Often link smaller towns and villages)			
    """
    _osm_highway_type: str = 'tertiary_link'
    detail_level: ClassVar[int] = 3

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

class HighwayGraphEdge():
    length:float
    tags:dict
    types=set()
    is_built = False

    edges:list[BMEdge] =[]
    vertices:list[BMVert] =[]

    def add_intermediate_vertex(self, vertex: BMVert):
        self.vertices.append(vertex)

    def add_intermediate_edge(self, edge: BMEdge):
        self.edges.append(edge)

    
    
    def is_continuation(self, required_tags:list)->bool:
        return all(t in self.types for t in required_tags)

    def __init__(self, length, tags) -> None:
        self.length = length
        self.tags = tags
        self.types = set()
        highway_type = self.tags.get('highway',None)
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

    def _node_distance(self, v1:HighwayGraphNode, v2:HighwayGraphNode)->float:
        return math.sqrt((v1.x-v2.x)**2 + (v1.y-v2.y)**2)
    
    def get_node_by_id(self, id:str)->HighwayGraphNode:
        return next((n for n in self.adj_list.values() if n.id==id),None)
    
    def get_valid_neighbors(self, node:HighwayGraphNode, valid_edges:list[str] =[])->list[tuple[HighwayGraphNode,HighwayGraphEdge]]:
        return [(n,e) for (n,e) in self.adj_list[node] if e.is_continuation(valid_edges)]

    def dfs_explore(self, start, path = [], visited = set(), valid_edges:list[str]=[]):
        path.append(start)
        visited.add(start)
        paths = []
        for (neighbor, edge) in self.adj_list[start]:
            if neighbor not in visited and edge.is_continuation(valid_edges):
                current_path = path+[neighbor]
                paths.append(current_path)
                paths.extend(self.dfs_explore(neighbor, current_path, visited, valid_edges=valid_edges))
        return paths
    
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
            # current_path = ','.join(list(filter(None,current_path.split(',')))+[str(current_node.id)]) 
            # list(filter(None,current_path.split(',')))+[str(current_node.id)]
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

    def  bfs_farthest(self, start_node, valid_edges = [])->tuple[list[int],float]:
        distances = self.bfs_explore(start_node, valid_edges)
        
        max_distance = max(distances.values())
        return next((([int(id) for id in path.split(',')], max_distance) for (path,distance) in distances.items() if distance == max_distance),None)

    def _explore_node(self, current_path, current_node, valid_edges, distances, visited):
        updated_path = f'{current_path},{current_node.id}'
        for (neighbor,edge) in self.adj_list[current_node]:
            if edge.is_continuation(valid_edges):
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[f'{updated_path},{neighbor.id}'] = distances[current_path]+edge.length

    def find_longest_path(self, valid_edges = [])->list[int]:
        starting_node = next((n1 for n1,adj in self.adj_list.items() if 
                              any(e.is_continuation(valid_edges) for _,e in adj)), None)
        
        if starting_node is None:
            return None
        first_path, distance = self.bfs_farthest(starting_node, valid_edges=valid_edges)
        first_node = first_path[-1]
        second_path, distance = self.bfs_farthest(first_node, valid_edges=valid_edges)
        return second_path
    
    def get_unbuilt_nodes(self, highway_types:list = []):
        for node,edges in self.adj_list.items():
            if node.vertex is None and any(not e.is_built and e.is_continuation(highway_types) for _,e in edges):
                yield node

        yield None