from gym.spaces import Discrete
import torch
import copy
import time
import math
from itertools import combinations
import torch.nn as nn
import numpy as np
from gym.utils import seeding
import os
import sys
import logging
import shutil
import gym
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)
import dreamplace.configure as configure
import dreamplace.Params as Params
import dreamplace.PlaceDB as PlaceDB
import dreamplace.NonLinearPlace as NonLinearPlace
import pdb
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors 
from collections import Counter
from rnd import RNDModel
import torch.optim as optim
from EA import *
from multiprocessing import Pool

np.set_printoptions(threshold=np.inf)
rnd = RNDModel((1, 1, 256, 256), 32*32)
forward_mse = nn.MSELoss(reduction='none')
optimizer = optim.Adam(rnd.predictor.parameters(), lr=2e-6)

def compute_intrinsic_reward(rnd, next_obs):
    next_obs = next_obs.clone()
    next_obs = next_obs.unsqueeze(0).unsqueeze(0)
    target_next_feature = rnd.target(next_obs)
    predict_next_feature = rnd.predictor(next_obs)

    forward_loss = forward_mse(predict_next_feature, target_next_feature).mean(-1)
    intrinsic_reward = (target_next_feature - predict_next_feature).pow(2).sum(1) / 2
    optimizer.zero_grad()
    forward_loss.backward()

    return intrinsic_reward.item()*1e-4

def place(params,gpu_id=[0]):
    """
    @brief Top API to run the entire placement flow. 
    @param params parameters 
    """
    assert (not params.gpu) or configure.compile_configurations["CUDA_FOUND"] == 'TRUE', \
            "CANNOT enable GPU without CUDA compiled"
    np.random.seed(params.random_seed)
    # read database
    placedb = PlaceDB.PlaceDB()
    placedb(params,gpu_id)

    timer = None
    # solve placement
    placer = NonLinearPlace.NonLinearPlace(params, placedb,timer)
    metrics = placer(params, placedb)
    result = metrics[-3][0]

    # write placement solution
    path = "%s/%s" % (params.result_dir, params.design_name())
    if not os.path.exists(path):
        os.system("mkdir -p %s" % (path))
    gp_out_file = os.path.join(
        path,
        "%s.gp.%s" % (params.design_name(), params.solution_file_suffix()))
    placedb.write(params, gp_out_file)

    # call external detailed placement
    # TODO: support more external placers, currently only support
    # 1. NTUplace3/NTUplace4h with Bookshelf format
    # 2. NTUplace_4dr with LEF/DEF format
    if params.detailed_place_engine and os.path.exists(
            params.detailed_place_engine):
        logging.info("Use external detailed placement engine %s" %
                     (params.detailed_place_engine))
        if params.solution_file_suffix() == "pl" and any(
                dp_engine in params.detailed_place_engine
                for dp_engine in ['ntuplace3', 'ntuplace4h']):
            dp_out_file = gp_out_file.replace(".gp.pl", "")
            # add target density constraint if provided
            target_density_cmd = ""
            if params.target_density < 1.0 and not params.routability_opt_flag:
                target_density_cmd = " -util %f" % (params.target_density)
            cmd = "%s -aux %s -loadpl %s %s -out %s -noglobal %s" % (
                params.detailed_place_engine, params.aux_input, gp_out_file,
                target_density_cmd, dp_out_file, params.detailed_place_command)
            logging.info("%s" % (cmd))
            # tt = time.time()
            os.system(cmd)
            # logging.info("External detailed placement takes %.2f seconds" %
            #              (time.time() - tt))

            if params.plot_flag:
                # read solution and evaluate
                placedb.read_pl(params, dp_out_file + ".ntup.pl")
                iteration = len(metrics)
                pos = placer.init_pos
                pos[0:placedb.num_physical_nodes] = placedb.node_x
                pos[placedb.num_nodes:placedb.num_nodes +
                    placedb.num_physical_nodes] = placedb.node_y
                hpwl, density_overflow, max_density = placer.validate(
                    placedb, pos, iteration)
                logging.info(
                    "iteration %4d, HPWL %.3E, overflow %.3E, max density %.3E"
                    % (iteration, hpwl, density_overflow, max_density))
                placer.plot(params, placedb, iteration, pos)
        elif 'ntuplace_4dr' in params.detailed_place_engine:
            dp_out_file = gp_out_file.replace(".gp.def", "")
            cmd = "%s" % (params.detailed_place_engine)
            for lef in params.lef_input:
                if "tech.lef" in lef:
                    cmd += " -tech_lef %s" % (lef)
                else:
                    cmd += " -cell_lef %s" % (lef)
            cmd += " -floorplan_def %s" % (gp_out_file)
            cmd += " -verilog %s" % (params.verilog_input)
            cmd += " -out ntuplace_4dr_out"
            cmd += " -placement_constraints %s/placement.constraints" % (
                os.path.dirname(params.verilog_input))
            cmd += " -noglobal %s ; " % (params.detailed_place_command)
            cmd += "mv ntuplace_4dr_out.fence.plt %s.fense.plt ; " % (
                dp_out_file)
            cmd += "mv ntuplace_4dr_out.init.plt %s.init.plt ; " % (
                dp_out_file)
            cmd += "mv ntuplace_4dr_out %s.ntup.def ; " % (dp_out_file)
            cmd += "mv ntuplace_4dr_out.ntup.overflow.plt %s.ntup.overflow.plt ; " % (
                dp_out_file)
            cmd += "mv ntuplace_4dr_out.ntup.plt %s.ntup.plt ; " % (
                dp_out_file)
            if os.path.exists("%s/dat" % (os.path.dirname(dp_out_file))):
                cmd += "rm -r %s/dat ; " % (os.path.dirname(dp_out_file))
            cmd += "mv dat %s/ ; " % (os.path.dirname(dp_out_file))
            # logging.info("%s" % (cmd))
            # tt = time.time()
            os.system(cmd)
            # logging.info("External detailed placement takes %.2f seconds" %
            #              (time.time() - tt))
        else:
            logging.warning(
                "External detailed placement only supports NTUplace3/NTUplace4dr API"
            )
    elif params.detailed_place_engine:
        logging.warning(
            "External detailed placement engine %s or aux file NOT found" %
            (params.detailed_place_engine))

    return result

def write(res,name,idmap,path):
    f = open(path+"/"+name+".pl", "w")
    with open("./data/"+name+".pl", "r") as f2:
        for line in f2:
            line = line.strip()
            l = line.split()
            if line and l[0][0] == 'o':
                inst_name = l[0]
                if inst_name in idmap:
                    index = idmap[inst_name]
                    l[1] = str(int(res[index][0]))
                    l[2] = str(int(res[index][1]))
                    line = '\t'.join(l)
            f.write(line)
            f.write('\n')

def read_node_file(path, benchmark):
    node_info = {}
    node_info_raw_id_name ={}
    node_cnt = 0
    with open(path,'r') as f:
        for line in f.readlines():
            if not line.startswith("\t"):
                continue
            line = line.strip().split()
            if line[-1] != "terminal":
                continue
            node_name = line[0]
            x = int(line[1])
            y = int(line[2])
            node_info[node_name] = {"id": node_cnt, "x": x , "y": y }
            node_info_raw_id_name[node_cnt] = node_name
            node_cnt += 1
    print("len node_info", len(node_info))
    return node_info, node_info_raw_id_name

def read_net_file(path, node_info):
    net_info = {}
    net_name = None
    net_cnt = 0
    with open(path,'r') as f:
        for line in f.readlines():
            if not line.startswith("\t") and not line.startswith("NetDegree"):
                continue
            line = line.strip().split()
            if line[0] == "NetDegree":
                net_name = line[-1]
            else:
                node_name = line[0]
                if node_name in node_info:
                    if not net_name in net_info:
                        net_info[net_name] = {}
                        net_info[net_name]["nodes"] = {}
                        net_info[net_name]["ports"] = {}
                    if not node_name in net_info[net_name]["nodes"]:
                        x_offset = float(line[-2])
                        y_offset = float(line[-1])
                        net_info[net_name]["nodes"][node_name] = {}
                        net_info[net_name]["nodes"][node_name] = {"x_offset": x_offset, "y_offset": y_offset}
    for net_name in list(net_info.keys()):
        if len(net_info[net_name]["nodes"]) <= 1:
            net_info.pop(net_name)
    for net_name in net_info:
        net_info[net_name]['id'] = net_cnt
        net_cnt += 1
    print("adjust net size = {}".format(len(net_info)))
    return net_info

def read_pl_file(path, node_info):
    max_height = 0
    max_width = 0
    with open(path,'r') as f:
        for line in f.readlines():
            if not line.startswith('o'):
                continue
            line = line.strip().split()
            node_name = line[0]
            if not node_name in node_info:
                continue
            place_x = int(line[1])
            place_y = int(line[2])
            max_height = max(max_height, node_info[node_name]["x"] + place_x)
            max_width = max(max_width, node_info[node_name]["y"] + place_y)
            node_info[node_name]["raw_x"] = place_x
            node_info[node_name]["raw_y"] = place_y
    return max(max_height, max_width), max(max_height, max_width)

def get_node_to_net_dict(node_info, net_info):
    node_to_net_dict = {}
    for node_name in node_info:
        node_to_net_dict[node_name] = set()
    for net_name in net_info:
        for node_name in net_info[net_name]["nodes"]:
            node_to_net_dict[node_name].add(net_name)
    for node_name in node_info:
        node_to_net_dict[node_name] = list(node_to_net_dict[node_name])
        node_to_net_dict[node_name].sort()
    return node_to_net_dict

def get_node_id_to_name_topology(node_info, node_to_net_dict, net_info, benchmark):
    node_id_to_name = []
    adjacency = {}
    for net_name in net_info:
        for node_name_1, node_name_2 in list(combinations(net_info[net_name]['nodes'],2)):
            if node_name_1 not in adjacency:
                adjacency[node_name_1] = set()
            if node_name_2 not in adjacency:
                adjacency[node_name_2] = set()
            adjacency[node_name_1].add(node_name_2)
            adjacency[node_name_2].add(node_name_1)

    visited_node = set()

    node_net_num = {}
    for node_name in node_info:
        node_net_num[node_name] = len(node_to_net_dict[node_name])
    
    node_net_num_fea= {}
    node_net_num_max = max(node_net_num.values())
    print("node_net_num_max", node_net_num_max)
    for node_name in node_info:
        node_net_num_fea[node_name] = node_net_num[node_name]/node_net_num_max
    
    node_area_fea = {}
    node_area_max_node = max(node_info, key = lambda x : node_info[x]['x'] * node_info[x]['y'])
    node_area_max = node_info[node_area_max_node]['x'] * node_info[node_area_max_node]['y']
    print("node_area_max = {}".format(node_area_max))
    for node_name in node_info:
        node_area_fea[node_name] = node_info[node_name]['x'] * node_info[node_name]['y'] / node_area_max
    
    if "V" in node_info:
        add_node = "V"
        visited_node.add(add_node)
        node_id_to_name.append((add_node, node_net_num[add_node]))
        node_net_num.pop(add_node)
    
    add_node = max(node_net_num, key = lambda v: node_net_num[v])
    visited_node.add(add_node)
    node_id_to_name.append((add_node, node_net_num[add_node]))
    node_net_num.pop(add_node)

    while len(node_id_to_name) < len(node_info):
        candidates = {}
        for node_name in visited_node:
            if node_name not in adjacency:
                continue
            for node_name_2 in adjacency[node_name]:
                if node_name_2 in visited_node:
                    continue
                if node_name_2 not in candidates:
                    candidates[node_name_2] = 0
                candidates[node_name_2] += 1
        for node_name in node_info:
            if node_name not in candidates and node_name not in visited_node:
                candidates[node_name] = 0
        if len(candidates) > 0:
            if benchmark != 'ariane':
                if benchmark == "bigblue3":
                    add_node = max(candidates, key = lambda v: candidates[v]*1 + node_net_num[v]*100000 +\
                        node_info[v]['x']*node_info[v]['y'] * 1 +int(hash(v)%10000)*1e-6)
                else:
                    add_node = max(candidates, key = lambda v: candidates[v]*1 + node_net_num[v]*1000 +\
                        node_info[v]['x']*node_info[v]['y'] * 1 +int(hash(v)%10000)*1e-6)
            else:
                add_node = max(candidates, key = lambda v: candidates[v]*30000 + node_net_num[v]*1000 +\
                    node_info[v]['x']*node_info[v]['y']*1 +int(hash(v)%10000)*1e-6)
        else:
            if benchmark != 'ariane':
                if benchmark == "bigblue3":
                    add_node = max(node_net_num, key = lambda v: node_net_num[v]*100000 + node_info[v]['x']*node_info[v]['y']*1)
                else:
                    add_node = max(node_net_num, key = lambda v: node_net_num[v]*1000 + node_info[v]['x']*node_info[v]['y']*1)
            else:
                add_node = max(node_net_num, key = lambda v: node_net_num[v]*1000 + node_info[v]['x']*node_info[v]['y']*1)

        visited_node.add(add_node)
        node_id_to_name.append((add_node, node_net_num[add_node])) 
        node_net_num.pop(add_node)
    for i, (node_name, _) in enumerate(node_id_to_name):
        node_info[node_name]["id"] = i
    # print("node_id_to_name")
    # print(node_id_to_name)
    node_id_to_name_res = [x for x, _ in node_id_to_name]
    return node_id_to_name_res

def path_copy(filePath,newFilePath):
    filename=os.listdir(filePath)
    for i in filename:
        shutil.copy(filePath+'/'+i,newFilePath+'/'+i)

class Placememt(gym.Env):
    def __init__(self, grid_shape=256,block='adaptec1',log_dir='./result'): 
        self.block_name = block
        self.macro_init()
        self.n = grid_shape
        self.grid_w = math.ceil(self.max_width/2560)*10
        self.grid_h = math.ceil(self.max_height/2560)*10
        tt = time.strptime(time.ctime())
        time_ = str(tt.tm_year)+'-'+str(tt.tm_mon)+str(tt.tm_mday)+'-'+str(tt.tm_hour)+str(tt.tm_min)+str(tt.tm_sec)
        self.log = log_dir
        self.macro_local = np.array([[-1.,-1.]]*self.macro_num,dtype=np.float32)
        # macro cluster
        self.soft_macro_num = int(self.macro_class.max())+1
        self.soft_macro_loca = np.array([[-1.,-1.]]*self.soft_macro_num,dtype=np.float32)
        #cluster内部有几个macro
        self.cluster_macro_num = np.array([[0]]*self.soft_macro_num,dtype=np.int32)       
        self.soft_macro_single_size = np.array([[-0.,-0.]]*self.soft_macro_num,dtype=np.float32)
        self.soft_macro_size = np.array([[-1.,-1.]]*self.soft_macro_num,dtype=np.float32)
        self.soft_macro_shape = np.array([[0,1]]*self.soft_macro_num,dtype=np.int32)
        self.cluster_macro_index = []
        for i in range(self.soft_macro_num):
            self.cluster_macro_index.append(np.where(self.macro_class==i)[0].tolist())
            self.cluster_macro_num[i][0] = np.where(self.macro_class==i)[0].shape[0]
            self.soft_macro_shape[i][0] = np.where(self.macro_class==i)[0].shape[0]
            self.soft_macro_single_size[i] = self.macro_size[np.where(self.macro_class==i)[0][0]]
        # sorted index
        single_size = self.soft_macro_single_size[:,0]*self.soft_macro_single_size[:,1]
        square = single_size*self.cluster_macro_num[:,0]
        self.place_index = square.argsort()[::-1]
        # sort the imformation
        self.cluster_macro_num = self.cluster_macro_num[self.place_index]
        self.soft_macro_single_size = self.soft_macro_single_size[self.place_index]
        self.soft_macro_shape = self.soft_macro_shape[self.place_index]
        self.soft_macro_size = self.soft_macro_size[self.place_index]
        self.soft_macro_islocated = np.array([[0]]*self.soft_macro_num,dtype=np.int32)
        
        self.action_space = Discrete(self.n * self.n)
        self.obs_space = (1, self.n, self.n)
        self.observation_space = gym.spaces.Dict({"Node_feature":gym.spaces.Box(low=-10000,high=10000,shape=(self.soft_macro_num,8),dtype=np.float32),
                                                 "Obs":gym.spaces.Box(low=0.,high=1.,shape=(1,self.n,self.n)),
                                                 "Mask":gym.spaces.Box(low=0.,high=1,shape=(self.n*self.n,)),
                                                 "Shape_Mask":gym.spaces.Box(low=0.,high=1.,shape=(self.macro_num,)),
                                                 "Current_macro_id":gym.spaces.Box(low=0,high=self.macro_num,shape=(1,))})
        self.best = -500
        self.params = Params.Params()
        # load parameters
        add = "./DREAMPlace/test/ispd2005/"+self.block_name+".json"
        self.params.load(add)

        os.environ["OMP_NUM_THREADS"] = "%d" % (self.params.num_threads)
        self.ori_path = '/'.join(self.params.__dict__['aux_input'].split('/')[:-1])

        logging.root.name = 'DREAMPlace'
        logging.basicConfig(level=logging.CRITICAL,format='[%(levelname)-7s] %(name)s - %(message)s',stream=sys.stdout)

    def macro_init(self):
        file = './benchmarks/ispd2005/'+self.block_name+'/'
        # read macro location
        node_path = file+self.block_name+'.nodes'
        self.macro_info,self.macro_info_raw_id_name = read_node_file(node_path,self.block_name)
        self.macro_num = len(self.macro_info)
        # read netlist
        net_path = file+self.block_name+'.nets'
        self.net_info = read_net_file(net_path, self.macro_info)
        self.net_num = len(self.net_info)

        pl_file = "./data/"+self.block_name+".pl"
        self.max_height, self.max_width = read_pl_file(pl_file, self.macro_info)

        self.macro_to_net_dict = get_node_to_net_dict(self.macro_info, self.net_info)
        #self.macro_id_to_name = get_node_id_to_name_topology(self.macro_info, self.macro_to_net_dict, self.net_info, self.block_name)
        self.macro_id_to_name = list(self.macro_info.keys())
        self.id_name_map = dict(zip(self.macro_id_to_name,[i for i in range(self.macro_num)]))
        net_num = np.zeros([self.macro_num,self.macro_num])
        with open('./macro_net.txt','w') as f:
            for i in range(self.macro_num):
                macro_list = []
                net_list = self.macro_to_net_dict[self.macro_id_to_name[i]]  #macro对应net
                for net in net_list:
                    macro_list += list(self.net_info[net]['nodes'].keys())     #相连的macro
                while self.macro_id_to_name[i] in macro_list:
                    macro_list.remove(self.macro_id_to_name[i])
                f.write(self.macro_id_to_name[i]+':'+' '.join(macro_list)+'\n')
                net_counter = Counter(macro_list)
                for key in net_counter.keys():
                    net_num[i][self.id_name_map[key]] = net_counter[key]

        value = list(self.macro_info.values())
        self.macro_ori = []
        self.macro_size = []
        for i in range(self.macro_num):
            self.macro_size.append([value[i]['x'],value[i]['y']])
            self.macro_ori.append([value[i]['raw_x'],value[i]['raw_y']])
        self.macro_size = np.array(self.macro_size)
        self.macro_ori = np.array(self.macro_ori)
        #self.macro_class = np.arange(0,self.macro_num)
        self.cluster_engine(net_num)   
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self,gpu_id):
        self.gpu_id = gpu_id
        # reset macro info   
        self.macro_local = np.array([[-1.,-1.]]*self.macro_num,dtype=np.float32)
        self.soft_macro_loca = np.array([[-1.,-1.]]*self.soft_macro_num,dtype=np.float32)
        self.soft_macro_size = np.array([[-1.,-1.]]*self.soft_macro_num,dtype=np.float32)
        self.soft_macro_shape = np.array([[0,1]]*self.soft_macro_num,dtype=np.int32)
        self.soft_macro_islocated = np.array([[0]]*self.soft_macro_num,dtype=np.int32)
        for i in range(self.soft_macro_num):
            self.soft_macro_shape[i][0] = np.where(self.macro_class==i)[0].shape[0]
        self.soft_macro_shape = self.soft_macro_shape[self.place_index]    
        # reset state
        self.obs_mask = torch.ones((256,256))
        #0:y,1:x
        self.max_grid_size = int(self.max_width/self.grid_w)
        self.obs_mask[:,int(self.max_width/self.grid_w)+1:] = 0.0
        self.obs_mask[int(self.max_height/self.grid_h)+1:,:] = 0.0
        self.placing_id = 0
        self.placing_shape = math.ceil(self.cluster_macro_num[0]/3)
        self.soft_macro_shape[self.placing_id] = np.array([self.placing_shape,math.ceil(self.cluster_macro_num[self.placing_id]/self.placing_shape)])
        self.soft_macro_size[self.placing_id] = (self.soft_macro_single_size[self.placing_id]+np.array([5,5]))*self.soft_macro_shape[self.placing_id]
                
        t0 = time.time()
        action_mask = self.compute_mask(self.soft_macro_size[0])
        t1 = time.time()
        # print(t1-t0)
        shape_mask = self.compute_shape_mask()
        self.soft_macro_single_size_norm = self.soft_macro_single_size/np.array([self.max_width,self.max_height])
        self.node_feature = np.concatenate((self.soft_macro_loca,self.soft_macro_single_size_norm,self.cluster_macro_num,self.soft_macro_shape,self.soft_macro_islocated),axis=1,dtype=np.float32)
        self.state = {"Node_feature":self.node_feature,"Obs":self.obs_mask,"Mask":action_mask,"Shape_Mask":shape_mask,"Current_macro_id":self.placing_id}
        state = copy.deepcopy(self.state)
        return state

    def step(self, actor):
        if self.state["Mask"].sum() == 0:
            actor[0] = 0
        action = actor[0]
        next_shape = actor[1]

        action_x = action % self.n
        x = action_x*self.grid_w
        action_y = action // self.n
        y = action_y*self.grid_h

        place_id = self.placing_id
        self.soft_macro_loca[place_id][0] = x
        self.soft_macro_loca[place_id][1] = y
        self.update_macro_loca(x,y,place_id)

        self.placing_id = (place_id+1)%(self.soft_macro_num)
        self.placing_shape = int(next_shape)
        self.soft_macro_shape[self.placing_id] = np.array([self.placing_shape,math.ceil(self.cluster_macro_num[self.placing_id]/self.placing_shape)])
        self.soft_macro_size[self.placing_id] = (self.soft_macro_single_size[self.placing_id]+np.array([5,5]))*self.soft_macro_shape[self.placing_id]
        next_soft_macro_size = self.soft_macro_size[self.placing_id]

        #update state
        self.update_obs(self.soft_macro_size[place_id],x,y)
        t3 = time.time()
        action_mask = self.compute_mask(next_soft_macro_size)
        t4 = time.time()
        #print(t4-t3)
        shape_mask = self.compute_shape_mask()
        self.state["Node_feature"][place_id][0] = x/self.max_width
        self.state["Node_feature"][place_id][1] = y/self.max_height
        self.state["Node_feature"][place_id][5] = self.soft_macro_shape[self.placing_id][0]
        self.state["Node_feature"][place_id][6] = self.soft_macro_shape[self.placing_id][1]
        self.state["Node_feature"][place_id][7] = 1.
        self.state["Mask"] = action_mask
        self.state['Obs'] = self.obs_mask
        self.state["Shape_Mask"] = shape_mask
        self.state["Current_macro_id"] = self.placing_id
        state = copy.deepcopy(self.state)
        
        #需要改一下结束条件
        if place_id == self.soft_macro_num-1:
            # 开始执行EA
            done = True
            boundary = np.array([self.max_width,self.max_height])
            wl, overf = self.new_cal_re()
            tt = time.strptime(time.ctime())
            t = str(tt.tm_year)+'-'+str(tt.tm_mon)+str(tt.tm_mday)+'-'+str(tt.tm_hour)+str(tt.tm_min)+str(tt.tm_sec)
            #self.render(wl,overf,'before',t)
            # population = Population(self.macro_local, self.macro_size, self.macro_class, boundary, self.soft_macro_shape, 5, 10, wl,self.id_name_map)
            # wl = population.Optimize()
            self.render(wl,overf,'after',t)
            reward = (10-1* wl * 1e-7)
            if reward > self.best:
                self.best = reward
        else:
            done = False
            
            #reward = 0.
            reward = compute_intrinsic_reward(rnd, self.obs_mask)
            if self.state["Mask"].sum() == 0:
                reward = -0.1
        return state, done, torch.FloatTensor([[reward]]), actor
    
    def update_obs(self,macro_size,x,y):
        action_x = math.floor(x/self.grid_w)
        action_y = math.floor(y/self.grid_h)
        macro_w,macro_h = math.ceil((macro_size[0]+x+5.0)/self.grid_w),math.ceil((macro_size[1]+y+5.0)/self.grid_h)
        macro_w = min(macro_w,self.n-1)
        macro_h = min(macro_h,self.n-1)
        for i in range(action_x,macro_w+1):
            for j in range(action_y,macro_h+1):
                self.obs_mask[j][i] = 0.0

    def compute_mask(self,macro_size):
        mask = np.zeros(self.n*self.n)
        macro_w,macro_h = math.ceil((macro_size[0]+5.0)/self.grid_w),math.ceil((macro_size[1]+5.0)/self.grid_h)
        macro_s = macro_h*macro_w
        for i in range(self.n-macro_h+1):
            for j in range(self.n-macro_w+1):
                if self.obs_mask[i][j] == 0:
                    continue
                if self.obs_mask[j:j+macro_h,i:i+macro_w].sum() == macro_s:
                    mask[j*self.n+i] = 1.0
        return mask
    def compute_shape_mask(self):
        mask = np.zeros(self.macro_num)
        next_id = (self.placing_id+1)%(self.soft_macro_num)
        for i in range(1,self.cluster_macro_num[next_id][0]+1):
            shape = np.array([i,math.ceil(self.cluster_macro_num[next_id]/i)])
            size = shape*self.soft_macro_single_size[next_id]
            if size[1]>=self.max_height:
                continue
            if size[0]>=self.max_width:
                continue
            mask[i] = 1
        return mask
    def update_macro_loca(self,x,y,place_id):
        # compute macro loca according the cluster loca
        column,row = 0,0
        distance = self.soft_macro_single_size[place_id]+np.array([5,5])
        for index in self.cluster_macro_index[self.place_index[place_id]]:
            self.macro_local[index] = np.array([int(x),int(y)])+np.array([column,row])*distance
            column += 1
            if column == self.soft_macro_shape[place_id][0]:
                column = 0
                row += 1
    def render(self,wl,overf,flag,times):
        color_list = list(mcolors.CSS4_COLORS.keys())
        ration = (self.max_width+5)/(self.max_height+5)
        fig = plt.figure(figsize = (int(10*ration),10))
        ax = fig.add_subplot(111)
        ticks_x = np.arange(0,self.max_width+5,(self.max_width)/10)
        ticks_y = np.arange(0,self.max_height+5,(self.max_height)/10)
        plt.xlim(0,self.max_width+5)
        plt.ylim(0,self.max_height+5)
        plt.title(' wl:'+str(wl)+' overflow:'+str(overf))
        ax.set_xticks(ticks_x)
        ax.set_yticks(ticks_y)
        ax.add_patch(plt.Rectangle((0,0),self.max_width,self.max_height,linewidth=2,edgecolor='k',facecolor='white'))
        for i in range(self.macro_num):
            rect_i = ax.add_patch(plt.Rectangle(self.macro_local[i],self.macro_size[i][0],self.macro_size[i][1],linewidth=2,edgecolor='k',facecolor='yellow'))
            # facecolor=color_list[int(self.macro_class_fig[i])]

            # ax.text(x = rect_i.get_x()+self.macro_size[i][0]/2,
            #         y = rect_i.get_y()+self.macro_size[i][1]/2,
            #         s = str(i+1),
            #         fontsize = 10)
        plt.savefig(self.log+'/'+times+flag+'+wl-'+str(wl)+'.jpg')
        plt.close()
    def cluster_engine(self,data):
        macro_cluster = [[i] for i in range(self.macro_num)]
        for k in range(self.macro_num):
            if max(macro_cluster)==127:
                break
            if data.sum() == 0:
                break
            sim = np.zeros_like(data)
            net_num=np.where(data==0,0,1).sum(0)
            for i in range(sim.shape[0]):
                for j in range(sim.shape[1]):
                    sim[i][j] = data[i][j]/(net_num[i]*net_num[j]+1e-9)
            index = np.unravel_index(sim.argmax(), sim.shape)
            macro_cluster[index[0]] += macro_cluster[index[1]]
            del macro_cluster[index[1]]
            data[index[0]] = data[index[0]]+data[index[1]]
            data[:,index[0]] = data[:,index[0]]+data[:,index[1]]
            data[index[0],index[0]] = 0
            data = np.delete(data,index[1],0)
            data = np.delete(data,index[1],1)
        self.macro_class = np.zeros(self.macro_num)
        classes = 0
        for macros in macro_cluster:
            size_dict = {}
            macros = np.array(macros)
            for index in macros:
                size = self.macro_size[index]
                if tuple(size) not in size_dict:
                    size_dict[tuple(size)] = classes
                    classes += 1
                self.macro_class[index] = size_dict[tuple(size)]
    
    def new_cal_re(self):
        #把result写回
        input_path = './worker_dir/'+str(time.time())[:10]+str(random.random())[2:]
        if not os.path.exists(input_path):
            os.makedirs(input_path)
        path_copy(self.ori_path,input_path)
        self.params.__dict__['aux_input'] = input_path+'/'+self.block_name+'.aux'
        write(self.macro_local,self.block_name,self.id_name_map,input_path)

        r = place(self.params)
        wl = float(r[0].hpwl.data)
        overf = float(r[0].overflow.data)
        print(wl,overf)
        shutil.rmtree(input_path)
        return wl,overf
def fullplace_envs(log_dir):
    return Placememt(log_dir=log_dir)