import numpy as np
import copy
import math
import copy
import random
import os
import shutil
import time
import dreamplace.configure as configure
import dreamplace.Params as Params
import dreamplace.PlaceDB as PlaceDB
import dreamplace.NonLinearPlace as NonLinearPlace

def path_copy(filePath,newFilePath):
    filename=os.listdir(filePath)
    for i in filename:
        shutil.copy(filePath+'/'+i,newFilePath+'/'+i)

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

def write(res,name,idmap):
    f = open("./benchmarks/ispd2005/"+name+"/"+name+".pl", "w")
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

class Individual:
    def __init__(self,macro_loca,fitness,shape,macro_class):
        self.macro_loca = macro_loca
        self.fitness = fitness
        self.class_shape = shape
        self.macro_class = macro_class

class operator:
    def __init__(self,macro_size,canvas_boundary,id_name_map):
        #输入为numpy数组
        self.macro_size = macro_size.copy()
        self.canvas_boundary = canvas_boundary.copy()
        self.num = self.macro_size.shape[0]
        self.id_name_map = id_name_map
        #DREAMPLace
        self.params = Params.Params()
        #load parameters
        self.block_name = 'adaptec1'
        add = "./DREAMPlace/test/ispd2005/"+self.block_name+".json"
        self.params.load(add)
        os.environ["OMP_NUM_THREADS"] = "%d" % (self.params.num_threads)
        self.ori_path = '/'.join(self.params.__dict__['aux_input'].split('/')[:-1])

    def Swap(self,individual,child_nums):
        #选定一个macro，然后随机选择一个和他大小相同的macro，交换两者位置和class
        res = []
        for i in  range(child_nums):
            macro_loca = individual.macro_loca.copy()
            macro_class = individual.macro_class.copy()
            select_id = random.randint(0,self.num-1)
            indexs = np.where((self.macro_size==self.macro_size[select_id]).all(1))[0]
            index = np.random.choice(indexs)
            if index == select_id:
                continue
            macro_loca[index], macro_loca[select_id] = macro_loca[select_id], macro_loca[index]
            macro_class[index], macro_class[select_id] = macro_class[select_id], macro_class[index]
            fit = self.eval_result(macro_loca)     #占位
            individ = Individual(macro_loca, fit, individual.class_shape, macro_class)
            res.append(individ)
        return res

    def Shuffle(self,individual,child_nums):
        #不改变cluster形状，随机排列顺序
        res = []
        for i in range(child_nums):
            macro_loca = individual.macro_loca.copy()
            macro_class = individual.macro_class.copy()
            class_id = random.randint(0,macro_class.max())
            macro_id = np.where(macro_class==class_id)[0]
            new_id = np.random.permutation(macro_id)
            for i in range(macro_id.shape[0]):
                macro_loca[new_id[i]] = individual.macro_loca[macro_id[i]]
            fit = self.eval_result(macro_loca)     #占位
            individ = Individual(macro_loca,fit,individual.class_shape,macro_class)
            res.append(individ)
        return res

    def Shape_change(self,individual,child_nums):
        #变换cluster形状。
        res = []
        for i in range(child_nums):
            macro_loca = individual.macro_loca.copy()
            macro_class = individual.macro_class.copy()
            class_id = random.randint(0,macro_class.max())
            class_shape = individual.class_shape.copy()
            macro_id = np.where(macro_class==class_id)[0]
            nums = len(macro_id)
            new_width = random.randint(1,nums)
            old_shape = class_shape[macro_id]
            new_shape = np.array([new_width,math.ceil(nums/new_width)])  
            if (new_shape == old_shape).all():
                continue
            # ...
        return individual
    def Move(self,individual,child_nums):
        #将某个macro加入到其他cluster
        res = []
        for i in range(child_nums):
            macro_loca = individual.copy()
            macro_class = individual.macro_class.copy()
            select_id = random.randint(0,self.num-1)
            class_list = np.unique(macro_class[np.where((self.macro_size==self.macro_size[select_id]).all(1))[0]])
            select_class = np.random.choice(class_list)
            if select_class == macro_class[select_id]:
                continue
            # ...
        return individual            

    def Macro_lega(self,macro_loca):
        #占位
        return macro_loca
    
    def eval_result(self,macro_loca):   
        #根据评估重新构建染色体
        input_path = './worker_dir/'+str(time.time())[:10]+str(random.random())[2:]
        if not os.path.exists(input_path):
            os.makedirs(input_path)
        path_copy(self.ori_path,input_path)
        self.params.__dict__['aux_input'] = input_path+'/'+self.block_name+'.aux'

        write(macro_loca,self.block_name,self.id_name_map)
        r = place(self.params)
        wl = float(r[0].hpwl.data)
        shutil.rmtree(input_path)

        return wl

class Mutation:
    def __init__(self,macro_size,canvas_boundary,id_name_map):
        self.macro_size = macro_size.copy()
        self.canvas_boundary = canvas_boundary.copy()
        self.oper = operator(self.macro_size,self.canvas_boundary,id_name_map)
    def step(self,individual,child_nums):
        act_id = np.array([1,2,3,4])
        prob = np.array([0.5,0.5,0,0])
        act = np.random.choice(act_id,size=1,p=prob)
        if act == 1:
            result = self.oper.Swap(individual,child_nums)
        elif act == 2:
            result = self.oper.Shuffle(individual,child_nums)
        elif act == 3:
            result = self.oper.Shape_change(individual,child_nums)
        elif act == 4:
            result = self.oper.Move(individual,child_nums)
        return result

class Population:
    def __init__(self, macro_loca, macro_size, macro_class, canvas_boundary, cluster_shape, n, max_num,ori_fit,id_name_map):
        # self.max_iter :最大迭代次数 
        # self.max_num :种群最大个体数
        self.max_iter = n
        self.max_num = max_num
        self.best_wl = 1
        self.individuals = []
        self.ori_local = macro_loca.copy()
        self.ori_shape = cluster_shape.copy()
        self.macro_size = macro_size.copy()
        self.canvas_boundary = canvas_boundary.copy()
        self.reset(ori_fit,macro_class)
        self.mutation = Mutation(self.macro_size,self.canvas_boundary,id_name_map)

    def reset(self,fit,macro_class):
        ori_individual = Individual(self.ori_local,fit,self.ori_shape,macro_class)
        self.individuals.append(ori_individual)

    def Optimize(self):
        new_inds = []
        for i in range(self.max_iter):
            for individual in self.individuals:
                new_inds += self.mutation.step(individual,1)
            self.individuals += new_inds
            self.Rank()
        return self.individuals[0].fitness
        
    def Rank(self):
        if len(self.individuals)<=self.max_num:
            return
        fits = [individ.fitness for individ in self.individuals]   
        fits = np.array(fits)
        sort_index = fits.argsort()[:10]
        self.individuals = [self.individuals[i] for i in sort_index]
