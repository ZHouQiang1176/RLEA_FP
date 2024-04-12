root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)
import dreamplace.configure as configure
import dreamplace.Params as Params
import dreamplace.PlaceDB as PlaceDB
import dreamplace.NonLinearPlace as NonLinearPlace
import os
import logging
import time
import random
import pickle
import torch
import argparse

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

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--block', default='adaptec1', help='worker arguments')
    parser.add_argument(
        '--', default='', help='worker arguments')
    
    args = parser.parse_args()
    return args

def task_get(log_path):
    
    return log_path

def task_check(log_path):
    target_log = log_path+'/result.pkl'
    while not os.path.isfile(target_log):
        time.sleep(random.random()*10)
    with open(target_log,'r') as f:
        ret = pickle.load(f)
    return ret
def task_submit(input_path,res):
    log_path = input_path+'/result.pkl'
    with open(log_path,'w') as f:
        f.write(res)
def main(args):
    #worker流程：
    #1.envs创建文件夹，生成任务；
    #2.worker获取文件夹名字，读取相关信息初始化dreamplace；
    #3.dreamplace完成评估，提交结果（task_submit)
    #4.envs利用task_check检查是否产生了结果。
    params = Params.Params()
    # load parameters
    add = "./DREAMPlace/test/ispd2005/"+args.block_name+".json"
    params.load(add)

    os.environ["OMP_NUM_THREADS"] = "%d" % (params.num_threads)
    while True:
        print('start')
        input_path = task_get()
        params.__dict__['aux_input'] = input_path+'/'+args.block_name+'.aux'
        res = place(params)
        task_submit(input_path,res)
        torch.cuda.empty_cache()
        
if __name__=='__main__':
    args = get_args()
    main(args)