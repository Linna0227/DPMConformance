from inspect import trace
from locale import currency
from matplotlib.pyplot import get
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.util import xes_constants as xes
from pm4py.util import constants as pm4_constants
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
import re
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.algo.conformance.alignments.petri_net.algorithm import Variants
from pytz import NonExistentTimeError
from pyspark import SparkContext
import time
from pm4py.objects.conversion.process_tree import converter as pt_converter
import sys
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.conversion.wf_net import converter as wf_net_converter
from pm4py.util import exec_utils
from pm4py.statistics.variants.log import get as variants_module
from pm4py.algo.conformance.alignments.petri_net.algorithm import Parameters
import hrdecomposetreela as dt
import time
from copy import copy
from pm4py.util.xes_constants import DEFAULT_NAME_KEY
from pm4py.objects.process_tree.obj import Operator
from pm4py.objects.process_tree.importer import importer as ptml_importer
from pm4py.util.constants import PARAMETER_CONSTANT_ACTIVITY_KEY, PARAMETER_CONSTANT_CASEID_KEY

'''describe:
Firstly, the trace and subnets are allocated, 
and fitness of each trace is calculated only with the corresponding subnets.
'''
def get_variants_structure(log, parameters):
    if parameters is None:
        parameters = copy({PARAMETER_CONSTANT_ACTIVITY_KEY: DEFAULT_NAME_KEY})

    parameters = copy(parameters)
    variants_idxs = exec_utils.get_param_value(Parameters.VARIANTS_IDX, parameters, None)
    if variants_idxs is None:
        variants_idxs = variants_module.get_variants_from_log_trace_idx(log, parameters=parameters)

    one_tr_per_var = []
    variants_list = []
    for index_variant, var in enumerate(variants_idxs):
        variants_list.append(var)

    for var in variants_list:
        one_tr_per_var.append(log[variants_idxs[var][0]])

    return variants_idxs, one_tr_per_var


def form_result(log, variants_idxs, all_result):
    al_idx = {}
    for index_variant, variant in enumerate(variants_idxs):
        for trace_idx in variants_idxs[variant]:
            al_idx[trace_idx] = all_result[index_variant]

    results_con = []
    for i in range(len(log)):
        results_con.append(al_idx[i])

    return results_con

# compute the cost of model

def compute_cost(node, cost):
    if node.operator is None:
        if node.label is not None:
            return 10000
        else:
            return 1
    else:
        if node.operator == Operator.SEQUENCE:
            for child in node.children:
                if child.operator is None:
                    cost += compute_cost(child, cost)
                else:
                    cost = compute_cost(child, cost)
                # print(child, ",", cost)

        if node.operator == Operator.XOR:
            select = list()

            for child in node.children:
                if child.operator is None:
                    cost += compute_cost(child, cost)
                    return cost
                else:
                    select.append(child)
            if len(select) == len(node.children):
                m = compute_cost(select[0], cost)
                # print("m", m)
                for opt in select[1:]:
                    n = compute_cost(opt, cost)
                    # print("n", n)
                    if n < m:
                        m = n
                cost = m

                    # print(cost)

            # print(child, "xor", cost)
        if node.operator == Operator.PARALLEL:
            for child in node.children:
                if child.operator is None:
                    cost += compute_cost(child, cost)
                else:
                    cost = compute_cost(child, cost)
                # print(child, "*", cost)
        if node.operator == Operator.LOOP:
            child = node.children[0]
            if child.operator is None:
                cost += compute_cost(child, cost)
            else:
                cost = compute_cost(child, cost)

    return cost


def get_tree_cost(tree):
    root = tree._get_root()
    current_node = root
    tree_cost= compute_cost(current_node, 0)
    return tree_cost

# covert rdd strings to log 
def covertToXlog(traces):
    log=EventLog()
    parameters={}
    activity_key = parameters[
        pm4_constants.PARAMETER_CONSTANT_ACTIVITY_KEY] if pm4_constants.PARAMETER_CONSTANT_ACTIVITY_KEY in parameters else xes.DEFAULT_NAME_KEY

    for traceString in traces:
        # traceString=re.sub(r'[0-9]+,', '', traceString)
        traceString=traceString.strip('\n')
        eventnames=traceString.split(",")
        trace=Trace()
        trace._set_attributes({'concept:name': eventnames[0]})
        for name in eventnames[1:]:
            event=Event()
            event[activity_key]=name
            trace.append(event)
        log.append(trace)
    return log

# call function reformat times: number of partition blocks  (2 times)
def reformat(partitiondata):
    updatedata=list()
    updatedata.append(covertToXlog(partitiondata))
    return iter(updatedata)


def evaluate_token(replay_traces):
    perc_fit_traces = 0.0
    average_fitness = 0.0
    log_fitness = 0
    no_traces = 0
    fit_traces = 0
    sum_of_fitness = 0
    total_m = 0
    total_c = 0
    total_p = 0
    total_r = 0
    for block in replay_traces:
        no_traces += len(block)
        fit_traces += len([x for x in block if x["trace_is_fit"]])
        sum_of_fitness += sum([x["trace_fitness"] for x in block])
        
        total_m += sum([x["missing_tokens"] for x in block])
        total_c += sum([x["consumed_tokens"] for x in block])
        total_r += sum([x["remaining_tokens"] for x in block])
        total_p += sum([x["produced_tokens"] for x in block])
    if no_traces > 0 and total_c > 0 and total_p > 0:
        perc_fit_traces = float(100.0 * fit_traces) / float(no_traces)
        average_fitness = float(sum_of_fitness) / float(no_traces)
        log_fitness = 0.5 * (1 - total_m / total_c) + 0.5 * (1 - total_r / total_p)
    print("m:",total_m,"  c:",total_c,"  p:",total_p,"  r:",total_r)
    return {"perc_fit_traces": perc_fit_traces, "average_trace_fitness": average_fitness, "log_fitness": log_fitness
             }
def get_treeleaves(ptree):
    root = ptree._get_root()
    leaves = list()
    if root._get_children() != list():
        child_nodes = root._get_children()
        getleaves = True
        while getleaves:
            leaves_to_replace = list()
            new_childnodes = list()
            for child in child_nodes:
                if child._get_children() != list():
                    leaves_to_replace.append(child)
                else:
                    leaves.append(child.label)
            if leaves_to_replace != list():
                for child in leaves_to_replace:
                    for el in child.children:
                        new_childnodes.append(el)
                child_nodes = new_childnodes
            else:
                getleaves = False
    else:
        leaves.append(root.label)
    return set(leaves)

def get_trace_set(trace_a):
    activity_key = "concept:name"
    trace_activities = [event[activity_key] for event in trace_a]
    trace_set = set(trace_activities)
    return trace_set

def get_model_score(set1, set2):
    a = set1.intersection(set2)
    b = set1.union(set2)
    score = len(a)/len(b)
    return score

def get_model(o_trace):
    trace_actset=get_trace_set(o_trace)
    current_score=0
    m_net=broadcast_net.value
    current_net=m_net[0][0]
    for one_net in m_net:
        model_score=get_model_score(one_net[1],trace_actset)
        if model_score >= current_score:
            current_score=model_score
            current_net=one_net[0]
    return current_net

# token replay
def cost_token(partitiondatas):
    replay_traces=list()
    for ilog in partitiondatas:
        variants_id,variant_traces=get_variants_structure(ilog,None)
        replay_ilog=list()
        for one_trace in variant_traces:
            confor_net=get_model(one_trace)
            log=EventLog()
            log.append(one_trace)
            replay_trace=token_replay.apply(log,confor_net[0],confor_net[1],confor_net[2])
            replay_ilog.append(replay_trace[0])
        r_f_ilog=form_result(ilog,variants_id,replay_ilog)
        replay_traces.append(r_f_ilog)

    return iter(replay_traces)

# alignmnet

def cost_alignment(partitiondatas):

    aligned_traces=list()
    for ilog in partitiondatas:
        variants_id,variant_traces=get_variants_structure(ilog,None)
        print(variant_traces)
        aligned_ilog=list()
        for one_trace in variant_traces:
            confer_net=get_model(one_trace)
            log=EventLog()
            log.append(one_trace)
            aligned_trace=alignments.apply_log(log,confer_net[0],confer_net[1],confer_net[2])

            aligned_trace[0]['bwc']=len(one_trace)
            aligned_ilog.append(aligned_trace[0])
        r_f_ilog=form_result(ilog,variants_id,aligned_ilog)
        aligned_traces.append(r_f_ilog)

    return iter(aligned_traces)


def evaluate_alignment(aligned_traces,best_worse_model_cost):

    # no_traces = len([x for x in aligned_traces if x is not None])
    no_traces = 0
    no_fit_traces = 0
    sum_fitness = 0.0
    sum_bwc = 0.0
    sum_cost = 0.0
    queued_states = 0
    traversed_arcs = 0


    

    for block in aligned_traces:
        for tr in block:
            if tr is not None:
                no_traces += 1
                if tr["fitness"] == 1.0:
                    no_fit_traces = no_fit_traces + 1
                # sum_fitness += tr["fitness"]
                sum_bwc =sum_bwc+ tr["bwc"]+best_worse_model_cost
                sum_cost += tr["cost"]

                queued_states += tr['queued_states']
                traversed_arcs += tr['traversed_arcs']
                sum_fitness+=tr["fitness"]





    perc_fit_traces = 0.0
    average_fitness = 0.0
    log_fitness = 0.0

    if no_traces > 0:
        perc_fit_traces = (100.0 * float(no_fit_traces)) / (float(no_traces))
        average_fitness = float(sum_fitness) / float(no_traces)
        log_fitness = 1.0 - float(sum_cost//10000) / float(sum_bwc)
    print("sum_cost:",sum_cost//10000)
    print("sum_bwc:",sum_bwc)
    return {"percFitTraces": perc_fit_traces, "averageFitness": average_fitness,
            "log_fitness": log_fitness, "queued_states": queued_states, "traversed_arcs": traversed_arcs}


if __name__=="__main__":
    tree_path = sys.argv[1]
    log_path = sys.argv[2]
    Dthreshold = sys.argv[3]
    partition_n = sys.argv[4]
    method = sys.argv[5]
    algor=sys.argv[6]

    Dthreshold=int(Dthreshold)
    partition_n=int( partition_n)

    tree = ptml_importer.apply(tree_path)
    start_time=time.time()
    best_worse_cost=get_tree_cost(tree)
    best_worse_cost=best_worse_cost//10000
    start_time1=time.time()
    sub_trees=dt.apply(tree, Dthreshold)
    end_time1=time.time()
    print("Decomposition is finished in {} s".format(end_time1-start_time1))

    
    start_time22=time.time()
    # print(start_time22)
    # print(start_time22)
    # print(start_time22)
    sub_nets=list()
    C_NET_START= time.time()
    for sub_tree in sub_trees:
        comb_net_actset=list()
        actset=get_treeleaves(sub_tree)
        net = pt_converter.apply(sub_tree)
        comb_net_actset.append(net)
        comb_net_actset.append(actset)
        sub_nets.append(comb_net_actset)
    C_NET_END=time.time()
    print("coversion is finished in {} s.".format(C_NET_END-C_NET_START))
    print("coversion is finished in {} s.".format(C_NET_END-C_NET_START))
    print("coversion is finished in {} s.".format(C_NET_END-C_NET_START))
    sc = SparkContext( )
    broadcast_net=sc.broadcast(sub_nets)
    broadcast_algor=sc.broadcast(algor)


    # log's type is rdd  
    start_time2=time.time()  
    log = sc.textFile(log_path,partition_n)

    # 'str' covert to event log
    sublogs=log.mapPartitions(reformat)
    # print("number of traces:",sublogs.count())

    if method == 'tokenreplay' or method == 'token':    
        cost=sublogs.mapPartitions(cost_token).collect()
        result=evaluate_token(cost)
        print(result)
        print(result)
    elif method == 'alignment':
        cost=sublogs.mapPartitions(cost_alignment).collect()
        result=evaluate_alignment(cost,best_worse_cost)
        print(result)

    end_time2=time.time()
 
    print("conformance is finished in {} s".format(end_time2-start_time2))

    print("all process is finished in {} s".format(end_time2-start_time))
    

    sc.stop()








