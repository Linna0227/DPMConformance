from pm4py.objects.process_tree.importer import importer as ptml_importer
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.algo.evaluation.replay_fitness.variants import alignment_based
import time
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.algo.evaluation.replay_fitness.variants import token_replay as tokenreplay_fitness
from pm4py.objects.conversion.process_tree import converter as pt_converter

from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.util import xes_constants as xes
from pm4py.util import constants as pm4_constants
def covertToXlog(traces):
    log=EventLog()
    parameters={}
    activity_key = parameters[
        pm4_constants.PARAMETER_CONSTANT_ACTIVITY_KEY] if pm4_constants.PARAMETER_CONSTANT_ACTIVITY_KEY in parameters else xes.DEFAULT_NAME_KEY

    for traceString in traces:
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


def evaluate_alignment(aligned_traces):

    no_traces = len([x for x in aligned_traces if x is not None])
    no_fit_traces = 0
    sum_fitness = 0.0
    sum_bwc = 0.0
    sum_cost = 0.0
    queued_states = 0
    traversed_arcs = 0

    for tr in aligned_traces:
        if tr is not None:
            if tr["fitness"] == 1.0:
                no_fit_traces = no_fit_traces + 1
            sum_fitness += tr["fitness"]
            sum_bwc += tr["bwc"]
            sum_cost += tr["cost"]
            # if tr["fitness"] < 1:
            #     print(tr)
            # print(tr["cost"])
            queued_states += tr['queued_states']
            traversed_arcs += tr['traversed_arcs']
    print("number os no_fit_traces",no_traces)
    print("sum_fitness",sum_fitness)

    perc_fit_traces = 0.0
    average_fitness = 0.0
    log_fitness = 0.0
    print("sum_cost:",sum_cost//10000)
    print("sum_bwc",sum_bwc)


    if no_traces > 0:
        perc_fit_traces = (100.0 * float(no_fit_traces)) / (float(no_traces))
        average_fitness = float(sum_fitness) / float(no_traces)
        log_fitness = 1.0 - float(sum_cost//10000) / float(sum_bwc)
    return {"percFitTraces": perc_fit_traces, "averageFitness": average_fitness,
            "log_fitness": log_fitness, "queued_states": queued_states, "traversed_arcs": traversed_arcs}


logpath="/home/hadoop/Projects/testdata/L5/L5.xes"
netpath="/home/hadoop/Projects/testdata/L5/L5.pnml"


log = xes_importer.apply(logpath)
net, initial_marking, final_marking = pnml_importer.apply(netpath)
print(logpath)
print(netpath)
# method = ""
method = "token"
if method == "alignment":
    # alignment
    # start = time.clock()
    start=time.time()
    aligned_traces = alignments.apply_log(log, net, initial_marking, final_marking)
    # aligned_traces = alignments.apply_log(log, net, initial_marking, final_marking, parameters=parameters)
    log_fitness = evaluate_alignment(aligned_traces)
    print(log_fitness)
    # end = time.clock()
    end=time.time()
    print("alignment is finished in {} s".format(end - start))
elif method == "token" or method == "tokenreplay" or method == "token replay":
    # tokenreplay
    # start_time = time.clock()
    start_time=time.time()
    replayed_traces = token_replay.apply(log, net, initial_marking, final_marking)
    replay_fitness = tokenreplay_fitness.evaluate(replayed_traces)
    print(replay_fitness)
    # end_time = time.clock()
    end_time=time.time()
    print("token replay is finished in {} s".format(end_time-start_time))
