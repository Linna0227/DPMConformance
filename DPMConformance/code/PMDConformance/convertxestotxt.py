from pm4py.objects.log.importer.xes import importer as xes_importer
import time
from pm4py.util import constants

CASE_ID_KEY = constants.PARAMETER_CONSTANT_CASEID_KEY
ACTIVITY_KEY = constants.PARAMETER_CONSTANT_ACTIVITY_KEY


def get_trace_set(trace_a):
    print(trace_a)
    activity_key = "concept:name"
    trace_activities = [event[activity_key] for event in trace_a]
    print(trace_activities)
    return trace_activities


def convertxestotxt(filename,savepath):
    log = xes_importer.apply(filename)
    f1 = open(savepath, 'a', encoding='UTF-8')
    for trace in log:
        strtrace = ""
        # print(trace)
        events = get_trace_set(trace)
        attributes = trace._get_attributes()
        case_id = attributes.get('concept:name')
        strtrace += case_id+","

        i = 0
        if len(events) > 1:
            for j in range(len(events)-1):
                # strtrace += events[i]+"+complete,"
                strtrace += events[i]+","
                i = i+1
            # strtrace += events[i]+"+complete"
            strtrace += events[i]
            strtrace += "\n"


            f1.write(strtrace)
        elif len(events) == 1:
            # strtrace = events[i]+"complete"+"\n"
            strtrace += events[i]+"\n"
            f1.write(strtrace)
        else:
            strtrace += " \n"
        print(strtrace)
    f1.close()




folder_name = ['BPM2013']
file_num = ['A','B','C','D','E','F','G']

for folder in folder_name:
    for num in file_num:
        filename="/home/hadoop/Projects/testdata/"+folder+"/log/pr"+num+"m6.xes"
        savepath="/home/hadoop/Projects/testdata/"+folder+"/txt/pr"+num+"m6.txt"
        start = time.clock()
        convertxestotxt(filename, savepath)
        end = time.clock()
        print("converter log used {} s.".format(end-start))

