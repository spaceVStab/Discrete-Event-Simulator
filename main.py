import numpy as np 
import os 
import random 
from model import Machine, Jobs

def printStatus(machines, jobs):
    print("Machines if occupied")
    for i in range(len(machines)):
        print("{} : {}".format(machines[i].machineName, str(machines[i].machineBusy)))
    print("Jobs if processed")
    for j in range(len(jobs)):
        print(jobs[j].jobName)
        print(jobs[j].processDetails)    


def getProcessingTime(MachineID, JobID, PROCESSING_TIME):
    return PROCESSING_TIME[JobID][MachineID]


def resetStates(NUM_MACHINES, NUM_JOBS):
    NUM_MACHINES = NUM_MACHINES
    NUM_JOBS = NUM_JOBS
    machines = [Machine(i) for i in range(NUM_MACHINES)]
    jobs = [Jobs(i) for i in range(NUM_JOBS)]
    # printStatus(machines, jobs)
    return machines, jobs

def scheduleJob(jobs, machines, machineID, jobID, time_, RELEASE_ACTION):
#     since it is checked that machineID and jobID are empty at time
    print("{} required {} at {}".format(jobs[jobID].jobName, machines[machineID].machineName, str(time_)))
    jobs[jobID].getProcessed()
    machines[machineID].processJob(jobID, time_)
    RELEASE_ACTION.append({'MachineId':machineID, 'JobID':jobID, 'EntryTime':machines[machineID].now, \
                           'ReleaseTime':time_, 'ProcessTime':machines[machineID].processTime})


def release(jobs, machines, machineID, jobID, time_, JOBS_PROCESSED):
    print("{} released {} at {}".format(jobs[jobID].jobName, machines[machineID].machineName, str(time_)))
    machines[machineID].releaseMachine()
    jobs[jobID].releaseJob()
    JOBS_PROCESSED[machineID, jobID] = 1

def runAEpisode(param):
    machines, jobs = resetStates(param['NUM_MACHINES'], param['NUM_JOBS'])

    for time in range(param['SIMULATION_TIME']):
        totalJobsDone = np.sum(param['JOBS_PROCESSED'])
        if totalJobsDone == (param['NUM_JOBS'] * param['NUM_MACHINES']):
            print("Simulation Over")
            break

        emptyMachines = [i for i in range(param['NUM_MACHINES']) if machines[i].machineBusy is False]
        for i, eM in enumerate(emptyMachines):
            print("For Machine %d"%eM)
            emptyJobs = [i for i in range(param['NUM_JOBS']) if jobs[i].jobBusy is False]
            if emptyJobs is None:
                print("Chill Maar !!")
                continue
            if np.sum(param['JOBS_PROCESSED'][eM]) == param['NUM_JOBS']:
                print("Machine {} done enough".format(str(eM)))
                continue
            print(emptyJobs)
            while(True):
                jobID = int(input("Give jobID for following job available\n"))
                if jobID not in emptyJobs:
                    print("What are you doing bruh??")
                elif param['JOBS_PROCESSED'][eM,jobID] == 1:
                    print("Jobs already done")
                else:
                    break
            scheduleJob(jobs, machines, eM, jobID, time, param['RELEASE_ACTION'])
    
        busyMachines = [i for i in range(param['NUM_MACHINES']) if machines[i].machineBusy is True]
        for i, bM in enumerate(busyMachines):
            jOT = machines[bM].jobOverTime
            if time == jOT:
                jobID = machines[bM].onJob
                release(jobs, machines, bM, jobID, time, param['JOBS_PROCESSED'])

def main():

    param = {
    'NUM_MACHINES' : 2,
    'NUM_JOBS' : 4,
    'PROCESSING_TIME' : [(3,2),(6,7),(9,4),(4,6)],
    'ORDER_OF_PROCESSING' : [(1,2),(2,1),(1,2),(1,2)],
    'JOBS_PROCESSED' : np.zeros((2,4), dtype=np.int8),
    'SCHEDULE_ACTION' : [],
    'RELEASE_ACTION' : [],
    'SIMULATION_TIME' : 1000
    }
    runAEpisode(param)
    print(param['RELEASE_ACTION'])
    
if __name__ == "__main__":
    main()