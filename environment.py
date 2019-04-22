import numpy as np 
from model import Machine, Jobs
import warnings
import random
from collections import defaultdict, OrderedDict, namedtuple
import itertools


class Environment(object):
    def __init__(self, config):
        self.config = config
        self.ProcessTime = np.zeros((self.config['NUM_MACHINES'], self.config['NUM_JOBS']))
        
    def reset(self):
        NUM_MACHINES = self.config['NUM_MACHINES']
        NUM_JOBS = self.config['NUM_JOBS']
        self.machines = [Machine(i) for i in range(NUM_MACHINES)]
        self.jobs = [Jobs(i) for i in range(NUM_JOBS)]
        self.NOW = 0
        self.jobs_processed = np.zeros((self.config['NUM_MACHINES'],self.config['NUM_JOBS']), dtype=np.int8)
        return self.returnObs()

    def getProcessTime(self, macID, jID):
        # import pdb; pdb.set_trace()
        if self.ProcessTime[macID, jID] == 0:
            t_time = int(random.normalvariate(15, 5))
            if t_time < 1:
                t_time = 1
            self.ProcessTime[macID, jID] = t_time
            with open('processtime_25_100.txt','a') as fp:
                fp.write("machine {} job {} time {}\n".format(str(macID), str(jID), str(self.ProcessTime[macID, jID])))
        return self.ProcessTime[macID, jID]

    def returnObs(self):
        obs = {}
        for i in range(self.config['NUM_MACHINES']):
            if self.machines[i].machineBusy is True:
                onJob = self.machines[i].onJob
                obs[self.machines[i].machineID] = int(onJob)
            else:
                obs[self.machines[i].machineID] = -1
        return obs

    def returnQuadInfo(self, done=False):
        obs = self.returnObs()
        if self.NOW == 0:
            reward = 0
        else:
            reward = self.calcMetrics()
        info = "nothing"
        return (obs, reward, done, info)

    def calcMetrics(self):
        totalJobsDone = np.sum(self.jobs_processed)
        throughput = np.divide(np.int(totalJobsDone), np.int(self.NOW))
        return throughput

    def getEmptyMachines(self):
        return [i for i in range(self.config['NUM_MACHINES']) if self.machines[i].machineBusy is False]

    def getBusyMachines(self):
        return [i for i in range(self.config['NUM_MACHINES']) if self.machines[i].machineBusy is True]

    def getEmptyJobs(self):
        return [i for i in range(self.config['NUM_JOBS']) if self.jobs[i].jobBusy is False]
        
    def getBusyJobs(self):
        return [i for i in range(self.config['NUM_JOBS']) if self.jobs[i].jobBusy is True]

    def checkCompletion(self):
        totalJobsDone = int(np.sum(self.jobs_processed))
        toDo = int(self.config['NUM_JOBS'] * self.config['NUM_MACHINES'])
        if totalJobsDone == toDo:
            return True 
        else:
            return False 

    def takeAction(self, action):
        if self.checkCompletion() is True:
            print("Simulation over")
            return self.returnQuadInfo(done=True)

        # machine:job
        actionDict = {i:a for i,a in enumerate(action)}
        
        while(True):
            # self.NOW = time

            for i, macID in enumerate(actionDict):
                if (actionDict[macID] != -1):
                    self.scheduleJob(macID, actionDict[macID], self.NOW)
            
            actionDict = {}

            # find the busy machines now
            busyMac = self.getBusyMachines()
            for i, bM in enumerate(busyMac):
                jOT = self.machines[bM].jobOverTime
                if int(self.NOW) == int(jOT):
                    jobID = self.machines[bM].onJob
                    self.release(bM, jobID, self.NOW)

            if self.checkCompletion() is True:
                print("Simulation over")
                return self.returnQuadInfo(done=True)

            # now check if there are empty machines
            emptyMac = self.getEmptyMachines()
            # since there are empty mac request for review
            if (len(emptyMac) != 0):
                self.NOW += 1
                return self.returnQuadInfo(done=False)

            self.NOW += 1

            if self.NOW > self.config['SIMULATION_TIME']:
                print("simulation over")
                break
        return self.returnQuadInfo(done=True)


    def step(self, action):
        quad = self.takeAction(action)
        return quad
    
    def scheduleJob(self, machineID, jobID, time_):
        self.jobs[jobID].getProcessed()
        p_time = self.getProcessTime(machineID, jobID)
        # print(time_, p_time)
        self.machines[machineID].processJob(jobID, time_, p_time)
        self.config['SCHEDULE_ACTION'].append({'MachineID':machineID, 'JobId':jobID, 'scheduledAt':time_, 'ProcessTime':p_time})
        return

    def release(self, machineID, jobID, time_):
        # print("hi")
        assert self.machines[machineID].jobOverTime == time_
        self.machines[machineID].releaseMachine()
        self.jobs[jobID].releaseJob()
        self.jobs_processed[machineID, jobID] = 1
        self.config['RELEASE_ACTION'].append({'MachineID':machineID, 'JobId':jobID, 'releasedAt':time_})
        return
    
    def getValidJobs(self, machineID):
        # since for processing order check if busy
        if self.machines[machineID].machineBusy is True:
            return -1
        else:
            valid_jobs = []
            # first get empty jobs
            empJobs = self.getEmptyJobs()
            # print(empJobs)
            for i,j in enumerate(empJobs):
                j_done_len = self.jobs[j].machineVisited
                if j_done_len < self.config['NUM_MACHINES']:
                    # import pdb; pdb.set_trace()
                    # print(j, j_done_len)
                    toDo = self.config['ORDER_OF_PROCESSING'][j][j_done_len]
                    if toDo == machineID:
                        valid_jobs.append(j)
            if len(valid_jobs) == 0:
                valid_jobs = -1
        return valid_jobs


            # done = len(self.machines[machineID].jobsDone)
            # if done == self.config['NUM_JOBS']:
            #     return -1
            # else:
            #     tempJob = self.config['ORDER_OF_PROCESSING'][machineID][done]
            #     if self.jobs[tempJob].jobBusy is True:
            #         return -1
            #     else:
            #         return tempJob


    def generatePossibleAction(self, obs):
        possibleAction = []     # list of possible action to take 
        state_info = {}     # state information
        # import pdb; pdb.set_trace()
        for machine, status in obs.items():
            if status == -1:
                valJob = self.getValidJobs(machine)
                # import pdb; pdb.set_trace()
                if valJob == -1:
                    possibleAction.append({machine:-1})
                else:
                    if not isinstance(valJob, list):
                        valJob = [valJob]
                    valJob += [-1]
                    possibleAction.append({machine:valJob})
                state_info[machine] = tuple(self.machines[machine].jobsDone)
            else:
                possibleAction.append({machine:-1})
                state_info[machine] = tuple(self.machines[machine].jobsDone + [status])
        # import pdb; pdb.set_trace()
        permuteAct = []
        machines = []
        for act in possibleAction:
            t_ = list(act.values())[0]
            if not isinstance(t_, list):
                t_ = [t_]
            permuteAct.append(t_)
            machines.append(list(act.keys())[0])

        totalAct = itertools.product(*permuteAct)

        actions = []

        for act in totalAct:
            temp = {}
            act_ = list(act)
            act_ = [a_ for a_ in act_ if a_!=-1]
            # import pdb; pdb.set_trace()
            if len(set(list(act_))) == len(list(act_)):
                for i,a_ in enumerate(act):
                    temp[machines[i]] = a_
                actions.append(temp)
            else:
                continue

        state_info = tuple(OrderedDict(sorted(state_info.items())).values())
        tempAct = []
        for a in actions:
            tempAct.append(tuple(OrderedDict(sorted(a.items())).values()))
        if tempAct == []:
            tempAct = [tuple([-1]*self.config['NUM_MACHINES'])]
        return state_info, tempAct