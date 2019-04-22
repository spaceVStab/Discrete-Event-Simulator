import numpy as np 
import os 
import random 

"""
Machines are created here
Whenever you use the machine release it
"""
class Machine(object):
    def __init__(self, name):
        self.machineID = int(name)
        self.machineName = "Machine_"+str(name)
        self.machineBusy = False
        self.jobOverTime = 0
        self.now = 0
        self.jobsDone = []

    def processJob(self, jobID, time, pTime):
        # check if machine is busy or not
        assert self.machineBusy == False
        self.onJob = jobID
        self.now = time
        self.processTime = pTime
        # import pdb; pdb.set_trace()
        self.jobOverTime = self.now + self.processTime
        self.machineBusy = True
        return

    def releaseMachine(self):
        # check if currently in use
        assert self.machineBusy == True
        self.jobsDone.append(self.onJob)
        self.machineBusy = False
        return

"""
Jobs are created here
Always use the job then release it too
"""
class Jobs(object):
    def __init__(self, name):
        self.jobID = name
        self.jobName = "Job_"+str(name)
        self.jobBusy = False
        self.processDetails = []
        self.noOfProcess = 0
        self.now = 0
        self.machineVisited = 0

    def getProcessed(self):
        # print("{} requested".format(self.jobName))
        assert self.jobBusy == False
        self.jobBusy = True
        return

    def releaseJob(self):
        # print("{} released".format(self.jobName))
        assert self.jobBusy == True
        self.machineVisited += 1
        self.jobBusy = False
        return