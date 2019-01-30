import numpy as np 
import os 
import random 

PROCESSING_TIME = [(3,2),(6,7),(9,4),(4,6)]

def getProcessingTime(MachineID, JobID):
    # return random.normalvariate(mean, sigma)
    return PROCESSING_TIME[JobID][MachineID]

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
    
    def processJob(self, jobID, time):
#         check if machine is busy or not
        if self.machineBusy is False:
            self.onJob = jobID
            self.now = time
            self.processTime = getProcessingTime(self.machineID, jobID)
            self.jobOverTime = self.now + self.processTime
            self.machineBusy = True
            return 
        else:
            print("Current Machine busy")
            self.machineBusy = False
        return
    
    def releaseMachine(self):
#         check if currently in use
        if self.machineBusy is True:
            self.machineBusy = False
        else:
            print("Machine was not is use")
        return
            
    def syncTime(self, delTime):
        self.now += delTime
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
        
    def getProcessed(self):
        print("{} requested".format(self.jobName))
        self.jobBusy = True
        return
    
    def releaseJob(self):
        print("{} released".format(self.jobName))
        self.jobBusy = False
        return
    
    def addDetails(self, MId, delTime, startTime):
        self.noOfProcess += 1
        self.processDetails.append((MId, delTime, startTime))
        return

