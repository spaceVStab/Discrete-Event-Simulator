import numpy as np 
from model import Machine, Jobs
import warnings

class Environment(object):
	def __init__(self, config):
		self.config = config
		# self.reset()
		# assert the config dict for required parameters


	def reset(self):
		NUM_MACHINES = self.config['NUM_MACHINES']
		NUM_JOBS = self.config['NUM_JOBS']
		machines = [Machine(i) for i in range(NUM_MACHINES)]
		jobs = [Jobs(i) for i in range(NUM_JOBS)]
		self.machines = machines
		self.jobs = jobs
		self.NOW = 0
		self.SIM_LEFT = self.config['SIMULATION_TIME'] - self.NOW

		# should return an obs : containing the information of machines empty and jobs empty
		return self.returnObs()


	def getEmptyMachines(self):
		emptyMachines = [i for i in range(self.config['NUM_MACHINES']) if self.machines[i].machineBusy is False]
		return emptyMachines

	def getBusyMachines(self):
		busyMachines = [i for i in range(self.config['NUM_MACHINES']) if self.machines[i].machineBusy is True]
		return busyMachines

	def getEmptyJobs(self):
		emptyJobs = [i for i in range(self.config['NUM_JOBS']) if self.jobs[i].jobBusy is False]
		return emptyJobs

	def getValidJobs(self, machinesID):
		totalJobs = self.getEmptyJobs()
		jobsDone = self.machines[machinesID].jobsDone
		validJobs = list(set(totalJobs) - set(jobsDone))

		if validJobs is None:
			# machine wont schedule now
			return -1
		elif np.sum(self.config['JOBS_PROCESSED'][machinesID]) == self.config['NUM_JOBS']:
			# machine has done scheduling all the jobs
			return -1
		else:
			return validJobs

	def returnObs(self):
		obs = {}
		for i in range(self.config['NUM_MACHINES']):
			if self.machines[i].machineBusy is True:
				onJob = self.machines[i].onJob
				obs[self.machines[i].machineID] = int(onJob)
			if self.machines[i].machineBusy is False:
				# if the jobs is not processing and not busy pass -1
				obs[self.machines[i].machineID] = -1
		return obs

	def returnQuadInfo(self, done=False):
		"""
		returns the information in the format (state, reward, done, info)
		"""
		obs = self.returnObs()
		# to be modified
		reward = self.calcMetrics()
		info = "Nothing"
		return (obs, reward, done, info)

	def checkCompletion(self):
		totalJobsDone = np.sum(self.config['JOBS_PROCESSED'])
		if totalJobsDone == (self.config['NUM_JOBS'] * self.config['NUM_MACHINES']):
			return True
		else:
			return False
			

	def takeAction(self, action):
		"""
		iterate time from now to sim time
		check if simulation is completed
		schedule jobs as per the actions at that time step / action = NULL
		release the jobs that need to be released
		check if any machine is free then trigger the return for the state and request for actions from the users
		"""
		if self.checkCompletion is True:
			print("Simulation Over")
			# return the observation pattern
			return self.returnQuadInfo(done=True)

		actionDict = {i:a for i,a in enumerate(action)}

		# action length is equal to the number of machines free rn.
		for time in range(self.NOW, self.config['SIMULATION_TIME']):
			# calculate the total jobs done and if saturated complete the episode
			self.NOW = time

			# check if this is first iteration relatively
			if actionDict is not None:
				for i, macID in enumerate(actionDict):
					if actionDict[macID] == -1:
						continue
					self.scheduleJob(macID, actionDict[macID], self.NOW)

			actionDict = None

			busyMac = self.getBusyMachines()
			for i, bM in enumerate(busyMac):
				jOT = self.machines[bM].jobOverTime
				if self.NOW == jOT:
					jobID = self.machines[bM].onJob
					self.release(bM, jobID, self.NOW)

			emptyMac = self.getEmptyMachines()
			if len(emptyMac) is not 0:
				# check if releasing a job completes the simulation
				if self.checkCompletion() is True:
					print("Simulation Over")
					quadInfo = self.returnQuadInfo(done=True)
					return quadInfo
				else:
					print("Empty machines found, go back to the agent")
					quadInfo = self.returnQuadInfo(done=False)
					self.NOW += 1
					return quadInfo
			
		print("Simulation time is over")
		return -1

	def step(self, action):
		# get the number of machines available assert if the action is suitable
		# emptyMachines = self.getEmptyMachines()
		# import pdb; pdb.set_trace()
		# assert len(action) == len(emptyMachines)
		# put the action -1 if the machine is not to be scheduled
		quad = self.takeAction(action)
		if quad == -1:
			print("Simulation maximum time is over")
			return None
		else: 
			return quad

	def scheduleJob(self, machineID, jobID, time_):
		#     since it is checked that machineID and jobID are empty at time
	    print("{} required {} at {}".format(self.jobs[jobID].jobName, self.machines[machineID].machineName, \
	    	str(time_)))
	    self.jobs[jobID].getProcessed()
	    self.machines[machineID].processJob(jobID, time_)
	    self.config['RELEASE_ACTION'].append({'MachineId':machineID, 'JobID':jobID, \
	    	'EntryTime':self.machines[machineID].now, 'ReleaseTime':self.machines[machineID].jobOverTime, \
	    	'ProcessTime':self.machines[machineID].processTime})

	def release(self, machineID, jobID, time_):
	    print("{} released {} at {}".format(self.jobs[jobID].jobName, self.machines[machineID].machineName, str(time_)))
	    self.machines[machineID].releaseMachine()
	    self.jobs[jobID].releaseJob()
	    self.config['JOBS_PROCESSED'][machineID, jobID] = 1

	def printStatus(self):
	    print("Machines if occupied")
	    for i in range(len(self.machines)):
	        print("{} : {}".format(self.machines[i].machineName, str(self.machines[i].machineBusy)))
	    print("Jobs if processed")
	    for j in range(len(self.jobs)):
	        print(self.jobs[j].jobName)
	        print(self.jobs[j].processDetails)    

	def getProcessingTime(self, MachineID, JobID):
	    return self.config['PROCESSING_TIME'][JobID][MachineID]

	def calcMetrics(self):
	#     calculate throughput, 
	    totalJobsDone = np.sum(self.config['JOBS_PROCESSED'])
	    throughput = float(totalJobsDone / self.NOW)
	    return throughput









