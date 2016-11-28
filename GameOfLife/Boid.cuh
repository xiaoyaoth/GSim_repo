#ifndef BOID_CUH
#define BOID_CUH

#include "gsimcore.cuh"
#ifdef _WIN32
#include "gsimvisual.cuh"
#endif

enum GOLAgentState {dead, live};

struct GOLAgentData : public GAgentData
{
	GOLAgentState state;
	__device__ void putDataInSmem(GAgent *ag);
};

#define NUM_DOT 35
__constant__ int2 liveDots[NUM_DOT];

class GOLModel;
class GOLAgent;

class ConflictResolver
{

};

__global__ void addAgents(GOLModel *golModel);
class GOLModel : public GModel
{
public:
	AgentPool<GOLAgent, GOLAgentData> *agentPool, *agentPoolHost;
	cudaEvent_t timerStart, timerStop;

	__host__ GOLModel(int numAgent)
	{
		agentPoolHost = new AgentPool<GOLAgent, GOLAgentData>(numAgent, numAgent, sizeof(GOLAgentData));
		util::hostAllocCopyToDevice<AgentPool<GOLAgent, GOLAgentData>>(agentPoolHost, &agentPool);

		//TODO: fix world initializer
		worldHost = new GWorld(modelHostParams.WIDTH, modelHostParams.HEIGHT);
		util::hostAllocCopyToDevice<GWorld>(worldHost, &world);

		util::hostAllocCopyToDevice<GOLModel>(this, (GOLModel**)&this->model);

		int2 liveDotsHost[NUM_DOT];
		int yDisp = 3;
		int i = 0; 
			 liveDotsHost[i] = make_int2(0, 2+yDisp);
		i++; liveDotsHost[i] = make_int2(0, 3+yDisp);
		i++; liveDotsHost[i] = make_int2(1, 2+yDisp);
		i++; liveDotsHost[i] = make_int2(1, 3+yDisp);
		i++; liveDotsHost[i] = make_int2(8, 3+yDisp);
		i++; liveDotsHost[i] = make_int2(8, 4+yDisp);
		i++; liveDotsHost[i] = make_int2(9, 2+yDisp);
		i++; liveDotsHost[i] = make_int2(9, 4+yDisp);
		i++; liveDotsHost[i] = make_int2(10, 2+yDisp);
		i++; liveDotsHost[i] = make_int2(10, 3+yDisp);
		i++; liveDotsHost[i] = make_int2(16, 4+yDisp);
		i++; liveDotsHost[i] = make_int2(16, 5+yDisp);
		i++; liveDotsHost[i] = make_int2(16, 6+yDisp);
		i++; liveDotsHost[i] = make_int2(17, 4+yDisp);
		i++; liveDotsHost[i] = make_int2(18, 5+yDisp);
		i++; liveDotsHost[i] = make_int2(22, 1+yDisp);
		i++; liveDotsHost[i] = make_int2(22, 2+yDisp);
		i++; liveDotsHost[i] = make_int2(23, 0+yDisp);
		i++; liveDotsHost[i] = make_int2(23, 2+yDisp);
		i++; liveDotsHost[i] = make_int2(24, 0+yDisp);
		i++; liveDotsHost[i] = make_int2(24, 1+yDisp);
		i++; liveDotsHost[i] = make_int2(24, 12+yDisp);
		i++; liveDotsHost[i] = make_int2(24, 13+yDisp);
		i++; liveDotsHost[i] = make_int2(25, 12+yDisp);
		i++; liveDotsHost[i] = make_int2(25, 14+yDisp);
		i++; liveDotsHost[i] = make_int2(26, 12+yDisp);
		i++; liveDotsHost[i] = make_int2(34, 0+yDisp);
		i++; liveDotsHost[i] = make_int2(34, 1+yDisp);
		i++; liveDotsHost[i] = make_int2(35, 0+yDisp);
		i++; liveDotsHost[i] = make_int2(35, 1+yDisp);
		i++; liveDotsHost[i] = make_int2(35, 7+yDisp);
		i++; liveDotsHost[i] = make_int2(35, 8+yDisp);
		i++; liveDotsHost[i] = make_int2(35, 9+yDisp);
		i++; liveDotsHost[i] = make_int2(36, 7+yDisp);
		i++; liveDotsHost[i] = make_int2(37, 8+yDisp);

		int2 liveDotsHost2[5];
		liveDotsHost2[0] = make_int2(2, 30-2);
		liveDotsHost2[1] = make_int2(2, 31-2);
		liveDotsHost2[2] = make_int2(2, 32-2);
		liveDotsHost2[3] = make_int2(1, 30-2);
		liveDotsHost2[4] = make_int2(0, 31-2);
		
		cudaMemcpyToSymbol(liveDots, &liveDotsHost, NUM_DOT * sizeof(int2));
	}

	__host__ void start()
	{
		int gSize = GRID_SIZE(this->agentPoolHost->numElem);
		addAgents<<<gSize, BLOCK_SIZE>>>((GOLModel*)this->model);
#ifdef _WIN32
		GSimVisual::getInstance().setWorld(this->world);
#endif
		cudaEventCreate(&timerStart);
		cudaEventCreate(&timerStop);
		cudaEventRecord(timerStart, 0);
	}

	__host__ void preStep()
	{
		this->agentPoolHost->registerPool(this->worldHost, this->schedulerHost, this->agentPool);
		cudaMemcpyToSymbol(modelDevParams, &modelHostParams, sizeof(modelConstants));
#ifdef _WIN32
		GSimVisual::getInstance().animate();
#endif
	}

	__host__ void step()
	{
		int numStepped = 0;
		//TODO: numStepped is not used? could be replaced by AGENT_NO
		numStepped += this->agentPoolHost->stepPoolAgent(this->model, numStepped);
	}

	__host__ void stop()
	{
		float time;
		cudaDeviceSynchronize();
		cudaEventRecord(timerStop, 0);
		cudaEventSynchronize(timerStop);
		cudaEventElapsedTime(&time, timerStart, timerStop);
		std::cout<<time<<std::endl;
#ifdef _WIN32
		GSimVisual::getInstance().stop();
#endif
	}
};

class GOLAgent : public GAgent
{
public:
	__device__ void init(GOLModel *model, GOLAgentState state, int dataSlot, float2 loc)
	{
		GOLAgentData *myData = &model->agentPool->dataArray[dataSlot];
		GOLAgentData *myDataCopy = &model->agentPool->dataCopyArray[dataSlot];
		myData->loc = loc;
		myData->state = state;
		if (state == live)
			this->color = colorConfigs.green;
		else 
			this->color = colorConfigs.blue;

		*myDataCopy = *myData;
		this->data = myData;
		this->dataCopy = myDataCopy;
	}

	__device__ void step(GModel *model)
	{
		GOLModel *golModel = (GOLModel *)model;
		const GWorld *world = golModel->world;
		GOLAgentData myData = *(GOLAgentData*)this->data;
		iterInfo info;

		world->neighborQueryInit(myData.loc, 2, info);
		GOLAgentData *elem = world->nextAgentDataFromSharedMem<GOLAgentData>(info);
		while(elem != NULL) {
			GOLAgentData otherData = *elem;
			int2 locDiff = make_int2(otherData.loc - myData.loc);
			locDiff = abs(locDiff);
			bool validNeighbor = locDiff.x <= 1 && locDiff.y <= 1 && otherData.state == live;
			validNeighbor = validNeighbor && locDiff.x + locDiff.y != 0;
			if (validNeighbor)
				info.count++;
			elem = world->nextAgentDataFromSharedMem<GOLAgentData>(info);
		}
		if (info.count == 3) {
			myData.state = live;
			this->color = colorConfigs.green;
		}
		if (info.count < 2 || info.count > 3) {
			this->color = colorConfigs.blue;
			myData.state = dead;
		}

		*(GOLAgentData*)this->dataCopy = myData;
	}

	__device__ void setDataInSmem(void *elem)
	{
		*(GOLAgentData*)elem = *(GOLAgentData*)this->data;
	}

};

__global__ void addAgents(GOLModel *golModel)
{
	uint idx = threadIdx.x + blockIdx.x * blockDim.x;
	int dataSlot = -1;
	if (idx < golModel->agentPool->numElem) {
		dataSlot = idx;
		int2 locInt = make_int2(0, 0);
		int numPerDim = (int)golModel->world->width;
		locInt.x = idx % numPerDim;
		locInt.y = idx / numPerDim;
		float2 loc = make_float2(locInt.x, locInt.y);
		GOLAgentState state = dead;
		for (int i = 0; i < NUM_DOT; i++)
			if (locInt.x == liveDots[i].x && locInt.y == liveDots[i].y)
				state = live;
		//GOLAgent *ag = new GOLAgent(golModel, state, dataSlot, loc);
		GOLAgent *ag = &golModel->agentPool->agentArray[dataSlot];
		ag->init(golModel, state, dataSlot, loc);
		golModel->agentPool->add(ag, dataSlot);
	}
}

__device__ void GOLAgentData::putDataInSmem(GAgent *ag)
{
	*this = *(GOLAgentData*)ag->data;
}

#endif