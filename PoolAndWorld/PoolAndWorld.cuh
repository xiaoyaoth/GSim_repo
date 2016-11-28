#ifndef BOID_CUH
#define BOID_CUH

#include "gsimcore.cuh"
#ifdef _WIN32
#include "gsimvisual.cuh"
#endif

enum BOID_TYPE {BOID_PREY, BOID_PREDATOR};

struct BoidAgentData : public GAgentData
{
	FLOATn lastd;
};

struct PreyAgentData : public BoidAgentData
{
	__device__ void putDataInSmem(GAgent *ag);
};


class BoidModel;
class PreyAgent;

__global__ void addAgents(BoidModel *bModel);

__constant__ int N_HAHA_PREY;
int MAX_N_HAHA_PREY;

__constant__ float range;

class BoidModel : public GModel{
public:
	GRandom *random, *randomHost;

	AgentPool<PreyAgent, PreyAgentData> *pool, *poolHost;
	
	cudaEvent_t timerStart, timerStop;
	
	__host__ BoidModel(int numPrey, float range_h)
	{
		MAX_N_HAHA_PREY = numPrey*2;
		cudaMemcpyToSymbol(N_HAHA_PREY, &numPrey, sizeof(int));

		poolHost = new AgentPool<PreyAgent, PreyAgentData>(numPrey, MAX_N_HAHA_PREY, sizeof(PreyAgentData));
		util::hostAllocCopyToDevice<AgentPool<PreyAgent, PreyAgentData>>(poolHost, &pool);

		worldHost = new GWorld(modelHostParams.WIDTH, modelHostParams.HEIGHT);
		util::hostAllocCopyToDevice<GWorld>(worldHost, &world);

		randomHost = new GRandom(modelHostParams.MAX_AGENT_NO);
		util::hostAllocCopyToDevice<GRandom>(randomHost, &random);

		util::hostAllocCopyToDevice<BoidModel>(this, (BoidModel**)&this->model);

		cudaMemcpyToSymbol(range, &range_h, sizeof(int));
	}

	__host__ void start()
	{
		int AGENT_NO = this->poolHost->numElem;
		int gSize = GRID_SIZE(AGENT_NO);
		addAgents<<<gSize, BLOCK_SIZE>>>((BoidModel*)this->model);
		GSimVisual::getInstance().setWorld(this->world);
		cudaEventCreate(&timerStart);
		cudaEventCreate(&timerStop);
		cudaEventRecord(timerStart, 0);
	}

	__host__ void registerPool() 
	{
		this->poolHost->registerPool(this->worldHost, this->schedulerHost, this->pool);
		cudaMemcpyToSymbol(modelDevParams, &modelHostParams, sizeof(modelConstants));
	}

	__host__ void preStep()
	{
		registerPool();
		GSimVisual::getInstance().animate();
	}

	__host__ void step()
	{
		int gSize = GRID_SIZE(this->poolHost->numElem);
		int numStepped = 0;

		numStepped += this->poolHost->stepPoolAgent(this->model, numStepped);
	}

	__host__ void stop()
	{
		float time;
		cudaDeviceSynchronize();
		cudaEventRecord(timerStop, 0);
		cudaEventSynchronize(timerStop);
		cudaEventElapsedTime(&time, timerStart, timerStop);
		std::cout<<time<<std::endl;

		GSimVisual::getInstance().stop();
	}
};

class PreyAgent : public GAgent
{
public:
	BoidModel *model;
	GRandom *random;
	AgentPool<PreyAgent, PreyAgentData> *pool;

	__device__ void init(BoidModel *bModel, int dataSlot)
	{
		this->model = bModel;
		this->random = bModel->random;
		this->pool = bModel->pool;
		this->color = colorConfigs.green;

		PreyAgentData *myData = &this->pool->dataArray[dataSlot];
		PreyAgentData *myDataCopy = &this->pool->dataCopyArray[dataSlot];
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		int dim = (int)sqrtf(pool->numElem);
		myData->loc.x = idx % dim;
		myData->loc.y = idx / dim;
		myData->lastd.x = 0;
		myData->lastd.y = 0;
		*myDataCopy = *myData;

		this->data = myData;
		this->dataCopy = myDataCopy;
	}

	__device__ void step(GModel *model)
	{
		BoidModel *boidModel = (BoidModel*) model;
		GWorld *world = boidModel->world;
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		iterInfo info;
		PreyAgentData myData = *(PreyAgentData*)this->data;
		world->neighborQueryInit(myData.loc, range, info);
		
		int counter = 0;

		PreyAgentData *elem = (PreyAgentData*)world->nextAgentData(info);
		//PreyAgentData *elem = world->nextAgentDataFromSharedMem<PreyAgentData>(info);
		while(elem) {
			PreyAgentData otherData = *elem;
			float ds = length(otherData.loc - myData.loc);
			if(ds <= range)
				counter++;
			elem = (PreyAgentData*)world->nextAgentData(info);
			//elem = world->nextAgentDataFromSharedMem<PreyAgentData>(info);

		}
	}

};

__global__ void addAgents(BoidModel *model)
{
	uint idx = threadIdx.x + blockIdx.x * blockDim.x;
	int dataSlot = -1;
	if (idx < N_HAHA_PREY) {
		dataSlot = idx;
		PreyAgent *ag = &model->pool->agentArray[dataSlot];
		ag->init(model, dataSlot);
		model->pool->add(ag, dataSlot);
	}
}

__device__ void PreyAgentData::putDataInSmem(GAgent *ag){
	*this = *(PreyAgentData*)ag->data;
}

#endif