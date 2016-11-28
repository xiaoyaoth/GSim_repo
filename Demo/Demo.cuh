#include "gsimcore.cuh"
#include "gsimvisual.cuh"

class DemoAgent;

struct DemoAgentData : public GAgentData
{
};

struct DataUnion
{
	union {
		DemoAgentData a;
	};
	__device__ void putDataInSmem(GAgent *ag);

};

__global__ void addAgents(AgentPool<DemoAgent, DemoAgentData> *pool);

class DemoModel : public GModel
{
	AgentPool<DemoAgent, DemoAgentData> *pool, *poolHost;

public:

	DemoModel() {
		int numElem;
		int numElemMax;

		poolHost = new AgentPool<DemoAgent, DemoAgentData> ( numElem, numElemMax, sizeof(DataUnion));
		util::hostAllocCopyToDevice< AgentPool<DemoAgent, DemoAgentData> > (poolHost, &pool);
	}

	void start() {
		int gSize = GRID_SIZE(poolHost->numElem);
		addAgents<<<gSize, BLOCK_SIZE>>>(this->pool);
		GSimVisual::getInstance().setWorld(this->world);
	}

	void preStep() {
		this->poolHost->registerPool(worldHost, NULL, this->pool);
		//cudaMemcpyToSymbol(modelDevParams, &modelHostParams, sizeof(modelConstants));
	}

	void step() {
		this->poolHost->stepPoolAgent(this->model, 0);
		GSimVisual::getInstance().animate();
	}

	void stop() {
		GSimVisual::getInstance().stop();
	}
};

class DemoAgent : public GAgent
{
public:
	__device__ void step(GModel *model) {
		GWorld *world = model->world;
		iterInfo info;
		DataUnion *elem = world->nextAgentDataFromSharedMem<DataUnion>(info);
	}
};

__global__ void addAgents(AgentPool<DemoAgent, DemoAgentData> *pool)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < pool->numElem) {
		int agentSlot = idx;
		int dataSlot = pool->dataSlot(agentSlot);
		DemoAgent *agent = &pool->agentArray[dataSlot];
		pool->add(agent, agentSlot);
	}
}
__device__ void putDataInSmem(GAgent *ag){

}

