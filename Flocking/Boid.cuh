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

struct SimParams
{
	float cohesion;
	float avoidance;
	float randomness;
	float consistency;
	float momentum;
	float deadFlockerProbability;
	float neighborhood;
	float jump;

	float maxForce;
	float maxForcePredator;
};

__constant__ SimParams params;

class BoidModel;
class PreyAgent;

__global__ void addAgents(BoidModel *bModel);

#define N_POOL 2
__constant__ int N_HAHA_PREY;
int MAX_N_HAHA_PREY;
__constant__ int N_HAHA_PREDATOR;
int MAX_N_HAHA_PREDAOTR;

class BoidModel : public GModel{
public:
	GRandom *random, *randomHost;

	AgentPool<PreyAgent, PreyAgentData> *pool, *poolHost;

	cudaEvent_t timerStart, timerStop;

	__host__ BoidModel(float range, int numPrey, int numPred)
	{
		MAX_N_HAHA_PREY = numPrey * 2;
		cudaMemcpyToSymbol(N_HAHA_PREY, &numPrey, sizeof(int));

		poolHost = new AgentPool<PreyAgent, PreyAgentData>(numPrey, MAX_N_HAHA_PREY, sizeof(PreyAgentData));
		util::hostAllocCopyToDevice< AgentPool<PreyAgent, PreyAgentData> >(poolHost, &pool);

		worldHost = new GWorld(modelHostParams.WIDTH, modelHostParams.HEIGHT);
		util::hostAllocCopyToDevice<GWorld>(worldHost, &world);

		randomHost = new GRandom(modelHostParams.MAX_AGENT_NO);
		util::hostAllocCopyToDevice<GRandom>(randomHost, &random);

		util::hostAllocCopyToDevice<BoidModel>(this, (BoidModel**)&this->model);

		SimParams paramHost;
		paramHost.cohesion = 1.0;
		paramHost.avoidance = 1.0;
		paramHost.randomness = 1.0;
		paramHost.consistency = 10.0;
		paramHost.momentum = 1.0;
		paramHost.deadFlockerProbability = 0.1;
		paramHost.neighborhood = range;
		paramHost.jump = 0.7;
		paramHost.maxForce = 6;
		paramHost.maxForcePredator = 10;

		cudaMemcpyToSymbol(params, &paramHost, sizeof(SimParams));
	}

	__host__ void start()
	{
		int AGENT_NO = this->poolHost->numElem;
		int gSize = GRID_SIZE(AGENT_NO);
		addAgents<<<gSize, BLOCK_SIZE>>>((BoidModel*)this->model);
#ifdef _WIN32
		GSimVisual::getInstance().setWorld(this->world);
#endif
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
#ifdef _WIN32
		GSimVisual::getInstance().animate();
#endif
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
#ifdef _WIN32
		GSimVisual::getInstance().stop();
#endif
	}
};


class PreyAgent : public GAgent
{
public:
	GRandom *random;

	__device__ void init(BoidModel *bModel, int dataSlot)
	{
		this->random = bModel->random;
		this->color = colorConfigs.black;

		PreyAgentData *myData = &bModel->pool->dataArray[dataSlot];
		PreyAgentData *myDataCopy = &bModel->pool->dataCopyArray[dataSlot];
		myData->loc.x = random->uniform() * modelDevParams.WIDTH;
		myData->loc.y = random->uniform() * modelDevParams.HEIGHT;
		myData->lastd.x = 0;
		myData->lastd.y = 0;
		*myDataCopy = *myData;

		this->data = myData;
		this->dataCopy = myDataCopy;
	}

	__device__ FLOATn consistency(const GWorld *world, iterInfo &info)
	{
		FLOATn res = make_float2(0,0);
		float ds;
		FLOATn m;
		PreyAgentData myData = *(PreyAgentData*)this->data;
		world->neighborQueryInit(myData.loc, params.neighborhood, info);
		PreyAgentData otherData;
		PreyAgentData *elem = world->nextAgentDataFromSharedMem<PreyAgentData>(info);
		while(elem != NULL){
			otherData = *elem;
			ds = length(myData.loc - otherData.loc);
			if (ds < params.neighborhood && ds > 0 ) {
				info.count++;
				m = otherData.lastd;
				res = res + m;
			}
			elem = world->nextAgentDataFromSharedMem<PreyAgentData>(info);
		}

		if (info.count > 0){
			res = res / info.count;
		}

		return res;
	}

	__device__ FLOATn cohesion(const GWorld *world, iterInfo &info)
	{
		FLOATn res = make_float2(0.0f,0.0f);
		float ds;
		FLOATn m;
		PreyAgentData myData = *(PreyAgentData*)this->data;
		world->neighborQueryInit(myData.loc, params.neighborhood, info);
		PreyAgentData otherData;
		PreyAgentData *elem = world->nextAgentDataFromSharedMem<PreyAgentData>(info);
		while(elem != NULL){
			otherData = *elem;
			ds = length(myData.loc - otherData.loc);
			if (ds < params.neighborhood && ds > 0) {
				info.count++;
				res = res + myData.loc - otherData.loc;
			}
			elem = world->nextAgentDataFromSharedMem<PreyAgentData>(info);
		}

		if (info.count > 0){
			res = res / info.count;
		}
		res = -res/10;
		return res;
	}

	__device__ FLOATn avoidance(const GWorld *world, iterInfo &info)
	{
		FLOATn res = make_float2(0,0);
		FLOATn delta = make_float2(0,0);
		float ds;
		PreyAgentData myData = *(PreyAgentData*)this->data;
		world->neighborQueryInit(myData.loc, params.neighborhood, info);
		PreyAgentData otherData;
		PreyAgentData *elem = world->nextAgentDataFromSharedMem<PreyAgentData>(info);
		while(elem != NULL){
			otherData = *elem;
			ds = length(myData.loc - otherData.loc);
			if (ds < params.neighborhood && ds > 0) {
				info.count++;
				delta = myData.loc - otherData.loc;
				float lensquared = dot(delta, delta);
				res = res + delta / ( lensquared *lensquared + 1 );
			}
			elem = world->nextAgentDataFromSharedMem<PreyAgentData>(info);
		}

		if (info.count > 0){
			res = res / info.count;
		}

		res = res * 400;
		return res;
	}

	__device__ FLOATn randomness(){
		float x = this->random->uniform() * 2 - 1.0;
		float y = this->random->uniform() * 2 - 1.0;
		float l = sqrtf(x * x + y * y);
		FLOATn res;
		res.x = 0.05 * x / l;
		res.y = 0.05 * y / l;
		return res;
	}

	__device__ void step(GModel *model)
	{
		BoidModel *boidModel = (BoidModel*) model;
		const GWorld *world = boidModel->world;
		PreyAgentData dataLocal = *(PreyAgentData*)this->data;
		iterInfo info;

		float dx = 0; 
		float dy = 0;

		FLOATn cohes = this->cohesion(world, info);
		FLOATn consi = this->consistency(world, info);
		FLOATn avoid = this->avoidance(world, info);
		FLOATn rdnes = this->randomness();
		FLOATn momen = dataLocal.lastd;
		dx = 0
			+ cohes.x * params.cohesion 
			+ avoid.x * params.avoidance
			+ consi.x * params.consistency
			+ rdnes.x * params.randomness
			+ momen.x * params.momentum
			;
		dy = 0
			+ cohes.y * params.cohesion
			+ avoid.y * params.avoidance
			+ consi.y * params.consistency
			+ rdnes.y * params.randomness
			+ momen.y * params.momentum
			;

		float dist = sqrtf(dx*dx + dy*dy);
		if (dist > 0){
			dx = dx / dist * params.jump;
			dy = dy / dist * params.jump;
		}


		PreyAgentData dummyDataPtr = *(PreyAgentData *)this->dataCopy;
		dummyDataPtr.lastd.x = dx;
		dummyDataPtr.lastd.y = dy;
		dummyDataPtr.loc.x = world->stx(dataLocal.loc.x + dx, world->width);
		dummyDataPtr.loc.y = world->sty(dataLocal.loc.y + dy, world->height);
		*(PreyAgentData *)this->dataCopy = dummyDataPtr;

		AgentPool<PreyAgent, PreyAgentData> *poolLocal = boidModel->pool;

		//float2 center = make_float2(modelDevParams.WIDTH / 2, modelDevParams.HEIGHT / 2);
		//if (length(center - dummyDataPtr.loc) < center.x * 0.1) {
		//if (this->random->uniform() < 0.1) {
		//	poolLocal->remove(this->ptrInPool);
		//	int agentSlot = poolLocal->agentSlot();
		//	int dataSlot = poolLocal->dataSlot(agentSlot);
		//	PreyAgent *ag = &poolLocal->agentArray[dataSlot];
		//	ag->init(boidModel, dataSlot);
		//	poolLocal->add(ag, agentSlot);
		//}
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