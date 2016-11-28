#ifndef BOID_CUH
#define BOID_CUH

#include "gsimcore.cuh"
#ifdef _WIN32
#include "gsimvisual.cuh"
#endif

enum BOID_TYPE {BOID_PREY, BOID_PREDATOR};
class HahaBoid;
struct dataUnion;

struct BoidAgentData : public GAgentData
{
	FLOATn lastd;
};

struct PreyAgentData : public BoidAgentData
{
};

struct PredatorAgentData: public BoidAgentData
{
};

struct HahaBoidData : public GAgentData
{
	FLOATn vel;
	FLOATn acc;
	int mass;
};

struct dataUnion
{
	BOID_TYPE bt;
	union {
		PreyAgentData boidAgentData;
		PredatorAgentData predatorAgentData;
		HahaBoidData hahaBoidData;
	};

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
class HahaPreyAgent;
class HahaPredatorAgent;

__global__ void addAgents(BoidModel *bModel);

#define N_POOL 2
__constant__ int N_HAHA_PREY;
int MAX_N_HAHA_PREY;
__constant__ int N_HAHA_PREDATOR;
int MAX_N_HAHA_PREDAOTR;

class BoidModel : public GModel{
public:
	bool poolMod;
	GRandom *random, *randomHost;
	AgentPool<HahaPreyAgent, HahaBoidData> *hahaPreyPool, *hahaPreyPoolHost;
	AgentPool<HahaPredatorAgent, HahaBoidData> *hahaPredatorPool, *hahaPredatorPoolHost;

	cudaEvent_t timerStart, timerStop;

	__host__ BoidModel(float range, int numPrey, int numPred)
	{
		poolMod = true;
		MAX_N_HAHA_PREDAOTR = numPred * 2;
		MAX_N_HAHA_PREY = numPrey * 2;
		cudaMemcpyToSymbol(N_HAHA_PREDATOR, &numPred, sizeof(int));
		cudaMemcpyToSymbol(N_HAHA_PREY, &numPrey, sizeof(int));

		hahaPreyPoolHost = new AgentPool<HahaPreyAgent, HahaBoidData>(numPrey, MAX_N_HAHA_PREY, sizeof(dataUnion));
		util::hostAllocCopyToDevice<AgentPool<HahaPreyAgent, HahaBoidData>>(hahaPreyPoolHost, &hahaPreyPool);

		hahaPredatorPoolHost = new AgentPool<HahaPredatorAgent, HahaBoidData>(numPred, MAX_N_HAHA_PREDAOTR, sizeof(dataUnion));
		util::hostAllocCopyToDevice<AgentPool<HahaPredatorAgent, HahaBoidData>>(hahaPredatorPoolHost, &hahaPredatorPool);
		
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
		int AGENT_NO = this->hahaPredatorPoolHost->numElem + this->hahaPreyPoolHost->numElem;
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
		this->hahaPreyPoolHost->registerPool(this->worldHost, this->schedulerHost, this->hahaPreyPool);
		this->hahaPredatorPoolHost->registerPool(this->worldHost, this->schedulerHost, this->hahaPredatorPool);
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
		//int gSize = GRID_SIZE(this->poolHost->numElem);
		int numStepped = 0;

		numStepped += this->hahaPreyPoolHost->stepPoolAgent(this->model, numStepped);
		numStepped += this->hahaPredatorPoolHost->stepPoolAgent(this->model, numStepped);
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

class HahaBoid : public GAgent
{
public:
	BoidModel *model;
	GRandom *random;
	BOID_TYPE bt;

	__device__ void avoidForce(const GWorld *world, iterInfo &info, int maxForce, HahaBoidData &myData)
	{
		FLOATn locSum = make_float2(0);
		int separation = myData.mass + 20;

		world->neighborQueryInit(myData.loc, separation, info);
		HahaBoidData otherData;
		dataUnion *elem = world->nextAgentDataFromSharedMem<dataUnion>(info);
		float ds = 0;
		while(elem != NULL){
			if (elem->bt != BOID_PREY) {
				elem = world->nextAgentDataFromSharedMem<dataUnion>(info);
				continue;
			}
			otherData = elem->hahaBoidData;//->boidAgentData;
			ds = length(myData.loc - otherData.loc);
			if (ds < separation && ds > 0) {
				info.count++;
				locSum += otherData.loc;
			}
			elem = world->nextAgentDataFromSharedMem<dataUnion>(info);
		}

		if (info.count > 0) {
			locSum /= info.count;
			FLOATn avoidVec = myData.loc - locSum;
			float mag = length(avoidVec);
			if(mag > maxForce * 2.5) avoidVec *= maxForce * 2.5 / mag;
			applyF(avoidVec, myData);
		}
	}

	__device__ void approachForce(const GWorld *world, iterInfo &info, float approachRadius, int maxForce, HahaBoidData &myData)
	{
		FLOATn locSum = make_float2(0);

		world->neighborQueryInit(myData.loc, approachRadius, info);
		HahaBoidData otherData;
		dataUnion *elem = world->nextAgentDataFromSharedMem<dataUnion>(info);
		float ds = 0;

		while(elem != NULL){
			if (elem->bt != BOID_PREY) {
				elem = world->nextAgentDataFromSharedMem<dataUnion>(info);
				continue;
			}
			otherData = elem->hahaBoidData;//->boidAgentData;
			ds = length(myData.loc - otherData.loc);
			if (ds < approachRadius && ds > 0) {
				info.count++;
				locSum += otherData.loc;
			}
			elem = world->nextAgentDataFromSharedMem<dataUnion>(info);
		}

		if (info.count > 0) {
			locSum /= info.count;
			FLOATn approachVec = locSum - myData.loc;
			float mag = length(approachVec);
			if(mag > maxForce) approachVec *= maxForce / mag;
			applyF(approachVec, myData);
		}
	}

	__device__ void alignForce(const GWorld *world, iterInfo &info, int maxForce, HahaBoidData &myData)
	{
		FLOATn velSum = make_float2(0);
		int alignRadius = myData.mass + 100;

		world->neighborQueryInit(myData.loc, alignRadius, info);
		dataUnion *elem = world->nextAgentDataFromSharedMem<dataUnion>(info);
		float ds = 0;

		while(elem != NULL){
			if (elem->bt != BOID_PREY) {
				elem = world->nextAgentDataFromSharedMem<dataUnion>(info);
				continue;
			}
			HahaBoidData otherData = elem->hahaBoidData;//->boidAgentData;
			ds = length(myData.loc - otherData.loc);
			if (ds < alignRadius && ds > 0) {
				info.count++;
				velSum += otherData.vel;
			}
			elem = world->nextAgentDataFromSharedMem<dataUnion>(info);
		}

		if (info.count > 0) {
			velSum /= info.count;
			FLOATn alignVec = velSum;
			float mag = length(alignVec);
			if(mag > maxForce) alignVec *= maxForce / mag;
			applyF(alignVec, myData);
		}
	}

	__device__ void repelForce(FLOATn obstacle, float radius, int maxForce, HahaBoidData &myData)
	{
		FLOATn futPos = myData.loc + myData.vel;
		FLOATn dist = obstacle - futPos;
		float d = length(dist);

		FLOATn repelVec = make_float2(0);

		if (d <=radius){
			repelVec = myData.loc-obstacle;
			repelVec /= length(repelVec);
			if (d != 0) {
				float mag = length(repelVec);
				if (mag != 0 ) repelVec *= maxForce * 7 / mag ;
				if (length(repelVec) < 0)
					repelVec.y = 0;
			}
			applyF(repelVec, myData);
		}
	}

	__device__ void repelForceAll(const GWorld *world, iterInfo &info, float radius, int maxForce, HahaBoidData &myData)
	{
		world->neighborQueryInit(myData.loc, radius, info);
		dataUnion *elem = world->nextAgentDataFromSharedMem<dataUnion>(info);
		float ds = 0;

		while(elem != NULL){
			if (elem->bt != BOID_PREDATOR) {
				elem = world->nextAgentDataFromSharedMem<dataUnion>(info);
				continue;
			}
			HahaBoidData otherData = elem->hahaBoidData;//->boidAgentData;
			ds = length(myData.loc - otherData.loc);
			if (ds < radius && ds > 0) {
				this->repelForce(otherData.loc, radius, maxForce, myData);
			}
			elem = world->nextAgentDataFromSharedMem<dataUnion>(info);
		}
	}

	__device__ void applyF(FLOATn &force, HahaBoidData &myData)
	{
		force /= (float)myData.mass;
		myData.acc += force;
	}

	__device__ void correct(float velLimit, float width, float height, HahaBoidData &myData)
	{
		myData.vel += myData.acc;
		myData.loc += myData.vel;
		myData.acc *= 0;
		float mag = length(myData.vel);
		if (mag > velLimit) myData.vel *= velLimit / mag;

		if (myData.loc.x < 0)
			myData.loc.x += width;
		if (myData.loc.x >= width)
			myData.loc.x -= width;
		if (myData.loc.y < 0)
			myData.loc.y += height;
		if (myData.loc.y >= height)
			myData.loc.y -= height;
	}
};

class HahaPreyAgent : public HahaBoid
{
public:
	__device__ void init(BoidModel *bModel, int dataSlot)
	{
		this->model = bModel;
		this->random = bModel->random;
		this->color = colorConfigs.black;
		this->bt = BOID_PREY;

		HahaBoidData *myData = &bModel->hahaPreyPool->dataArray[dataSlot];
		HahaBoidData *myDataCopy = &bModel->hahaPreyPool->dataCopyArray[dataSlot];

		myData->loc = make_float2(random->uniform() * modelDevParams.WIDTH, random->uniform() * modelDevParams.HEIGHT);
		myData->vel = make_float2(0,0);
		myData->acc = make_float2(0,0);
		myData->mass = 5 * random->uniform() + 5;

		*myDataCopy = *myData;
		this->data = myData;
		this->dataCopy = myDataCopy;
	}
	
	__device__ void step(GModel *model)
	{
		BoidModel *bModel = (BoidModel*)model;
		const GWorld *world = bModel->world;
		HahaBoidData myData = *(HahaBoidData*)this->data;
		iterInfo info;

		float approachRadius = myData.mass + 60;
		float repelRadius = 60;
		int maxForce = params.maxForce;
		this->repelForceAll(world, info, repelRadius, maxForce, myData);
		this->avoidForce(world, info, maxForce, myData);
		this->approachForce(world, info, approachRadius, maxForce, myData);
		this->alignForce(world, info, maxForce, myData);
		
		this->correct(5, world->width, world->height, myData);

		//if (this->random->uniform() < 0.1) {
		//	bModel->hahaPreyPool->remove(this->ptrInPool);
		//	int agentSlot = bModel->hahaPreyPool->agentSlot();
		//	int dataSlot = bModel->hahaPreyPool->dataSlot(agentSlot);
		//	HahaPreyAgent *ag = &bModel->hahaPreyPool->agentArray[dataSlot];

		//	ag->init(bModel, dataSlot);
		//	bModel->hahaPreyPool->add(ag, agentSlot);
		//}

		*(HahaBoidData*)this->dataCopy = myData;
	}

	__device__ void setDataInSmem(void *elem)
	{
		dataUnion *dataInSmem = (dataUnion*)elem;
		dataInSmem->bt = BOID_PREY;
		dataInSmem->hahaBoidData = *(HahaBoidData*)this->data;
	}
};

class HahaPredatorAgent : public HahaBoid
{
public:
	__device__ void init(BoidModel *bModel, int dataSlot)
	{
		this->model = bModel;
		this->random = bModel->random;
		this->color = colorConfigs.red;
		this->bt = BOID_PREDATOR;

		HahaBoidData *myData = &bModel->hahaPredatorPool->dataArray[dataSlot];
		HahaBoidData *myDataCopy = &bModel->hahaPredatorPool->dataCopyArray[dataSlot];

		myData->loc = make_float2(random->uniform() * modelDevParams.WIDTH, random->uniform() * modelDevParams.HEIGHT);
		myData->vel = make_float2(0,0);
		myData->acc = make_float2(0,0);
		myData->mass = 7 * random->uniform() + 8;

		*myDataCopy = *myData;
		this->data = myData;
		this->dataCopy = myDataCopy;
	}

	__device__ void step(GModel *model)
	{
		BoidModel *bModel = (BoidModel*)model;
		const GWorld *world = bModel->world;
		HahaBoidData myData = *(HahaBoidData*)this->data;
		iterInfo info;

		float approachRadius = myData.mass + 260;
		float repelRadius = 30;
		int maxForce = params.maxForcePredator;
		this->repelForceAll(world, info, repelRadius, maxForce, myData);
		this->avoidForce(world, info, maxForce, myData);
		this->approachForce(world, info, approachRadius, maxForce, myData);
		this->alignForce(world, info, maxForce, myData);

		this->correct(6, world->width, world->height, myData);

		*(HahaBoidData*)this->dataCopy = myData;
	}

	__device__ void setDataInSmem(void *elem)
	{
		dataUnion *dataInSmem = (dataUnion*)elem;
		dataInSmem->bt = BOID_PREDATOR;
		dataInSmem->hahaBoidData = *(HahaBoidData*)this->data;
	}
};

__device__ void dataUnion::putDataInSmem(GAgent *ag)
{
	BOID_TYPE bt = ((HahaBoid*)ag)->bt;
	this->bt = bt;
	this->hahaBoidData = *(HahaBoidData*)ag->data;
}

__global__ void addAgents(BoidModel *model)
{
	uint idx = threadIdx.x + blockIdx.x * blockDim.x;
	int dataSlot = -1;
	if (idx < N_HAHA_PREY) {
		dataSlot = idx;
		HahaPreyAgent *ag = &model->hahaPreyPool->agentArray[dataSlot];
		ag->init(model, dataSlot);
		model->hahaPreyPool->add(ag, dataSlot);
	} else if (idx < N_HAHA_PREY + N_HAHA_PREDATOR) {
		dataSlot = idx - N_HAHA_PREY;
		HahaPredatorAgent *ag = &model->hahaPredatorPool->agentArray[dataSlot];
		ag->init(model, dataSlot);
		model->hahaPredatorPool->add(ag, dataSlot);
	}
}
#endif