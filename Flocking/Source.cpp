//class BoidModel : public GModel{
//public:
//	GRandom *random, *randomHost;
//	AgentPool *pool, *poolHost;
//	BoidModel *devPtrToSelf;
//
//	__host__ BoidModel(float range, int numBoid){
//		//initialize modules and simulation parameters
//	}
//	__host__ void start(){
//		int gSize = GRID_SIZE(this->poolHost->numElem;);
//		addAgents<<<gSize, BLOCK_SIZE>>>((BoidModel*)this->model);
//		GSimVisual::getInstance().setWorld(this->world);
//	}
//	__host__ void preStep(){
//		this->poolHost->registerPool(this->worldHost, this->pool);
//		GSimVisual::getInstance().animate();
//	}
//	__host__ void step(){
//		this->poolHost->stepPoolAgent(this->model);
//	}
//	__host__ void stop(){
//		GSimVisual::getInstance().stop();
//	}
//};
//
//class Boid : public GAgent
//{
//public:
//	BoidModel *model;
//	GRandom *random;
//	AgentPool *pool;
//
//	__device__ void init(BoidModel *bModel, int dataSlot){
//		/*initialize agent.*/
//	}
//	__device__ void step(GModel *model){
//		BoidModel *boidModel = (BoidModel*) model;
//		const GWorld *world = boidModel->world;
//
//		float2 m, cohes, consi, avoid;
//		int counter = 0;
//		BoidData myData = *(BoidData*)this->data;
//		world->neighborQueryInit(myData.loc, params.neighborhood, info);
//		BoidData *other = world->nextAgentDataFromSharedMem<BoidData>(info);
//		while(other){
//			otherData = *other;
//			float ds = length(myData.loc - otherData.loc);
//			if (ds < params.neighborhood && ds > 0 ) {
//				counter++;
//				updateCohension(myData, otherData, cohes);
//				updateConsistency(myData, otherData, consi);
//				updateAvoidance(myData, otherData, avoid);
//			}
//			other = world->nextAgentDataFromSharedMem(info);
//		}
//		float2 delta = (cones + cons + avoid) / counter; 
//
//		BoidData *dummyDataPtr = (BoidData *)this->dataCopy;
//		dummyDataPtr->lastd.x = dx;
//		dummyDataPtr->lastd.y = dy;
//		dummyDataPtr->loc.x = dataLocal.loc.x + dx, world->width;
//		dummyDataPtr->loc.y = dataLocal.loc.y + dy, world->height;
//	}
//};