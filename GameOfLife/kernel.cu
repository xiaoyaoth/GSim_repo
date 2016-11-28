#include "gsimcore.cuh"
//#include "boid.cuh"
#ifdef _WIN32
#include <Windows.h>
#else
#include <sys/time.h>
#endif
#include "Boid.cuh"
int main(int argc, char *argv[]){
	//argv[1]: config.txt
	//argv[2]: numAgent
	init<GOLAgentData>(argv[1]);
	int agentNum = (int)(modelHostParams.WIDTH * modelHostParams.HEIGHT);
	printf("numAgent: %d\n", agentNum);
	GOLModel *model_h = new GOLModel(agentNum);
	/*Main work started here*/
	doLoop(model_h);
}