#include "gsimcore.cuh"
//#include "boid.cuh"
#ifdef _WIN32
#include <Windows.h>
#else
#include <sys/time.h>
#endif
//#include "test.cuh"
#include "AddingDeleting.cuh"
int main(int argc, char *argv[]){
	//argv[1]: config.txt
	//argv[2]: numAgent
	init<PreyAgentData>(argv[1]);
	int numPrey = atoi(argv[2]);
	int delRange = atoi(argv[3]);
	BoidModel *model_h = new BoidModel(numPrey, delRange);
	/*Main work started here*/
	doLoop(model_h);
}
