#include "gsimcore.cuh"
//#include "boid.cuh"
#ifdef _WIN32
#include <Windows.h>
#else
#include <sys/time.h>
#endif
//#include "test.cuh"
#include "Boid.cuh"
int main(int argc, char *argv[]){
	//argv[1]: config.txt
	//argv[2]: query range of flocking model
	//argv[3-4]: predation parameters
	init<dataUnion>(argv[1]);
	int numPrey = atoi(argv[3]);
	int numPred = atoi(argv[4]);
	BoidModel *model_h = new BoidModel(atof(argv[2]), numPrey, numPred);
	/*Main work started here*/
	doLoop(model_h);
}
