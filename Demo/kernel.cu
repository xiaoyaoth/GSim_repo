#include "Demo.cuh"

int main(int argc, char **argv)
{
	init<DataUnion>(argv[1]);
	DemoModel *dModel = new DemoModel();
	doLoop(dModel);
}