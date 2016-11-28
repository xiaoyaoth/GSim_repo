
/*
 * FLAME GPU v 1.2.0 for CUDA 6
 * Copyright 2014 University of Sheffield.
 * Author: Dr Paul Richmond 
 * Contact: p.richmond@sheffield.ac.uk (http://www.paulrichmond.staff.shef.ac.uk)
 *
 * University of Sheffield retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * University of Sheffield is strictly prohibited.
 *
 * For terms of licence agreement please attached licence or view licence 
 * on www.flamegpu.com website.
 * 
 */

#include <header.h>
#include <stdio.h>
#include <cutil.h>
#ifdef VISUALISATION
#include <GL/glew.h>
#include <GL/glut.h>
#endif

/* IO Variables*/
char inputfile[100];          /**< Input path char buffer*/
char outputpath[100];         /**< Output path char buffer*/
int CUDA_argc;				  /**< number of CUDA arguments*/
char** CUDA_argv;			  /**< CUDA arguments*/

#define OUTPUT_TO_XML 0


/** checkUsage
 * Function to check the correct number of arguments
 * @param arc	main argument count
 * @param argv	main argument values
 * @return true if usage is correct, otherwise false
 */
int checkUsage( int argc, char** argv){
	//Check usage
#ifdef VISUALISATION
	printf("FLAMEGPU Visualisation mode\n");
	if(argc < 2)
	{
		printf("Usage: main [XML model data] [Optional CUDA arguments]\n");
		return false;
	}
#else
	printf("FLAMEGPU Console mode\n");
	if(argc < 3)
	{
		printf("Usage: main [XML model data] [Itterations] [Optional CUDA arguments]\n");
		return false;
	}
#endif
	return true;
}


/** setFilePaths
 * Function to set global variables for the input XML file and its directory location
 *@param input input path of model xml file
 */
void setFilePaths(char* input){
	//Copy input file
	strcpy(inputfile, input);
	printf("Initial states: %s\n", inputfile);

	//Calculate the output path from the path of the input file
	int i = 0;
	int lastd = -1;
	while(inputfile[i] != '\0')
	{
		/* For windows directories */
		if(inputfile[i] == '\\') lastd=i;
		/* For unix directories */
		if(inputfile[i] == '/') lastd=i;
		i++;
	}
	strcpy(outputpath, inputfile);
	outputpath[lastd+1] = '\0';
	printf("Ouput dir: %s\n", outputpath);
}


void initCUDA(int argc, char** argv){
	//start position of CUDA arguments in arg v
	int CUDA_start_args;
#ifdef VISUALISATION
	//less model file argument
	CUDA_argc = argc-1;
	CUDA_start_args = 2;
#else
	//less model file and itterations arguments
	CUDA_argc = argc-2;
	CUDA_start_args = 3;
#endif

	//copy first argument
	CUDA_argv = new char*[CUDA_argc];
	size_t dst_size = strlen(argv[0])+1; //+/0
	CUDA_argv[0] = new char[dst_size];
	strcpy_s(CUDA_argv[0], dst_size, argv[0]);
	
	//all args after FLAME GPU specific are passed to CUDA
	int j = 1;
	for (int i=CUDA_start_args; i<argc; i++){
		dst_size = strlen(argv[i])+1; //+/0
		CUDA_argv[j] = new char[dst_size];
		strcpy_s(CUDA_argv[j], dst_size, argv[i]);
		j++;
	}

	//CUT_DEVICE_INIT(CUDA_argc, CUDA_argv);
}


/**
 * Program main (Handles arguments)
 */
int main( int argc, char** argv) 
{
	//check usage mode
	if (!checkUsage(argc, argv))
		exit(0);

	//get the directory paths
	setFilePaths(argv[1]);

	//initialise CUDA
	initCUDA(argc, argv);

#ifdef VISUALISATION
	//Init visualisation must be done before simulation init
	initVisualisation();
#endif

	//initialise the simulation
    initialise(inputfile);

    
#ifdef VISUALISATION
	runVisualisation();
	exit(0);
#else	
	//Get the number of itterations
	int itterations = atoi(argv[2]);
	if (itterations == 0)
	{
		printf("Second argument must be an integer (Number of Itterations)\n");
		exit(0);
	}
  
	//Benchmark simulation
	unsigned int timer = 0;
	CUT_SAFE_CALL( cutCreateTimer(&timer));
	CUT_SAFE_CALL( cutStartTimer( timer));

	for (int i=0; i< itterations; i++)
	{
		printf("Processing Simulation Step %i", i+1);

		//single simulation itteration
		singleIteration();

		if (OUTPUT_TO_XML)
		{
			saveIterationData(outputpath, i+1, 
				//default state Boid agents
				get_host_Boid_default_agents(), get_device_Boid_default_agents(), get_agent_Boid_default_count());
			
				printf(": Saved to XML:");
		}

		printf(": Done\n");
	}

	//CUDA stop timing
	cudaThreadSynchronize();
	CUT_SAFE_CALL( cutStopTimer( timer));
	printf( "Total Processing time: %f (ms)\n", cutGetTimerValue( timer));
	CUT_SAFE_CALL( cutDeleteTimer( timer));

#endif

	cleanup();
    CUT_EXIT(CUDA_argc, CUDA_argv);
}
