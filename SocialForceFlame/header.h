
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

#ifndef __HEADER
#define __HEADER

#include <cutil.h>
#include <cutil_math.h>

/* General standard definitions */
//Threads per block (agents per block)
#define THREADS_PER_TILE 64
//Definition for any agent function or helper function
#define __FLAME_GPU_FUNC__ __device__
//Definition for a function used to initialise environment variables
#define __FLAME_GPU_INIT_FUNC__ 

#define NUM_AGENT 16384 

/* Agent population size definifions must be a multiple of THREADS_PER_TILE (defualt 64) */
//Maximum buffer size (largest agent buffer size)
#define buffer_size_MAX NUM_AGENT

//Maximum population size of xmachine_memory_Boid
#define xmachine_memory_Boid_MAX NUM_AGENT
  
  
/* Message poulation size definitions */
//Maximum population size of xmachine_mmessage_location
#define xmachine_message_location_MAX NUM_AGENT



/* Spatial partitioning grid size definitions */
//xmachine_message_location partition grid size (gridDim.X*gridDim.Y*gridDim.Z)
#define xmachine_message_location_grid_size 4913
  
  
/* enum types */

/**
 * MESSAGE_OUTPUT used for all continuous messaging
 */
enum MESSAGE_OUTPUT{
	single_message,
	optional_message,
};

/**
 * AGENT_TYPE used for templates device message functions
 */
enum AGENT_TYPE{
	CONTINUOUS,
	DISCRETE_2D
};

#define DIST(ax, ay, bx, by) sqrt((ax-bx)*(ax-bx)+(ay-by)*(ay-by))

struct obstacleLine
{
	float sx;
	float sy;
	float ex;
	float ey;

	__host__ void init(float sxx, float syy, float exx, float eyy)
	{
		sx = sxx;
		sy = syy;
		ex = exx;
		ey = eyy;
	}

	__device__ float pointToLineDist(float2 loc, float &crx, float &cry) 
	{
		float d = DIST(sx, sy, ex, ey);
		float t0 = ((ex - sx) * (loc.x - sx) + (ey - sy) * (loc.y - sy)) / (d * d);

		if(t0 < 0){
			d = sqrt((loc.x - sx) * (loc.x - sx) + (loc.y - sy) * (loc.y - sy));
		}else if(t0 > 1){
			d = sqrt((loc.x - ex) * (loc.x - ex) + (loc.y - ey) * ( loc.y - ey));
		}else{
			d = sqrt(
				(loc.x - (sx + t0 * ( ex  - sx))) * (loc.x - (sx + t0 * ( ex  - sx))) +
				(loc.y - (sy + t0 * ( ey  - sy))) * (loc.y - (sy + t0 * ( ey  - sy)))
				);
		}
		crx = sx + t0 * (ex - sx);
		cry = sy + t0 * (ey - sy);
		return d;
	}

	__device__ int intersection2LineSeg(float p0x, float p0y, float p1x, float p1y, float &ix, float &iy)
	{
		float s1x, s1y, s2x, s2y;
		s1x = p1x - p0x;
		s1y = p1y - p0y;
		s2x = ex - sx;
		s2y = ey - sy;

		float s, t;
		s = (-s1y * (p0x - sx) + s1x * (p0y - sy)) / (-s2x * s1y + s1x * s2y);
		t = ( s2x * (p0y - sy) - s2y * (p0x - sx)) / (-s2x * s1y + s1x * s2y);

		if (s >= 0 && s <= 1 && t >= 0 && t <= 1)
		{
			// Collision detected
			if (ix != NULL)
				ix = p0x + (t * s1x);
			if (iy != NULL)
				iy = p0y + (t * s1y);
			return 1;
		}
		return 0; // No collision
	}
};

__constant__ struct obstacleLine obsLines[10];
__constant__ int obsLineNum;

#define WIDTH_D 100
#define HEIGHT_D 100


/* Agent structures */

/** struct xmachine_memory_Boid
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_Boid
{
    int id;    /**< X-machine memory variable id of type int.*/
    float x;    /**< X-machine memory variable x of type float.*/
    float y;    /**< X-machine memory variable y of type float.*/
    float goalX;    /**< X-machine memory variable goalX of type float.*/
    float goalY;    /**< X-machine memory variable goalY of type float.*/
    float velX;    /**< X-machine memory variable velX of type float.*/
    float velY;    /**< X-machine memory variable velY of type float.*/
    float velMax;    /**< X-machine memory variable velMax of type float.*/
    float mass;    /**< X-machine memory variable mass of type float.*/
};



/* Message structures */

/** struct xmachine_message_location
 * Spatial Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_location
{	
    /* Spatial Partitioning Variables */
    int3 _relative_cell;    /**< Relative cell position from agent grid cell poistion range -1 to 1 */
    int _cell_index_max;    /**< Max boundary value of current cell */
    int3 _agent_grid_cell;  /**< Agents partition cell position */
    int _cell_index;        /**< Index of position in current cell */  
      
    float x;        /**< Message variable x of type float.*/  
    float y;        /**< Message variable y of type float.*/  
    float z;        /**< Message variable z of type float.*/  
    float velX;        /**< Message variable velX of type float.*/  
    float velY;        /**< Message variable velY of type float.*/  
    float mass;        /**< Message variable mass of type float.*/
};



/* Agent lists. Structure of Array (SoA) for memory coalescing on GPU */

/** struct xmachine_memory_Boid_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_Boid_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_Boid_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_Boid_MAX];  /**< Used during parallel prefix sum */
    
    int id [xmachine_memory_Boid_MAX];    /**< X-machine memory variable list id of type int.*/
    float x [xmachine_memory_Boid_MAX];    /**< X-machine memory variable list x of type float.*/
    float y [xmachine_memory_Boid_MAX];    /**< X-machine memory variable list y of type float.*/
    float goalX [xmachine_memory_Boid_MAX];    /**< X-machine memory variable list goalX of type float.*/
    float goalY [xmachine_memory_Boid_MAX];    /**< X-machine memory variable list goalY of type float.*/
    float velX [xmachine_memory_Boid_MAX];    /**< X-machine memory variable list velX of type float.*/
    float velY [xmachine_memory_Boid_MAX];    /**< X-machine memory variable list velY of type float.*/
    float velMax [xmachine_memory_Boid_MAX];    /**< X-machine memory variable list velMax of type float.*/
    float mass [xmachine_memory_Boid_MAX];    /**< X-machine memory variable list mass of type float.*/
};



/* Message lists. Structure of Array (SoA) for memory coalescing on GPU */

/** struct xmachine_message_location_list
 * Spatial Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_location_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_location_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_location_MAX];  /**< Used during parallel prefix sum */
    
    float x [xmachine_message_location_MAX];    /**< Message memory variable list x of type float.*/
    float y [xmachine_message_location_MAX];    /**< Message memory variable list y of type float.*/
    float z [xmachine_message_location_MAX];    /**< Message memory variable list z of type float.*/
    float velX [xmachine_message_location_MAX];    /**< Message memory variable list velX of type float.*/
    float velY [xmachine_message_location_MAX];    /**< Message memory variable list velY of type float.*/
    float mass [xmachine_message_location_MAX];    /**< Message memory variable list mass of type float.*/
    
};



/* Spatialy Partitioned Message boundary Matrices */

/** struct xmachine_message_location_PBM
 * Partition Boundary Matrix (PBM) for xmachine_message_location 
 */
struct xmachine_message_location_PBM
{
	int start[xmachine_message_location_grid_size];
	int end[xmachine_message_location_grid_size];
};



/* Random */
/** struct RNG_rand48 
 *	structure used to hold list seeds
 */
struct RNG_rand48
{
  uint2 A, C;
  uint2 seeds[buffer_size_MAX];
};


/* Random Functions (usable in agent functions) implemented in FLAMEGPU_Kernels */

/**
 * Templated random function using a DISCRETE_2D template calculates the agent index using a 2D block
 * which requires extra processing but will work for CONTINUOUS agents. Using a CONTINUOUS template will
 * not work for DISCRETE_2D agent.
 * @param	rand48	an RNG_rand48 struct which holds the seeds sued to generate a random number on the GPU
 * @return			returns a random float value
 */
template <int AGENT_TYPE> __FLAME_GPU_FUNC__ float rnd(RNG_rand48* rand48);
/**
 * Non templated random function calls the templated version with DISCRETE_2D which will work in either case
 * @param	rand48	an RNG_rand48 struct which holds the seeds sued to generate a random number on the GPU
 * @return			returns a random float value
 */
__FLAME_GPU_FUNC__ float rnd(RNG_rand48* rand48);

/* Agent function prototypes */

/**
 * outputdata FLAMEGPU Agent Function
 * @param agent Pointer to an agent structre of type xmachine_memory_Boid. This represents a single agent instance and can be modified directly.
 * @param location_messages Pointer to output message list of type xmachine_message_location_list. Must be passed as an argument to the add_location_message function ??.
 */
__FLAME_GPU_FUNC__ int outputdata(xmachine_memory_Boid* agent, xmachine_message_location_list* location_messages);

/**
 * inputdata FLAMEGPU Agent Function
 * @param agent Pointer to an agent structre of type xmachine_memory_Boid. This represents a single agent instance and can be modified directly.
 * @param location_messages  location_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_location_message and get_next_location_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_location_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.
 */
__FLAME_GPU_FUNC__ int inputdata(xmachine_memory_Boid* agent, xmachine_message_location_list* location_messages, xmachine_message_location_PBM* partition_matrix);

  
/* Message Function Prototypes for Spatially Partitioned location message implemented in FLAMEGPU_Kernels */

/** add_location_message
 * Function for all types of message partitioning
 * Adds a new location agent to the xmachine_memory_location_list list using a linear mapping
 * @param agents	xmachine_memory_location_list agent list
 * @param x	message variable of type float
 * @param y	message variable of type float
 * @param z	message variable of type float
 * @param velX	message variable of type float
 * @param velY	message variable of type float
 * @param mass	message variable of type float
 */
 
 __FLAME_GPU_FUNC__ void add_location_message(xmachine_message_location_list* location_messages, float x, float y, float z, float velX, float velY, float mass);
 
/** get_first_location_message
 * Get first message function for spatially partitioned messages
 * @param location_messages message list
 * @param partition_matrix the boundary partition matrix for the spatially partitioned message list
 * @param agentx x position of the agent
 * @param agenty y position of the agent
 * @param agentz z position of the agent
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_location * get_first_location_message(xmachine_message_location_list* location_messages, xmachine_message_location_PBM* partition_matrix, float x, float y, float z);

/** get_next_location_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memeory or texture cache implementation depending on AGENT_TYPE
 * @param current the current message struct
 * @param location_messages message list
 * @param partition_matrix the boundary partition matrix for the spatially partitioned message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_location * get_next_location_message(xmachine_message_location* current, xmachine_message_location_list* location_messages, xmachine_message_location_PBM* partition_matrix);
  
  
  
/* Agent Function Prototypes implemented in FLAMEGPU_Kernels */

/** add_Boid_agent
 * Adds a new continuous valued Boid agent to the xmachine_memory_Boid_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_Boid_list agent list
 * @param id	agent agent variable of type int
 * @param x	agent agent variable of type float
 * @param y	agent agent variable of type float
 * @param goalX	agent agent variable of type float
 * @param goalY	agent agent variable of type float
 * @param velX	agent agent variable of type float
 * @param velY	agent agent variable of type float
 * @param velMax	agent agent variable of type float
 * @param mass	agent agent variable of type float
 */
__FLAME_GPU_FUNC__ void add_Boid_agent(xmachine_memory_Boid_list* agents, int id, float x, float y, float goalX, float goalY, float velX, float velY, float velMax, float mass);


  
/* Simulation function prototypes implemented in simulation.cu */

/** initialise
 * Initialise the simulation. Allocated host and device memory. Reads the initial agent configuration from XML.
 * @param input	XML file path for agent initial configuration
 */
extern "C" void initialise(char * input);

/** cleanup
 * Function cleans up any memory allocations on the host and device
 */
extern "C" void cleanup();

/** singleIteration
 *	Performs a single itteration of the simulation. I.e. performs each agent function on each function layer in the correct order.
 */
extern "C" void singleIteration();

/** saveIterationData
 * Reads the current agent data fromt he device and saves it to XML
 * @param	outputpath	file path to XML file used for output of agent data
 * @param	itteration_number
 * @param h_Boids Pointer to agent list on the host
 * @param d_Boids Pointer to agent list on the GPU device
 * @param h_xmachine_memory_Boid_count Pointer to agent counter
 */
extern "C" void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_Boid_list* h_Boids_default, xmachine_memory_Boid_list* d_Boids_default, int h_xmachine_memory_Boid_default_count);


/** readInitialStates
 * Reads the current agent data fromt he device and saves it to XML
 * @param	inputpath	file path to XML file used for input of agent data
 * @param h_Boids Pointer to agent list on the host
 * @param h_xmachine_memory_Boid_count Pointer to agent counter
 */
extern "C" void readInitialStates(char* inputpath, xmachine_memory_Boid_list* h_Boids, int* h_xmachine_memory_Boid_count);


/* Return functions used by external code to get agent data from device */

    
/** get_agent_Boid_MAX_count
 * Gets the max agent count for the Boid agent type 
 * @return		the maximum Boid agent count
 */
extern "C" int get_agent_Boid_MAX_count();



/** get_agent_Boid_default_count
 * Gets the agent count for the Boid agent type in state default
 * @return		the current Boid agent count in state default
 */
extern "C" int get_agent_Boid_default_count();

/** reset_default_count
 * Resets the agent count of the Boid in state default to 0. This is usefull for interacting with some visualisations.
 */
extern "C" void reset_Boid_default_count();

/** get_device_Boid_default_agents
 * Gets a pointer to xmachine_memory_Boid_list on the GPU device
 * @return		a xmachine_memory_Boid_list on the GPU device
 */
extern "C" xmachine_memory_Boid_list* get_device_Boid_default_agents();

/** get_host_Boid_default_agents
 * Gets a pointer to xmachine_memory_Boid_list on the CPU host
 * @return		a xmachine_memory_Boid_list on the CPU host
 */
extern "C" xmachine_memory_Boid_list* get_host_Boid_default_agents();


/** sort_Boids_default
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_Boids_default(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_Boid_list* agents));


  
  
/* global constant variables */


/** getMaximumBound
 * Returns the maximum agent positions determined from the initial loading of agents
 * @return 	a three component float indicating the maximum x, y and z positions of all agents
 */
float3 getMaximumBounds();

/** getMinimumBounds
 * Returns the minimum agent positions determined from the initial loading of agents
 * @return 	a three component float indicating the minimum x, y and z positions of all agents
 */
float3 getMinimumBounds();
    
    
#ifdef VISUALISATION
/** initVisualisation
 * Prototype for method which initialises the visualisation. Must be implemented in seperate file
 * @param argc	the argument count from the main function used with GLUT
 * @param argv	the argument values fromt the main function used with GLUT
 */
extern "C" void initVisualisation();

extern "C" void runVisualisation();


#endif

#endif //__HEADER

