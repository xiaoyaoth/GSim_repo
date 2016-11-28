
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

// includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

//Disable internal thrust warnings about conversions
#pragma warning (disable : 4267)
#pragma warning (disable : 4244)

// include FLAME kernels
#include "FLAMEGPU_kernals.cu"

/* SM padding and offset variables */
int SM_START;
int PADDING;

/* Agent Memory */

/* Boid Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_Boid_list* d_Boids;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_Boid_list* d_Boids_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_Boid_list* d_Boids_new;  /**< Pointer to new agent list on the device (used to hold new agents bfore they are appended to the population)*/
int h_xmachine_memory_Boid_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_Boid_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_Boid_values;  /**< Agent sort identifiers value */
    
/* Boid state variables */
xmachine_memory_Boid_list* h_Boids_default;      /**< Pointer to agent list (population) on host*/
xmachine_memory_Boid_list* d_Boids_default;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_Boid_default_count;   /**< Agent population size counter */ 


/* Message Memory */

/* location Message variables */
xmachine_message_location_list* h_locations;         /**< Pointer to message list on host*/
xmachine_message_location_list* d_locations;         /**< Pointer to message list on device*/
xmachine_message_location_list* d_locations_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_location_count;         /**< message list counter*/
int h_message_location_output_type;   /**< message output type (single or optional)*/
/* Spatial Partitioning Variables*/
uint * d_xmachine_message_location_keys;	  /**< message sort identifier keys*/
uint * d_xmachine_message_location_values;  /**< message sort identifier values */
xmachine_message_location_PBM * d_location_partition_matrix;  /**< Pointer to PCB matrix */
float3 h_message_location_min_bounds;           /**< min bounds (x,y,z) of partitioning environment */
float3 h_message_location_max_bounds;           /**< max bounds (x,y,z) of partitioning environment */
int3 h_message_location_partitionDim;           /**< partition dimensions (x,y,z) of partitioning environment */
float h_message_location_radius;                 /**< partition radius (used to determin the size of the partitions) */
/* Texture offset values for host */
int h_tex_xmachine_message_location_x_offset;
int h_tex_xmachine_message_location_y_offset;
int h_tex_xmachine_message_location_z_offset;
int h_tex_xmachine_message_location_velX_offset;
int h_tex_xmachine_message_location_velY_offset;
int h_tex_xmachine_message_location_mass_offset;
int h_tex_xmachine_message_location_pbm_start_offset;
int h_tex_xmachine_message_location_pbm_end_offset;


/*Global condition counts*/

/* RNG rand48 */
RNG_rand48* h_rand48;    /**< Pointer to RNG_rand48 seed list on host*/
RNG_rand48* d_rand48;    /**< Pointer to RNG_rand48 seed list on device*/

/* CUDA Parallel Primatives variables */
int scan_last_sum;           /**< Indicates if the position (in message list) of last message*/
int scan_last_included;      /**< Indicates if last sum value is included in the total sum count*/

/* Agent function prototypes */

/** Boid_outputdata
 * Agent function prototype for outputdata function of Boid agent
 */
void Boid_outputdata();

/** Boid_inputdata
 * Agent function prototype for inputdata function of Boid agent
 */
void Boid_inputdata();

  
void setPaddingAndOffset()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
    int x64_sys = 0;

	// This function call returns 9999 for both major & minor fields, if no CUDA capable devices are present
	if (deviceProp.major == 9999 && deviceProp.minor == 9999){
		printf("Error: There is no device supporting CUDA.\n");
		exit(0);
	}
    
    //check if double is used and supported
#ifdef _DOUBLE_SUPPORT_REQUIRED_
    printf("Simulation requires full precision double values\n");
    if ((deviceProp.major < 2)&&(deviceProp.minor < 3)){
        printf("Error: Hardware does not support full precision double values!\n");
        exit(0);
    }
    
#endif

    //check 32 or 64bit
    x64_sys = (sizeof(void*)==8);
    if (x64_sys)
    {
        printf("64Bit System Detected\n");
    }
    else
    {
        printf("32Bit System Detected\n");
    }

    //check for FERMI
	  if ((deviceProp.major > 2)){
		  printf("Compute %d card detected\n", deviceProp.major);
      SM_START = 0;
      PADDING = 0;
	  }	
    else if ((deviceProp.major == 2)){
		  printf("Fermi Card (compute 2.0) detected\n");
          if (x64_sys){
              SM_START = 0;
              PADDING = 0;
          }else
          {
              SM_START = 0;
              PADDING = 0;
          }
	  }
    //not fermi
    else{
  	    printf("Pre FERMI Card detected (less than compute 2.0)\n");
        if (x64_sys){
            SM_START = 0;
            PADDING = 4;
        }else
        {
            SM_START = 0;
            PADDING = 4;
        }
    }
  
    //copy padding and offset to GPU
    CUDA_SAFE_CALL(cudaMemcpyToSymbol( d_SM_START, &SM_START, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol( d_PADDING, &PADDING, sizeof(int)));

        
}


void initialise(char * inputfile){

	int obsLineNumHost = 2;
	size_t obsLinesSize = sizeof(struct obstacleLine) * obsLineNumHost;
	struct obstacleLine *obsLinesHost = (struct obstacleLine *)malloc(obsLinesSize);
	obsLinesHost[0].init(0.25 * WIDTH_D, -20, 0.25 * WIDTH_D, 0.5 * HEIGHT_D - 2);
	obsLinesHost[1].init(0.25 * WIDTH_D, 0.5 * HEIGHT_D + 1, 0.25 * WIDTH_D, HEIGHT_D + 20);

	cudaMemcpyToSymbol(obsLines, obsLinesHost, obsLinesSize, 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(obsLineNum, &obsLineNumHost, sizeof(int), 0, cudaMemcpyHostToDevice);

    //set the padding and offset values depending on architecture and OS
    setPaddingAndOffset();
  

	printf("Allocating Host and Device memeory\n");
  
	/* Agent memory allocation (CPU) */
	int xmachine_Boid_SoA_size = sizeof(xmachine_memory_Boid_list);
	h_Boids_default = (xmachine_memory_Boid_list*)malloc(xmachine_Boid_SoA_size);

	/* Message memory allocation (CPU) */
	int message_location_SoA_size = sizeof(xmachine_message_location_list);
	h_locations = (xmachine_message_location_list*)malloc(message_location_SoA_size);

    //Exit if agent or message buffer sizes are to small for function outpus
			
	/* Set spatial partitioning location message variables (min_bounds, max_bounds)*/
	h_message_location_radius = (float)6;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol( d_message_location_radius, &h_message_location_radius, sizeof(float)));	
	    h_message_location_min_bounds = make_float3((float)0, (float)0, (float)0);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol( d_message_location_min_bounds, &h_message_location_min_bounds, sizeof(float3)));	
	h_message_location_max_bounds = make_float3((float)100, (float)100, (float)100);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol( d_message_location_max_bounds, &h_message_location_max_bounds, sizeof(float3)));	
	h_message_location_partitionDim.x = (int)ceil((h_message_location_max_bounds.x - h_message_location_min_bounds.x)/h_message_location_radius);
	h_message_location_partitionDim.y = (int)ceil((h_message_location_max_bounds.y - h_message_location_min_bounds.y)/h_message_location_radius);
	h_message_location_partitionDim.z = (int)ceil((h_message_location_max_bounds.z - h_message_location_min_bounds.z)/h_message_location_radius);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol( d_message_location_partitionDim, &h_message_location_partitionDim, sizeof(int3)));	
	


	//read initial states
	readInitialStates(inputfile, h_Boids_default, &h_xmachine_memory_Boid_default_count);
	
	
	/* Boid Agent memory allocation (GPU) */
	CUDA_SAFE_CALL( cudaMalloc( (void**) &d_Boids, xmachine_Boid_SoA_size));
	CUDA_SAFE_CALL( cudaMalloc( (void**) &d_Boids_swap, xmachine_Boid_SoA_size));
	CUDA_SAFE_CALL( cudaMalloc( (void**) &d_Boids_new, xmachine_Boid_SoA_size));
    //continuous agent sort identifiers
  CUDA_SAFE_CALL( cudaMalloc( (void**) &d_xmachine_memory_Boid_keys, xmachine_memory_Boid_MAX* sizeof(uint)));
	CUDA_SAFE_CALL( cudaMalloc( (void**) &d_xmachine_memory_Boid_values, xmachine_memory_Boid_MAX* sizeof(uint)));
	/* default memory allocation (GPU) */
	CUDA_SAFE_CALL( cudaMalloc( (void**) &d_Boids_default, xmachine_Boid_SoA_size));
	CUDA_SAFE_CALL( cudaMemcpy( d_Boids_default, h_Boids_default, xmachine_Boid_SoA_size, cudaMemcpyHostToDevice));
    
	/* location Message memory allocation (GPU) */
	CUDA_SAFE_CALL( cudaMalloc( (void**) &d_locations, message_location_SoA_size));
	CUDA_SAFE_CALL( cudaMalloc( (void**) &d_locations_swap, message_location_SoA_size));
	CUDA_SAFE_CALL( cudaMemcpy( d_locations, h_locations, message_location_SoA_size, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL( cudaMalloc( (void**) &d_location_partition_matrix, sizeof(xmachine_message_location_PBM)));
	CUDA_SAFE_CALL( cudaMalloc( (void**) &d_xmachine_message_location_keys, xmachine_message_location_MAX* sizeof(uint)));
	CUDA_SAFE_CALL( cudaMalloc( (void**) &d_xmachine_message_location_values, xmachine_message_location_MAX* sizeof(uint)));
		

	/*Set global condition counts*/

	/* RNG rand48 */
	int h_rand48_SoA_size = sizeof(RNG_rand48);
	h_rand48 = (RNG_rand48*)malloc(h_rand48_SoA_size);
	//allocate on GPU
	CUDA_SAFE_CALL( cudaMalloc( (void**) &d_rand48, h_rand48_SoA_size));
	// calculate strided iteration constants
	static const unsigned long long a = 0x5DEECE66DLL, c = 0xB;
	int seed = 123;
	unsigned long long A, C;
	A = 1LL; C = 0LL;
	for (unsigned int i = 0; i < buffer_size_MAX; ++i) {
		C += A*c;
		A *= a;
	}
	h_rand48->A.x = A & 0xFFFFFFLL;
	h_rand48->A.y = (A >> 24) & 0xFFFFFFLL;
	h_rand48->C.x = C & 0xFFFFFFLL;
	h_rand48->C.y = (C >> 24) & 0xFFFFFFLL;
	// prepare first nThreads random numbers from seed
	unsigned long long x = (((unsigned long long)seed) << 16) | 0x330E;
	for (unsigned int i = 0; i < buffer_size_MAX; ++i) {
		x = a*x + c;
		h_rand48->seeds[i].x = x & 0xFFFFFFLL;
		h_rand48->seeds[i].y = (x >> 24) & 0xFFFFFFLL;
	}
	//copy to device
	CUDA_SAFE_CALL( cudaMemcpy( d_rand48, h_rand48, h_rand48_SoA_size, cudaMemcpyHostToDevice));

	/* Call all init functions */
	
} 


void sort_Boids_default(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_Boid_list* agents))
{
	dim3 grid;
	dim3 threads;
	int tile_size = (int)ceil((float)h_xmachine_memory_Boid_default_count/THREADS_PER_TILE);
	grid.x = tile_size;
	threads.x = THREADS_PER_TILE;

	//generate sort keys
	generate_key_value_pairs<<<grid, threads>>>(d_xmachine_memory_Boid_keys, d_xmachine_memory_Boid_values, d_Boids_default);
  CUT_CHECK_ERROR("Kernel execution failed");

  //updated Thrust sort
  thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_Boid_keys),  thrust::device_pointer_cast(d_xmachine_memory_Boid_keys) + h_xmachine_memory_Boid_default_count,  thrust::device_pointer_cast(d_xmachine_memory_Boid_values));
  CUT_CHECK_ERROR("Kernel execution failed");

  //reorder agents
  reorder_Boid_agents<<<grid, threads>>>(d_xmachine_memory_Boid_values, d_Boids_default, d_Boids_swap);
	CUT_CHECK_ERROR("Kernel execution failed");

	//swap
	xmachine_memory_Boid_list* d_Boids_temp = d_Boids_default;
	d_Boids_default = d_Boids_swap;
	d_Boids_swap = d_Boids_temp;	
}


void cleanup(){

	/* Agent data free*/
	
	/* Boid Agent variables */
	CUDA_SAFE_CALL(cudaFree(d_Boids));
	CUDA_SAFE_CALL(cudaFree(d_Boids_swap));
	CUDA_SAFE_CALL(cudaFree(d_Boids_new));
	
	free( h_Boids_default);
	CUDA_SAFE_CALL(cudaFree(d_Boids_default));
	

	/* Message data free */
	
	/* location Message variables */
	free( h_locations);
	CUDA_SAFE_CALL(cudaFree(d_locations));
	CUDA_SAFE_CALL(cudaFree(d_locations_swap));
	CUDA_SAFE_CALL(cudaFree(d_location_partition_matrix));
	CUDA_SAFE_CALL(cudaFree(d_xmachine_message_location_keys));
	CUDA_SAFE_CALL(cudaFree(d_xmachine_message_location_values));
	
}

void singleIteration(){

	/* set all non partitioned and spatial partitionded message counts to 0*/
	h_message_location_count = 0;
	//upload to device constant
	CUDA_SAFE_CALL(cudaMemcpyToSymbol( d_message_location_count, &h_message_location_count, sizeof(int)));
	

	/* Call agent functions in order itterating through the layer functions */
	
	/* Layer 1*/
	Boid_outputdata();
	
	/* Layer 2*/
	Boid_inputdata();
	

			
	//Syncronise thread blocks (and relax)
	cudaThreadSynchronize();
}

/* Environment functions */



/* Agent data access functions*/

    
int get_agent_Boid_MAX_count(){
    return xmachine_memory_Boid_MAX;
}


int get_agent_Boid_default_count(){
	//continuous agent
	return h_xmachine_memory_Boid_default_count;
	
}

xmachine_memory_Boid_list* get_device_Boid_default_agents(){
	return d_Boids_default;
}

xmachine_memory_Boid_list* get_host_Boid_default_agents(){
	return h_Boids_default;
}



/* Agent functions */


/** Boid_outputdata
 * Agent function prototype for outputdata function of Boid agent
 */
void Boid_outputdata(){
	dim3 grid;
	dim3 threads;
	int sm_size;
	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_Boid_default_count == 0)
	{
		return;
	}
	
	
	//SET GRID AND BLOCK SIZES
	//set tile size depending on agent count, set a 1d grid and block
	int tile_size = (int)ceil((float)h_xmachine_memory_Boid_default_count/THREADS_PER_TILE);
	grid.x = tile_size;
	threads.x = THREADS_PER_TILE;
	sm_size = SM_START;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_Boid_list* Boids_default_temp = d_Boids;
	d_Boids = d_Boids_default;
	d_Boids_default = Boids_default_temp;
	//set working count to current state count
	h_xmachine_memory_Boid_count = h_xmachine_memory_Boid_default_count;
	CUDA_SAFE_CALL( cudaMemcpyToSymbol( d_xmachine_memory_Boid_count, &h_xmachine_memory_Boid_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_Boid_default_count = 0;
	CUDA_SAFE_CALL( cudaMemcpyToSymbol( d_xmachine_memory_Boid_default_count, &h_xmachine_memory_Boid_default_count, sizeof(int)));	
	

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_location_count + h_xmachine_memory_Boid_count > xmachine_message_location_MAX){
		printf("Error: Buffer size of location message will be exceeded in function outputdata\n");
		exit(0);
	}
	
	//SET THE OUTPUT MESSAGE TYPE
	//Set the message_type for non partitioned and spatially partitioned message outputs
	h_message_location_output_type = single_message;
	CUDA_SAFE_CALL( cudaMemcpyToSymbol( d_message_location_output_type, &h_message_location_output_type, sizeof(int)));
	
	
	//MAIN XMACHINE FUNCTION CALL (outputdata)
	//Reallocate   : false
	//Input        : 
	//Output       : location
	//Agent Output : 
	GPUFLAME_outputdata<<<grid, threads, sm_size>>>(d_Boids, d_locations);
	CUT_CHECK_ERROR("Kernel execution failed");
    
    
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	h_message_location_count += h_xmachine_memory_Boid_count;	
	//Copy count to device
	CUDA_SAFE_CALL( cudaMemcpyToSymbol( d_message_location_count, &h_message_location_count, sizeof(int)));	
	
	//HASH, SORT, REORDER AND BUILD PMB FOR SPATIAL PARTITIONING MESSAGE OUTPUTS
	//Get message hash values for sorting
	hash_location_messages<<<grid, threads>>>(d_xmachine_message_location_keys, d_xmachine_message_location_values, d_locations);
    CUT_CHECK_ERROR("Kernel execution failed");
    //Sort
    thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_message_location_keys),  thrust::device_pointer_cast(d_xmachine_message_location_keys) + h_message_location_count,  thrust::device_pointer_cast(d_xmachine_message_location_values));
    CUT_CHECK_ERROR("Kernel execution failed");
    //reorder and build pcb
    CUDA_SAFE_CALL(cudaMemset(d_location_partition_matrix->start, 0xffffffff, xmachine_message_location_grid_size* sizeof(int)));
	int reorder_sm_size = sizeof(unsigned int)*(THREADS_PER_TILE+1);
	reorder_location_messages<<<grid, threads, reorder_sm_size>>>(d_xmachine_message_location_keys, d_xmachine_message_location_values, d_location_partition_matrix, d_locations, d_locations_swap);
	CUT_CHECK_ERROR("Kernel execution failed");
	//swap ordered list
	xmachine_message_location_list* d_locations_temp = d_locations;
	d_locations = d_locations_swap;
	d_locations_swap = d_locations_temp;
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Boid_default_count+h_xmachine_memory_Boid_count > xmachine_memory_Boid_MAX){
		printf("Error: Buffer size of outputdata agents in state default will be exceeded moving working agents to next state in function outputdata\n");
		exit(0);
	}
	//append agents to next state list
	append_Boid_Agents<<<grid, threads>>>(d_Boids_default, d_Boids, h_xmachine_memory_Boid_default_count, h_xmachine_memory_Boid_count);
	CUT_CHECK_ERROR("Kernel execution failed");
	//update new state agent size
	h_xmachine_memory_Boid_default_count += h_xmachine_memory_Boid_count;
	CUDA_SAFE_CALL( cudaMemcpyToSymbol( d_xmachine_memory_Boid_default_count, &h_xmachine_memory_Boid_default_count, sizeof(int)));	
	
	
}



/** Boid_inputdata
 * Agent function prototype for inputdata function of Boid agent
 */
void Boid_inputdata(){
	dim3 grid;
	dim3 threads;
	int sm_size;
	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_Boid_default_count == 0)
	{
		return;
	}
	
	
	//SET GRID AND BLOCK SIZES
	//set tile size depending on agent count, set a 1d grid and block
	int tile_size = (int)ceil((float)h_xmachine_memory_Boid_default_count/THREADS_PER_TILE);
	grid.x = tile_size;
	threads.x = THREADS_PER_TILE;
	sm_size = SM_START;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_Boid_list* Boids_default_temp = d_Boids;
	d_Boids = d_Boids_default;
	d_Boids_default = Boids_default_temp;
	//set working count to current state count
	h_xmachine_memory_Boid_count = h_xmachine_memory_Boid_default_count;
	CUDA_SAFE_CALL( cudaMemcpyToSymbol( d_xmachine_memory_Boid_count, &h_xmachine_memory_Boid_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_Boid_default_count = 0;
	CUDA_SAFE_CALL( cudaMemcpyToSymbol( d_xmachine_memory_Boid_default_count, &h_xmachine_memory_Boid_default_count, sizeof(int)));	
	

	//******************************** AGENT FUNCTION *******************************

	
	//UPDATE SHARED MEMEORY SIZE FOR EACH FUNCTION INPUT
	//Continuous agent and message input is spatially partitioned
	sm_size += (threads.x * sizeof(xmachine_message_location));
	
    //all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (threads.x * PADDING);
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	//continuous agent with discrete or partitioned message input uses texture caching
	size_t tex_xmachine_message_location_x_byte_offset;    
    CUDA_SAFE_CALL( cudaBindTexture(&tex_xmachine_message_location_x_byte_offset, tex_xmachine_message_location_x, d_locations->x, sizeof(int)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_x_offset = (int)tex_xmachine_message_location_x_byte_offset / sizeof(float);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol( d_tex_xmachine_message_location_x_offset, &h_tex_xmachine_message_location_x_offset, sizeof(int)));
    size_t tex_xmachine_message_location_y_byte_offset;    
    CUDA_SAFE_CALL( cudaBindTexture(&tex_xmachine_message_location_y_byte_offset, tex_xmachine_message_location_y, d_locations->y, sizeof(int)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_y_offset = (int)tex_xmachine_message_location_y_byte_offset / sizeof(float);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol( d_tex_xmachine_message_location_y_offset, &h_tex_xmachine_message_location_y_offset, sizeof(int)));
    size_t tex_xmachine_message_location_z_byte_offset;    
    CUDA_SAFE_CALL( cudaBindTexture(&tex_xmachine_message_location_z_byte_offset, tex_xmachine_message_location_z, d_locations->z, sizeof(int)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_z_offset = (int)tex_xmachine_message_location_z_byte_offset / sizeof(float);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol( d_tex_xmachine_message_location_z_offset, &h_tex_xmachine_message_location_z_offset, sizeof(int)));
    size_t tex_xmachine_message_location_velX_byte_offset;    
    CUDA_SAFE_CALL( cudaBindTexture(&tex_xmachine_message_location_velX_byte_offset, tex_xmachine_message_location_velX, d_locations->velX, sizeof(int)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_velX_offset = (int)tex_xmachine_message_location_velX_byte_offset / sizeof(float);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol( d_tex_xmachine_message_location_velX_offset, &h_tex_xmachine_message_location_velX_offset, sizeof(int)));
    size_t tex_xmachine_message_location_velY_byte_offset;    
    CUDA_SAFE_CALL( cudaBindTexture(&tex_xmachine_message_location_velY_byte_offset, tex_xmachine_message_location_velY, d_locations->velY, sizeof(int)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_velY_offset = (int)tex_xmachine_message_location_velY_byte_offset / sizeof(float);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol( d_tex_xmachine_message_location_velY_offset, &h_tex_xmachine_message_location_velY_offset, sizeof(int)));
    size_t tex_xmachine_message_location_mass_byte_offset;    
    CUDA_SAFE_CALL( cudaBindTexture(&tex_xmachine_message_location_mass_byte_offset, tex_xmachine_message_location_mass, d_locations->mass, sizeof(int)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_mass_offset = (int)tex_xmachine_message_location_mass_byte_offset / sizeof(float);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol( d_tex_xmachine_message_location_mass_offset, &h_tex_xmachine_message_location_mass_offset, sizeof(int)));
    //bind pbm start and end indices to textures
    size_t tex_xmachine_message_location_pbm_start_byte_offset;
    size_t tex_xmachine_message_location_pbm_end_byte_offset;
    CUDA_SAFE_CALL( cudaBindTexture(&tex_xmachine_message_location_pbm_start_byte_offset, tex_xmachine_message_location_pbm_start, d_location_partition_matrix->start, sizeof(int)*xmachine_message_location_grid_size));
    h_tex_xmachine_message_location_pbm_start_offset = (int)tex_xmachine_message_location_pbm_start_byte_offset / sizeof(int);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol( d_tex_xmachine_message_location_pbm_start_offset, &h_tex_xmachine_message_location_pbm_start_offset, sizeof(int)));
    CUDA_SAFE_CALL( cudaBindTexture(&tex_xmachine_message_location_pbm_end_byte_offset, tex_xmachine_message_location_pbm_end, d_location_partition_matrix->end, sizeof(int)*xmachine_message_location_grid_size));
    h_tex_xmachine_message_location_pbm_end_offset = (int)tex_xmachine_message_location_pbm_end_byte_offset / sizeof(int);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol( d_tex_xmachine_message_location_pbm_end_offset, &h_tex_xmachine_message_location_pbm_end_offset, sizeof(int)));
    
	
	//MAIN XMACHINE FUNCTION CALL (inputdata)
	//Reallocate   : false
	//Input        : location
	//Output       : 
	//Agent Output : 
	GPUFLAME_inputdata<<<grid, threads, sm_size>>>(d_Boids, d_locations, d_location_partition_matrix);
	CUT_CHECK_ERROR("Kernel execution failed");
    
    
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	//continuous agent with discrete or partitioned message input uses texture caching
	CUDA_SAFE_CALL( cudaUnbindTexture(tex_xmachine_message_location_x));
	CUDA_SAFE_CALL( cudaUnbindTexture(tex_xmachine_message_location_y));
	CUDA_SAFE_CALL( cudaUnbindTexture(tex_xmachine_message_location_z));
	CUDA_SAFE_CALL( cudaUnbindTexture(tex_xmachine_message_location_velX));
	CUDA_SAFE_CALL( cudaUnbindTexture(tex_xmachine_message_location_velY));
	CUDA_SAFE_CALL( cudaUnbindTexture(tex_xmachine_message_location_mass));
	//unbind pbm indices
    CUDA_SAFE_CALL( cudaUnbindTexture(tex_xmachine_message_location_pbm_start));
    CUDA_SAFE_CALL( cudaUnbindTexture(tex_xmachine_message_location_pbm_end));
    
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Boid_default_count+h_xmachine_memory_Boid_count > xmachine_memory_Boid_MAX){
		printf("Error: Buffer size of inputdata agents in state default will be exceeded moving working agents to next state in function inputdata\n");
		exit(0);
	}
	//append agents to next state list
	append_Boid_Agents<<<grid, threads>>>(d_Boids_default, d_Boids, h_xmachine_memory_Boid_default_count, h_xmachine_memory_Boid_count);
	CUT_CHECK_ERROR("Kernel execution failed");
	//update new state agent size
	h_xmachine_memory_Boid_default_count += h_xmachine_memory_Boid_count;
	CUDA_SAFE_CALL( cudaMemcpyToSymbol( d_xmachine_memory_Boid_default_count, &h_xmachine_memory_Boid_default_count, sizeof(int)));	
	
	
}


 
extern "C" void reset_Boid_default_count()
{
    h_xmachine_memory_Boid_default_count = 0;
}
