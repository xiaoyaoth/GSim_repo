

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

#ifndef _FLAMEGPU_KERNELS_H_
#define _FLAMEGPU_KERNELS_H_

#include <header.h>


/* Agent count constants */

__constant__ int d_xmachine_memory_Boid_count;

/* Agent state count constants */

__constant__ int d_xmachine_memory_Boid_default_count;


/* Message constants */

/* location Message variables */
/* Non partitioned and spatial partitioned message variables  */
__constant__ int d_message_location_count;         /**< message list counter*/
__constant__ int d_message_location_output_type;   /**< message output type (single or optional)*/
//Spatial Partitioning Variables
__constant__ float3 d_message_location_min_bounds;           /**< min bounds (x,y,z) of partitioning environment */
__constant__ float3 d_message_location_max_bounds;           /**< max bounds (x,y,z) of partitioning environment */
__constant__ int3 d_message_location_partitionDim;           /**< partition dimensions (x,y,z) of partitioning environment */
__constant__ float d_message_location_radius;                 /**< partition radius (used to determin the size of the partitions) */

	
    
//include each function file

#include "functions.c"
    
/* Texture bindings */
/* location Message Bindings */texture<float, 1, cudaReadModeElementType> tex_xmachine_message_location_x;
__constant__ int d_tex_xmachine_message_location_x_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_location_y;
__constant__ int d_tex_xmachine_message_location_y_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_location_z;
__constant__ int d_tex_xmachine_message_location_z_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_location_velX;
__constant__ int d_tex_xmachine_message_location_velX_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_location_velY;
__constant__ int d_tex_xmachine_message_location_velY_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_location_mass;
__constant__ int d_tex_xmachine_message_location_mass_offset;
texture<int, 1, cudaReadModeElementType> tex_xmachine_message_location_pbm_start;
__constant__ int d_tex_xmachine_message_location_pbm_start_offset;
texture<int, 1, cudaReadModeElementType> tex_xmachine_message_location_pbm_end;
__constant__ int d_tex_xmachine_message_location_pbm_end_offset;


    
#define WRAP(x,m) (((x)<m)?(x):(x%m)) /**< Simple wrap */
#define sWRAP(x,m) (((x)<m)?(((x)<0)?(m+(x)):(x)):(m-(x))) /**<signed integer wrap (no modulus) for negatives where 2m > |x| > m */

//PADDING WILL ONLY AVOID SM CONFLICTS FOR 32BIT
//SM_OFFSET REQUIRED AS FERMI STARTS INDEXING MEMORY FROM LOCATION 0 (i.e. NULL)??
__constant__ int d_SM_START;
__constant__ int d_PADDING;

//SM addressing macro to avoid conflicts (32 bit only)
#define SHARE_INDEX(i, s) (__mul24((s + d_PADDING), i)+d_SM_START) /**<offset struct size by padding to avoid bank conflicts */

//if doubel support is needed then define the following function which requires sm_13 or later
#ifdef _DOUBLE_SUPPORT_REQUIRED_
__inline__ __device__ double tex1DfetchDouble(texture<int2, 1, cudaReadModeElementType> tex, int i)
{
	int2 v = tex1Dfetch(tex, i);
  //IF YOU HAVE AN ERROR HERE THEN YOU ARE USING DOUBLE VALUES IN AGENT MEMORY AND NOT COMPILING FOR DOUBLE SUPPORTED HARDWARE
  //To compile for double supported hardware change the CUDA Build rule property "Use sm_13 Architecture (double support)" on the CUDA-Specific Propert Page of the CUDA Build Rule for simulation.cu
	return __hiloint2double(v.y, v.x);
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Helper functions */
/** next_cell
 * Function used for finding the next cell when using spatial partitioning
 * Upddates the relative cell variable which can have value of -1, 0 or +1
 * @param relative_cell pointer to the relative cell position
 * @return boolean if there is a next cell. True unless relative_Cell value was 1,1,1
 */
__device__ int next_cell3D(int3* relative_cell)
{
	if (relative_cell->x < 1)
	{
		relative_cell->x++;
		return true;
	}
	relative_cell->x = -1;

	if (relative_cell->y < 1)
	{
		relative_cell->y++;
		return true;
	}
	relative_cell->y = -1;
	
	if (relative_cell->z < 1)
	{
		relative_cell->z++;
		return true;
	}
	relative_cell->z = -1;
	
	return false;
}

/** next_cell2D
 * Function used for finding the next cell when using spatial partitioning. Z component is ignored
 * Upddates the relative cell variable which can have value of -1, 0 or +1
 * @param relative_cell pointer to the relative cell position
 * @return boolean if there is a next cell. True unless relative_Cell value was 1,1
 */
__device__ int next_cell2D(int3* relative_cell)
{
	if (relative_cell->x < 1)
	{
		relative_cell->x++;
		return true;
	}
	relative_cell->x = -1;

	if (relative_cell->y < 1)
	{
		relative_cell->y++;
		return true;
	}
	relative_cell->y = -1;
	
	return false;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dyanamically created Boid agent functions */

/** reset_Boid_scan_input
 * Boid agent reset scan input function
 * @param agents The xmachine_memory_Boid_list agent list
 */
__global__ void reset_Boid_scan_input(xmachine_memory_Boid_list* agents){

	//global thread index
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	agents->_position[index] = 0;
	agents->_scan_input[index] = 0;
}



/** scatter_Boid_Agents
 * Boid scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_Boid_list agent list destination
 * @param agents_src xmachine_memory_Boid_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void scatter_Boid_Agents(xmachine_memory_Boid_list* agents_dst, xmachine_memory_Boid_list* agents_src, int dst_agent_count, int number_to_scatter){
	//global thread index
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	int _scan_input = agents_src->_scan_input[index];

	//if optional message is to be written. 
	//must check agent is within number to scatter as unused threads may have scan input = 1
	if ((_scan_input == 1)&&(index < number_to_scatter)){
		int output_index = agents_src->_position[index] + dst_agent_count;

		//AoS - xmachine_message_location Un-Coalesced scattered memory write     
        agents_dst->_position[output_index] = output_index;        
		agents_dst->id[output_index] = agents_src->id[index];        
		agents_dst->x[output_index] = agents_src->x[index];        
		agents_dst->y[output_index] = agents_src->y[index];        
		agents_dst->goalX[output_index] = agents_src->goalX[index];        
		agents_dst->goalY[output_index] = agents_src->goalY[index];        
		agents_dst->velX[output_index] = agents_src->velX[index];        
		agents_dst->velY[output_index] = agents_src->velY[index];        
		agents_dst->velMax[output_index] = agents_src->velMax[index];        
		agents_dst->mass[output_index] = agents_src->mass[index];
	}
}

/** append_Boid_Agents
 * Boid scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_Boid_list agent list destination
 * @param agents_src xmachine_memory_Boid_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void append_Boid_Agents(xmachine_memory_Boid_list* agents_dst, xmachine_memory_Boid_list* agents_src, int dst_agent_count, int number_to_append){
	//global thread index
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	//must check agent is within number to append as unused threads may have scan input = 1
    if (index < number_to_append){
	    int output_index = index + dst_agent_count;

	    //AoS - xmachine_message_location Un-Coalesced scattered memory write
	    agents_dst->_position[output_index] = output_index;
	    agents_dst->id[output_index] = agents_src->id[index];
	    agents_dst->x[output_index] = agents_src->x[index];
	    agents_dst->y[output_index] = agents_src->y[index];
	    agents_dst->goalX[output_index] = agents_src->goalX[index];
	    agents_dst->goalY[output_index] = agents_src->goalY[index];
	    agents_dst->velX[output_index] = agents_src->velX[index];
	    agents_dst->velY[output_index] = agents_src->velY[index];
	    agents_dst->velMax[output_index] = agents_src->velMax[index];
	    agents_dst->mass[output_index] = agents_src->mass[index];
    }
}

/** add_Boid_agent
 * Continuous Boid agent add agent function writes agent data to agent swap
 * @param agents xmachine_memory_Boid_list to add agents to 
 * @param id agent variable of type int
 * @param x agent variable of type float
 * @param y agent variable of type float
 * @param goalX agent variable of type float
 * @param goalY agent variable of type float
 * @param velX agent variable of type float
 * @param velY agent variable of type float
 * @param velMax agent variable of type float
 * @param mass agent variable of type float
 */
template <int AGENT_TYPE>
__device__ void add_Boid_agent(xmachine_memory_Boid_list* agents, int id, float x, float y, float goalX, float goalY, float velX, float velY, float velMax, float mass){
	
	int index;
    
    //calculate the agents index in global agent list (depends on agent type)
	if (AGENT_TYPE == DISCRETE_2D){
		int width = __mul24(blockDim.x, gridDim.x);
		int2 global_position;
		global_position.x = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
		global_position.y = __mul24(blockIdx.y,blockDim.y) + threadIdx.y;
		index = global_position.x + __mul24(global_position.y, width);
	}else//AGENT_TYPE == CONTINOUS
		index = threadIdx.x + blockIdx.x*blockDim.x;

	//for prefix sum
	agents->_position[index] = 0;
	agents->_scan_input[index] = 1;

	//write data to new buffer
	agents->id[index] = id;
	agents->x[index] = x;
	agents->y[index] = y;
	agents->goalX[index] = goalX;
	agents->goalY[index] = goalY;
	agents->velX[index] = velX;
	agents->velY[index] = velY;
	agents->velMax[index] = velMax;
	agents->mass[index] = mass;

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_Boid_agent(xmachine_memory_Boid_list* agents, int id, float x, float y, float goalX, float goalY, float velX, float velY, float velMax, float mass){
    add_Boid_agent<DISCRETE_2D>(agents, id, x, y, goalX, goalY, velX, velY, velMax, mass);
}

/** reorder_Boid_agents
 * Continuous Boid agent areorder function used after key value pairs have been sorted
 * @param values sorted index values
 * @param unordered_agents list of unordered agents
 * @ param ordered_agents list used to output ordered agents
 */
__global__ void reorder_Boid_agents(unsigned int* values, xmachine_memory_Boid_list* unordered_agents, xmachine_memory_Boid_list* ordered_agents)
{
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	uint old_pos = values[index];

	//reorder agent data
	ordered_agents->id[index] = unordered_agents->id[old_pos];
	ordered_agents->x[index] = unordered_agents->x[old_pos];
	ordered_agents->y[index] = unordered_agents->y[old_pos];
	ordered_agents->goalX[index] = unordered_agents->goalX[old_pos];
	ordered_agents->goalY[index] = unordered_agents->goalY[old_pos];
	ordered_agents->velX[index] = unordered_agents->velX[old_pos];
	ordered_agents->velY[index] = unordered_agents->velY[old_pos];
	ordered_agents->velMax[index] = unordered_agents->velMax[old_pos];
	ordered_agents->mass[index] = unordered_agents->mass[old_pos];
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dyanamically created location message functions */


/** add_location_message
 * Add non partitioned or spatially partitioned location message
 * @param messages xmachine_message_location_list message list to add too
 * @param x agent variable of type float
 * @param y agent variable of type float
 * @param z agent variable of type float
 * @param velX agent variable of type float
 * @param velY agent variable of type float
 * @param mass agent variable of type float
 */
__device__ void add_location_message(xmachine_message_location_list* messages, float x, float y, float z, float velX, float velY, float mass){

	//global thread index
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x + d_message_location_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_location_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_location_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_location Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->x[index] = x;
	messages->y[index] = y;
	messages->z[index] = z;
	messages->velX[index] = velX;
	messages->velY[index] = velY;
	messages->mass[index] = mass;

}

/**
 * Scatter non partitioned or spatially partitioned location message (for optional messages)
 * @param messages scatter_optional_location_messages Sparse xmachine_message_location_list message list
 * @param message_swap temp xmachine_message_location_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_location_messages(xmachine_message_location_list* messages, xmachine_message_location_list* messages_swap){
	//global thread index
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_location_count;

		//AoS - xmachine_message_location Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->x[output_index] = messages_swap->x[index];
		messages->y[output_index] = messages_swap->y[index];
		messages->z[output_index] = messages_swap->z[index];
		messages->velX[output_index] = messages_swap->velX[index];
		messages->velY[output_index] = messages_swap->velY[index];
		messages->mass[output_index] = messages_swap->mass[index];				
	}
}

/** reset_location_swaps
 * Reset non partitioned or spatially partitioned location message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_location_swaps(xmachine_message_location_list* messages_swap){

	//global thread index
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

/** message_location_grid_position
 * Calculates the grid cell position given an float3 vector
 * @param position float3 vector representing a position
 */
__device__ int3 message_location_grid_position(float3 position)
{
    int3 gridPos;
    gridPos.x = floor((position.x - d_message_location_min_bounds.x) * (float)d_message_location_partitionDim.x / (d_message_location_max_bounds.x - d_message_location_min_bounds.x));
    gridPos.y = floor((position.y - d_message_location_min_bounds.y) * (float)d_message_location_partitionDim.y / (d_message_location_max_bounds.y - d_message_location_min_bounds.y));
    gridPos.z = floor((position.z - d_message_location_min_bounds.z) * (float)d_message_location_partitionDim.z / (d_message_location_max_bounds.z - d_message_location_min_bounds.z));

	//do wrapping or bounding
	

    return gridPos;
}

/** message_location_hash
 * Given the grid position in partition space this function calculates a hash value
 * @param gridPos The position in partition space
 */
__device__ unsigned int message_location_hash(int3 gridPos)
{
	//cheap bounding without mod (within range +- partition dimension)
	gridPos.x = (gridPos.x<0)? d_message_location_partitionDim.x-1: gridPos.x; 
	gridPos.x = (gridPos.x>=d_message_location_partitionDim.x)? 0 : gridPos.x; 
	gridPos.y = (gridPos.y<0)? d_message_location_partitionDim.y-1 : gridPos.y; 
	gridPos.y = (gridPos.y>=d_message_location_partitionDim.y)? 0 : gridPos.y; 
	gridPos.z = (gridPos.z<0)? d_message_location_partitionDim.z-1: gridPos.z; 
	gridPos.z = (gridPos.z>=d_message_location_partitionDim.z)? 0 : gridPos.z; 

	//unique id
	return __umul24(__umul24(gridPos.z, d_message_location_partitionDim.y), d_message_location_partitionDim.x) + __umul24(gridPos.y, d_message_location_partitionDim.x) + gridPos.x;
}

/** hash_location_messages
 * Kernal function for calculating a hash value for each messahe depending on its position
 * @param keys output for the hash key
 * @param values output for the index value
 * @param messages the message list used to generate the hash value outputs
 */
__global__ void hash_location_messages(uint* keys, uint* values, xmachine_message_location_list* messages)
{
	unsigned int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	float3 position = make_float3(messages->x[index], messages->y[index], messages->z[index]);
	int3 grid_position = message_location_grid_position(position);
	unsigned int hash = message_location_hash(grid_position);

	keys[index] = hash;
  values[index] = index;
}

/** reorder_location_messages
 * Reorders the messages accoring to the ordered sort identifiers and builds a Partition Boundary Matrix by looking at the previosu threads sort id.
 * @param keys the sorted hash keys
 * @param values the sorted index values
 * @param matrix the PBM
 * @param unordered_messages the original unordered message data
 * @param ordered_messages buffer used to scatter messages into the correct order
 */
__global__ void reorder_location_messages(uint* keys, uint* values, xmachine_message_location_PBM* matrix, xmachine_message_location_list* unordered_messages, xmachine_message_location_list* ordered_messages)
{
	extern __shared__ int sm_data [];

	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	//load threads sort key into sm
	uint key = keys[index];
	uint old_pos = values[index];

	sm_data[threadIdx.x] = key;
	__syncthreads();
	
	unsigned int prev_key;

	//if first thread then no prev sm value so get prev from global memory 
	if (threadIdx.x == 0)
	{
		//first thread has no prev value so ignore
		if (index != 0)
			prev_key = keys[index-1];
	}
	//get previous ident from sm
	else	
	{
		prev_key = sm_data[threadIdx.x-1];
	}

	//TODO: Check key is not out of bounds

	//set partition boundaries
	if (index < d_message_location_count)
	{
		//if first thread then set first partition cell start
		if (index == 0)
		{
			matrix->start[key] = index;
		}

		//if edge of a boundr update start and end of partition
		else if (prev_key != key)
		{
			//set start for key
			matrix->start[key] = index;

			//set end for key -1
			matrix->end[prev_key] = index;
		}

		//if last thread then set final partition cell end
		if (index == d_message_location_count-1)
		{
			matrix->end[key] = index+1;
		}
	}
	
	//finally reorder agent data
	ordered_messages->x[index] = unordered_messages->x[old_pos];
	ordered_messages->y[index] = unordered_messages->y[old_pos];
	ordered_messages->z[index] = unordered_messages->z[old_pos];
	ordered_messages->velX[index] = unordered_messages->velX[old_pos];
	ordered_messages->velY[index] = unordered_messages->velY[old_pos];
	ordered_messages->mass[index] = unordered_messages->mass[old_pos];
}

/** load_next_location_message
 * Used to load the next message data to shared memory
 * Idea is check the current cell index to see if we can simpley get a message from the current cell
 * If we are at the end of the current cell then loop till we find the next cell with messages (this way we ignore cells with no messages)
 * @param messages the message list
 * @param partition_matrix the PBM
 * @param relative_cell the relative partition cell position from the agent position
 * @param cell_index_max the maximum index of the currnt partition cell
 * @param agent_grid_cell the agents partition cell position
 * @param cell_index the current cell index in agent_grid_cell+relative_cell
 * @return true if a messag has been loaded into sm false otherwise
 */
__device__ int load_next_location_message(xmachine_message_location_list* messages, xmachine_message_location_PBM* partition_matrix, int3 relative_cell, int cell_index_max, int3 agent_grid_cell, int cell_index)
{
	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];

	int move_cell = true;
	cell_index ++;

	//see if we need to move to a new partition cell
	if(cell_index < cell_index_max)
		move_cell = false;

	while(move_cell)
	{
		//get the next relative grid position 
        if (next_cell3D(&relative_cell))
		{
			//calculate the next cells grid position and hash
			int3 next_cell_position = agent_grid_cell + relative_cell;
			int next_cell_hash = message_location_hash(next_cell_position);
			//use the hash to calculate the start index
			int cell_index_min = tex1Dfetch(tex_xmachine_message_location_pbm_start, next_cell_hash + d_tex_xmachine_message_location_pbm_start_offset);

			//check for messages in the cell (empty cells with have a start index of baadf00d
			if (cell_index_min != 0xffffffff)
			{
				//if there are messages in the cell then update the cell index max value
				cell_index_max = tex1Dfetch(tex_xmachine_message_location_pbm_end, next_cell_hash + d_tex_xmachine_message_location_pbm_end_offset);
				//start from the cell index min
				cell_index = cell_index_min;
				//exit the loop as we have found a valid cell with message data
				move_cell = false;
			}
		}
		else
		{
			//we have exhausted all the neightbouring cells so there are no more messages
			return false;
		}
	}
	
	//get message data using texture fetch
	xmachine_message_location temp_message;
	temp_message._relative_cell = relative_cell;
	temp_message._cell_index_max = cell_index_max;
	temp_message._cell_index = cell_index;
	temp_message._agent_grid_cell = agent_grid_cell;

	//Using texture cache
  temp_message.x = tex1Dfetch(tex_xmachine_message_location_x, cell_index + d_tex_xmachine_message_location_x_offset); temp_message.y = tex1Dfetch(tex_xmachine_message_location_y, cell_index + d_tex_xmachine_message_location_y_offset); temp_message.z = tex1Dfetch(tex_xmachine_message_location_z, cell_index + d_tex_xmachine_message_location_z_offset); temp_message.velX = tex1Dfetch(tex_xmachine_message_location_velX, cell_index + d_tex_xmachine_message_location_velX_offset); temp_message.velY = tex1Dfetch(tex_xmachine_message_location_velY, cell_index + d_tex_xmachine_message_location_velY_offset); temp_message.mass = tex1Dfetch(tex_xmachine_message_location_mass, cell_index + d_tex_xmachine_message_location_mass_offset); 

	//load it into shared memory (no sync as no sharing between threads)
	int message_index = SHARE_INDEX(threadIdx.x, sizeof(xmachine_message_location));
	xmachine_message_location* sm_message = ((xmachine_message_location*)&message_share[message_index]);
	sm_message[0] = temp_message;

	return true;
}

/*
 * get first non partitioned location message (first batch load into shared memory)
 */
__device__ xmachine_message_location* get_first_location_message(xmachine_message_location_list* messages, xmachine_message_location_PBM* partition_matrix, float x, float y, float z){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];

	int3 relative_cell = make_int3(-2, -1, -1);
	int cell_index_max = 0;
	int cell_index = 0;
	float3 position = make_float3(x, y, z);
	int3 agent_grid_cell = message_location_grid_position(position);
	
	if (load_next_location_message(messages, partition_matrix, relative_cell, cell_index_max, agent_grid_cell, cell_index))
	{
		int message_index = SHARE_INDEX(threadIdx.x, sizeof(xmachine_message_location));
		return ((xmachine_message_location*)&message_share[message_index]);
	}
	else
	{
		return false;
	}
}

/*
 * get next non partitioned location message (either from SM or next batch load)
 */
__device__ xmachine_message_location* get_next_location_message(xmachine_message_location* message, xmachine_message_location_list* messages, xmachine_message_location_PBM* partition_matrix){
	
	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//TODO: check message count
	
	if (load_next_location_message(messages, partition_matrix, message->_relative_cell, message->_cell_index_max, message->_agent_grid_cell, message->_cell_index))
	{
		//get conflict free address of 
		int message_index = SHARE_INDEX(threadIdx.x, sizeof(xmachine_message_location));
		return ((xmachine_message_location*)&message_share[message_index]);
	}
	else
		return false;
	
}



	
/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dyanamically created GPU kernals  */



/**
 *
 */
__global__ void GPUFLAME_outputdata(xmachine_memory_Boid_list* agents, xmachine_message_location_list* location_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_Boid_count)
        return;
    

	//SoA to AoS - xmachine_memory_outputdata Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_Boid agent;
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.goalX = agents->goalX[index];
	agent.goalY = agents->goalY[index];
	agent.velX = agents->velX[index];
	agent.velY = agents->velY[index];
	agent.velMax = agents->velMax[index];
	agent.mass = agents->mass[index];

	//FLAME function call
	int dead = !outputdata(&agent, location_messages	);
	
	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_outputdata Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->goalX[index] = agent.goalX;
	agents->goalY[index] = agent.goalY;
	agents->velX[index] = agent.velX;
	agents->velY[index] = agent.velY;
	agents->velMax[index] = agent.velMax;
	agents->mass[index] = agent.mass;
}

/**
 *
 */
__global__ void GPUFLAME_inputdata(xmachine_memory_Boid_list* agents, xmachine_message_location_list* location_messages, xmachine_message_location_PBM* partition_matrix){
	
	//continuous agent: index is agent position in 1D agent list
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_Boid_count)
        return;
    

	//SoA to AoS - xmachine_memory_inputdata Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_Boid agent;
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.goalX = agents->goalX[index];
	agent.goalY = agents->goalY[index];
	agent.velX = agents->velX[index];
	agent.velY = agents->velY[index];
	agent.velMax = agents->velMax[index];
	agent.mass = agents->mass[index];

	//FLAME function call
	int dead = !inputdata(&agent, location_messages, partition_matrix);
	
	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_inputdata Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->goalX[index] = agent.goalX;
	agents->goalY[index] = agent.goalY;
	agents->velX[index] = agent.velX;
	agents->velY[index] = agent.velY;
	agents->velMax[index] = agent.velMax;
	agents->mass[index] = agent.mass;
}

	
	
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Rand48 functions */

__device__ static uint2 RNG_rand48_iterate_single(uint2 Xn, uint2 A, uint2 C)
{
	unsigned int R0, R1;

	// low 24-bit multiplication
	const unsigned int lo00 = __umul24(Xn.x, A.x);
	const unsigned int hi00 = __umulhi(Xn.x, A.x);

	// 24bit distribution of 32bit multiplication results
	R0 = (lo00 & 0xFFFFFF);
	R1 = (lo00 >> 24) | (hi00 << 8);

	R0 += C.x; R1 += C.y;

	// transfer overflows
	R1 += (R0 >> 24);
	R0 &= 0xFFFFFF;

	// cross-terms, low/hi 24-bit multiplication
	R1 += __umul24(Xn.y, A.x);
	R1 += __umul24(Xn.x, A.y);

	R1 &= 0xFFFFFF;

	return make_uint2(R0, R1);
}

//Templated function
template <int AGENT_TYPE>
__device__ float rnd(RNG_rand48* rand48){

	int index;
	
	//calculate the agents index in global agent list
	if (AGENT_TYPE == DISCRETE_2D){
		int width = __mul24(blockDim.x, gridDim.x);
		int2 global_position;
		global_position.x = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
		global_position.y = __mul24(blockIdx.y,blockDim.y) + threadIdx.y;
		index = global_position.x + __mul24(global_position.y, width);
	}else//AGENT_TYPE == CONTINOUS
		index = threadIdx.x + blockIdx.x*blockDim.x;

	uint2 state = rand48->seeds[index];
	uint2 A = rand48->A;
	uint2 C = rand48->C;

	int rand = ( state.x >> 17 ) | ( state.y << 7);

	// this actually iterates the RNG
	state = RNG_rand48_iterate_single(state, A, C);

	rand48->seeds[index] = state;

	return (float)rand/2147483647;
}

__device__ float rnd(RNG_rand48* rand48){
	return rnd<DISCRETE_2D>(rand48);
}

#endif //_FLAMEGPU_KERNELS_H_
