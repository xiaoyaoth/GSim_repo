
/*
* Copyright 2011 University of Sheffield.
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

#ifndef _FLAMEGPU_FUNCTIONS
#define _FLAMEGPU_FUNCTIONS

#include <header.h>


/**
* outputdata FLAMEGPU Agent Function
* Automatically generated using functions.xslt
* @param agent Pointer to an agent structre of type xmachine_memory_Boid. This represents a single agent instance and can be modified directly.
* @param location_messages Pointer to output message list of type xmachine_message_location_list. Must be passed as an argument to the add_location_message function ??.
*/
__FLAME_GPU_FUNC__ int outputdata(xmachine_memory_Boid* agent, xmachine_message_location_list* location_messages){

	add_location_message(location_messages, agent->x, agent->y, 0, agent->velX, agent->velY, agent->mass);
	/* //Template for message output function use 
	* 
	* float x = 0;
	* float y = 0;
	* float velX = 0;
	* float velY = 0;
	* add_location_message(location_messages, x, y, velX, velY);
	*/



	return 0;
}

/**
* inputdata FLAMEGPU Agent Function
* Automatically generated using functions.xslt
* @param agent Pointer to an agent structre of type xmachine_memory_Boid. This represents a single agent instance and can be modified directly.
* @param location_messages  location_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_location_message and get_next_location_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_location_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.
*/

#define	tao 0.5
#define	AA 2000
#define	BB 0.1
#define	k1 (1.2 * 100000)
#define k2 (2.4 * 100000)
#define	maxv 3
#define cMass 100
#define WIDTH_D 100
#define HEIGHT_D 100

__device__ void computeSocialForce(const float2& loc, const float2 &velo, const float & mass, const xmachine_message_location &otherData, float2 &fSum){
	//other's data
	const float2& locOther = make_float2(otherData.x, otherData.y);
	const float2& veloOther = make_float2(otherData.velX, otherData.velY);
	const float& massOther = otherData.mass;

	float d = 1e-15 + sqrt((loc.x - locOther.x) * (loc.x - locOther.x) + (loc.y - locOther.y) * (loc.y - locOther.y));
	float dDelta = mass / cMass + massOther / cMass - d;
	float fExp = AA * exp(dDelta / BB);
	float fKg = dDelta < 0 ? 0 : k1 *dDelta;
	float nijx = (loc.x - locOther.x) / d;
	float nijy = (loc.y - locOther.y) / d;
	float fnijx = (fExp + fKg) * nijx;
	float fnijy = (fExp + fKg) * nijy;
	float fkgx = 0;
	float fkgy = 0;
	if (dDelta > 0) {
		float tix = - nijy;
		float tiy = nijx;
		fkgx = k2 * dDelta;
		fkgy = k2 * dDelta;
		float vijDelta = (veloOther.x - velo.x) * tix + (veloOther.y - velo.y) * tiy;
		fkgx = fkgx * vijDelta * tix;
		fkgy = fkgy * vijDelta * tiy;
	}
	fSum.x += fnijx + fkgx;
	fSum.y += fnijy + fkgy;
}

__device__ float correctCrossBoader(float val, float limit)
{
	if (val > limit)
		return limit-1;
	else if (val < 0)
		return 0;
	return val;
}

__FLAME_GPU_FUNC__ int inputdata(xmachine_memory_Boid* agent, xmachine_message_location_list* location_messages, xmachine_message_location_PBM* partition_matrix){

	float2 loc = make_float2(agent->x, agent->y);
	float2 goal = make_float2(agent->goalX, agent->goalY);
	float2 velo = make_float2(agent->velX, agent->velY);
	float v0 = agent->velMax;
	float mass = agent->mass;

	//compute the direction
	float2 dvt;	dvt.x = 0;	dvt.y = 0;
	float2 diff; diff.x = 0; diff.y = 0;
	float d0 = sqrt((loc.x - goal.x) * (loc.x - goal.x) + (loc.y - goal.y) * (loc.y - goal.y));
	diff.x = v0 * (goal.x - loc.x) / d0;
	diff.y = v0 * (goal.y - loc.y) / d0;
	dvt.x = (diff.x - velo.x) / tao;
	dvt.y = (diff.y - velo.y) / tao;


	//compute force with other agents
	float2 fSum = make_float2(0,0);
	int counter = 0;
	xmachine_message_location* current_message = get_first_location_message(location_messages, partition_matrix, loc.x, loc.y, 0);
	while (current_message) {
		xmachine_message_location otherDataLocal = *current_message;
		float2 otherLoc = make_float2(otherDataLocal.x, otherDataLocal.y);
		float ds = length(otherLoc - loc);
		if (ds < 6 && ds > 0) {
			counter++;
			computeSocialForce(loc, velo, mass, otherDataLocal, fSum);
		}
		current_message = get_next_location_message(current_message, location_messages, partition_matrix);
	}

	//compute force with wall
	for (int wallIdx = 0; wallIdx < obsLineNum; wallIdx++) {
		float diw, crx, cry;
		diw = obsLines[wallIdx].pointToLineDist(loc, crx, cry);
		float virDiw = DIST(loc.x, loc.y, crx, cry);
		float niwx = (loc.x - crx) / virDiw;
		float niwy = (loc.y - cry) / virDiw;
		float drw = mass / cMass - diw;
		float fiw1 = AA * exp(drw / BB);
		if (drw > 0)
			fiw1 += k1 * drw;
		float fniwx = fiw1 * niwx;
		float fniwy = fiw1 * niwy;

		float fiwKgx = 0, fiwKgy = 0;
		if (drw > 0)
		{
			float fiwKg = k2 * drw * (velo.x * (-niwy) + velo.y * niwx);
			fiwKgx = fiwKg * (-niwy);
			fiwKgy = fiwKg * niwx;
		}

		fSum.x += fniwx - fiwKgx;
		fSum.y += fniwy - fiwKgy;
	}

	//sum up
	dvt.x += fSum.x / mass;
	dvt.y += fSum.y / mass;

	float2 newVelo = velo;
	float2 newLoc = loc;
	float2 newGoal = goal;
	float tick = 0.1;
	newVelo.x += dvt.x * tick; // * (1 + this->random->gaussian() * 0.1);
	newVelo.y += dvt.y * tick; // * (1 + this->random->gaussian() * 0.1);
	float dv = sqrt(newVelo.x * newVelo.x + newVelo.y * newVelo.y);

	if (dv > maxv) {
		newVelo.x = newVelo.x * maxv / dv;
		newVelo.y = newVelo.y * maxv / dv;
	}

	float mint = 1;
	for (int wallIdx = 0; wallIdx < obsLineNum; wallIdx++) 
	{
		float crx, cry, tt;
		int ret = obsLines[wallIdx].intersection2LineSeg(
			loc.x, 
			loc.y, 
			loc.x + 0.5 * newVelo.x * tick,
			loc.y + 0.5 * newVelo.y * tick,
			crx,
			cry
			);
		if (ret == 1) 
		{
			if (fabs(crx - loc.x) > 0)
				tt = (crx - loc.x) / (newVelo.x * tick);
			else
				tt = (crx - loc.y) / (newVelo.y * tick + 1e-20);
			if (tt < mint)
				mint = tt;
		}
	}

	newVelo.x *= mint;
	newVelo.y *= mint;
	newLoc.x += newVelo.x * tick;
	newLoc.y += newVelo.y * tick;

	if ((newLoc.x - mass/cMass <= 0.25 * WIDTH_D) && (newLoc.y - mass/cMass > 0.5 * HEIGHT_D - 2) && (newLoc.y - mass/cMass < 0.5 * HEIGHT_D + 1)) 
	{
		newGoal.x = 0;
	}
loc.x;loc.y;
	newLoc.x = correctCrossBoader(newLoc.x, WIDTH_D);
	newLoc.y = correctCrossBoader(newLoc.y, HEIGHT_D);

	agent->x = newLoc.x;
	agent->y = newLoc.y;
	agent->velX = newVelo.x;
	agent->velY = newVelo.y;
	agent->goalX = newGoal.x;
	agent->goalY = newGoal.y;

	return 0;
}




#endif //_FLAMEGPU_FUNCTIONS
