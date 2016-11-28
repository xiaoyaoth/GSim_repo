
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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cutil.h>
#include <limits.h>

// include header
#include <header.h>

float3 agent_maximum;
float3 agent_minimum;

void readIntArrayInput(char* buffer, int *array, unsigned int expected_items){
    unsigned int i = 0;
    const char s[2] = ",";
    char * token;

    token = strtok(buffer, s);
    while (token != NULL){
        if (i>=expected_items){
            printf("Error: Agent Memeory array has too many items, expected %d!\n", expected_items);
            exit(0);
        }
        
        array[i++] = atoi(token);
        
        token = strtok(NULL, s);
    }
    if (i != expected_items){
        printf("Error: Agent Memeory array has %d items, expected %d!\n", i, expected_items);
        exit(0);
    }
}

void readFloatArrayInput(char* buffer, float *array, unsigned int expected_items){
    unsigned int i = 0;
    const char s[2] = ",";
    char * token;

    token = strtok(buffer, s);
    while (token != NULL){
        if (i>=expected_items){
            printf("Error: Agent Memeory array has too many items, expected %d!\n", expected_items);
            exit(0);
        }
        
        array[i++] = atof(token);
        
        token = strtok(NULL, s);
    }
    if (i != expected_items){
        printf("Error: Agent Memeory array has %d items, expected %d!\n", i, expected_items);
        exit(0);
    }
}

void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_Boid_list* h_Boids_default, xmachine_memory_Boid_list* d_Boids_default, int h_xmachine_memory_Boid_default_count)
{
	//Device to host memory transfer
	
	CUDA_SAFE_CALL( cudaMemcpy( h_Boids_default, d_Boids_default, sizeof(xmachine_memory_Boid_list), cudaMemcpyDeviceToHost));
	
	/* Pointer to file */
	FILE *file;
	char data[100];

	sprintf(data, "%s%i.xml", outputpath, iteration_number);
	//printf("Writing itteration %i data to %s\n", iteration_number, data);
	file = fopen(data, "w");
	fputs("<states>\n<itno>", file);
	sprintf(data, "%i", iteration_number);
	fputs(data, file);
	fputs("</itno>\n", file);
	fputs("<environment>\n" , file);
	fputs("</environment>\n" , file);

	//Write each Boid agent to xml
	for (int i=0; i<h_xmachine_memory_Boid_default_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>Boid</name>\n", file);
        
		fputs("<id>", file);
        sprintf(data, "%i", h_Boids_default->id[i]);
		fputs(data, file);
		fputs("</id>\n", file);
        
		fputs("<x>", file);
        sprintf(data, "%f", h_Boids_default->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
        
		fputs("<y>", file);
        sprintf(data, "%f", h_Boids_default->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
        
		fputs("<goalX>", file);
        sprintf(data, "%f", h_Boids_default->goalX[i]);
		fputs(data, file);
		fputs("</goalX>\n", file);
        
		fputs("<goalY>", file);
        sprintf(data, "%f", h_Boids_default->goalY[i]);
		fputs(data, file);
		fputs("</goalY>\n", file);
        
		fputs("<velX>", file);
        sprintf(data, "%f", h_Boids_default->velX[i]);
		fputs(data, file);
		fputs("</velX>\n", file);
        
		fputs("<velY>", file);
        sprintf(data, "%f", h_Boids_default->velY[i]);
		fputs(data, file);
		fputs("</velY>\n", file);
        
		fputs("<velMax>", file);
        sprintf(data, "%f", h_Boids_default->velMax[i]);
		fputs(data, file);
		fputs("</velMax>\n", file);
        
		fputs("<mass>", file);
        sprintf(data, "%f", h_Boids_default->mass[i]);
		fputs(data, file);
		fputs("</mass>\n", file);
        
		fputs("</xagent>\n", file);
	}
	
	

	fputs("</states>\n" , file);
	
	/* Close the file */
	fclose(file);
}

void readInitialStates(char* inputpath, xmachine_memory_Boid_list* h_Boids, int* h_xmachine_memory_Boid_count)
{

	int temp = 0;
	int* itno = &temp;

	/* Pointer to file */
	FILE *file;
	/* Char and char buffer for reading file to */
	char c = ' ';
	char buffer[10000];
	char agentname[1000];

	/* Pointer to x-memory for initial state data */
	/*xmachine * current_xmachine;*/
	/* Variables for checking tags */
	int reading, i;
	int in_tag, in_itno, in_name;
    int in_Boid_id;
    int in_Boid_x;
    int in_Boid_y;
    int in_Boid_goalX;
    int in_Boid_goalY;
    int in_Boid_velX;
    int in_Boid_velY;
    int in_Boid_velMax;
    int in_Boid_mass;

	/* for continuous agents: set agent count to zero */	
	*h_xmachine_memory_Boid_count = 0;
	
	/* Variables for initial state data */
	int Boid_id;
	float Boid_x;
	float Boid_y;
	float Boid_goalX;
	float Boid_goalY;
	float Boid_velX;
	float Boid_velY;
	float Boid_velMax;
	float Boid_mass;
	
	/* Open config file to read-only */
	if((file = fopen(inputpath, "r"))==NULL)
	{
		printf("error opening initial states\n");
		exit(0);
	}
	
	/* Initialise variables */
    agent_maximum.x = 0;
    agent_maximum.y = 0;
    agent_maximum.z = 0;
    agent_minimum.x = 0;
    agent_minimum.y = 0;
    agent_minimum.z = 0;
	reading = 1;
	in_tag = 0;
	in_itno = 0;
	in_name = 0;
	in_Boid_id = 0;
	in_Boid_x = 0;
	in_Boid_y = 0;
	in_Boid_goalX = 0;
	in_Boid_goalY = 0;
	in_Boid_velX = 0;
	in_Boid_velY = 0;
	in_Boid_velMax = 0;
	in_Boid_mass = 0;
	//set all Boid values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_Boid_MAX; k++)
	{	
		h_Boids->id[k] = 0;
		h_Boids->x[k] = 0;
		h_Boids->y[k] = 0;
		h_Boids->goalX[k] = 0;
		h_Boids->goalY[k] = 0;
		h_Boids->velX[k] = 0;
		h_Boids->velY[k] = 0;
		h_Boids->velMax[k] = 0;
		h_Boids->mass[k] = 0;
	}
	

	/* Default variables for memory */
    Boid_id = 0;
    Boid_x = 0;
    Boid_y = 0;
    Boid_goalX = 0;
    Boid_goalY = 0;
    Boid_velX = 0;
    Boid_velY = 0;
    Boid_velMax = 0;
    Boid_mass = 0;

	/* Read file until end of xml */
    i = 0;
	while(reading==1)
	{
		/* Get the next char from the file */
		c = (char)fgetc(file);
		
		/* If the end of a tag */
		if(c == '>')
		{
			/* Place 0 at end of buffer to make chars a string */
			buffer[i] = 0;
			
			if(strcmp(buffer, "states") == 0) reading = 1;
			if(strcmp(buffer, "/states") == 0) reading = 0;
			if(strcmp(buffer, "itno") == 0) in_itno = 1;
			if(strcmp(buffer, "/itno") == 0) in_itno = 0;
			if(strcmp(buffer, "name") == 0) in_name = 1;
			if(strcmp(buffer, "/name") == 0) in_name = 0;
			if(strcmp(buffer, "/xagent") == 0)
			{
				if(strcmp(agentname, "Boid") == 0)
				{		
					if (*h_xmachine_memory_Boid_count > xmachine_memory_Boid_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent Boid exceeded whilst reading data\n", xmachine_memory_Boid_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(0);
					}
                    
					h_Boids->id[*h_xmachine_memory_Boid_count] = Boid_id;
					h_Boids->x[*h_xmachine_memory_Boid_count] = Boid_x;//Check maximum x value
                    if(agent_maximum.x < Boid_x)
                        agent_maximum.x = (float)Boid_x;
                    //Check minimum x value
                    if(agent_minimum.x > Boid_x)
                        agent_minimum.x = (float)Boid_x;
                    
					h_Boids->y[*h_xmachine_memory_Boid_count] = Boid_y;//Check maximum y value
                    if(agent_maximum.y < Boid_y)
                        agent_maximum.y = (float)Boid_y;
                    //Check minimum y value
                    if(agent_minimum.y > Boid_y)
                        agent_minimum.y = (float)Boid_y;
                    
					h_Boids->goalX[*h_xmachine_memory_Boid_count] = Boid_goalX;
					h_Boids->goalY[*h_xmachine_memory_Boid_count] = Boid_goalY;
					h_Boids->velX[*h_xmachine_memory_Boid_count] = Boid_velX;
					h_Boids->velY[*h_xmachine_memory_Boid_count] = Boid_velY;
					h_Boids->velMax[*h_xmachine_memory_Boid_count] = Boid_velMax;
					h_Boids->mass[*h_xmachine_memory_Boid_count] = Boid_mass;
					(*h_xmachine_memory_Boid_count) ++;	
				}
				else
				{
					printf("Warning: agent name undefined - '%s'\n", agentname);
				}
				

				
				/* Reset xagent variables */
                Boid_id = 0;
                Boid_x = 0;
                Boid_y = 0;
                Boid_goalX = 0;
                Boid_goalY = 0;
                Boid_velX = 0;
                Boid_velY = 0;
                Boid_velMax = 0;
                Boid_mass = 0;

			}
			if(strcmp(buffer, "id") == 0) in_Boid_id = 1;
			if(strcmp(buffer, "/id") == 0) in_Boid_id = 0;
			if(strcmp(buffer, "x") == 0) in_Boid_x = 1;
			if(strcmp(buffer, "/x") == 0) in_Boid_x = 0;
			if(strcmp(buffer, "y") == 0) in_Boid_y = 1;
			if(strcmp(buffer, "/y") == 0) in_Boid_y = 0;
			if(strcmp(buffer, "goalX") == 0) in_Boid_goalX = 1;
			if(strcmp(buffer, "/goalX") == 0) in_Boid_goalX = 0;
			if(strcmp(buffer, "goalY") == 0) in_Boid_goalY = 1;
			if(strcmp(buffer, "/goalY") == 0) in_Boid_goalY = 0;
			if(strcmp(buffer, "velX") == 0) in_Boid_velX = 1;
			if(strcmp(buffer, "/velX") == 0) in_Boid_velX = 0;
			if(strcmp(buffer, "velY") == 0) in_Boid_velY = 1;
			if(strcmp(buffer, "/velY") == 0) in_Boid_velY = 0;
			if(strcmp(buffer, "velMax") == 0) in_Boid_velMax = 1;
			if(strcmp(buffer, "/velMax") == 0) in_Boid_velMax = 0;
			if(strcmp(buffer, "mass") == 0) in_Boid_mass = 1;
			if(strcmp(buffer, "/mass") == 0) in_Boid_mass = 0;
			
			
			/* End of tag and reset buffer */
			in_tag = 0;
			i = 0;
		}
		/* If start of tag */
		else if(c == '<')
		{
			/* Place /0 at end of buffer to end numbers */
			buffer[i] = 0;
			/* Flag in tag */
			in_tag = 1;
			
			if(in_itno) *itno = atoi(buffer);
			if(in_name) strcpy(agentname, buffer);
			else
			{
				if(in_Boid_id){ 
                    Boid_id = (int) atoi(buffer);    
                }
				if(in_Boid_x){ 
                    Boid_x = (float) atof(buffer);    
                }
				if(in_Boid_y){ 
                    Boid_y = (float) atof(buffer);    
                }
				if(in_Boid_goalX){ 
                    Boid_goalX = (float) atof(buffer);    
                }
				if(in_Boid_goalY){ 
                    Boid_goalY = (float) atof(buffer);    
                }
				if(in_Boid_velX){ 
                    Boid_velX = (float) atof(buffer);    
                }
				if(in_Boid_velY){ 
                    Boid_velY = (float) atof(buffer);    
                }
				if(in_Boid_velMax){ 
                    Boid_velMax = (float) atof(buffer);    
                }
				if(in_Boid_mass){ 
                    Boid_mass = (float) atof(buffer);    
                }
				
			}
			
			/* Reset buffer */
			i = 0;
		}
		/* If in tag put read char into buffer */
		else if(in_tag)
		{
			buffer[i] = c;
			i++;
		}
		/* If in data read char into buffer */
		else
		{
			buffer[i] = c;
			i++;
		}
	}
	/* Close the file */
	fclose(file);
}

float3 getMaximumBounds(){
    return agent_maximum;
}

float3 getMinimumBounds(){
    return agent_minimum;
}

