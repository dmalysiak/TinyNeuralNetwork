/*
 ============================================================================
 Name        : TinyNN.c
 Author      : Darius Malysiak
 Version     : 0.01
 Copyright   : It's mine
 Description : A tiny 2-layer NN (one input neuron and one output neuron) implementation with backpropagation training
 ============================================================================
 */

//general NN params
#define NEURON_COUNT 2000
#define SAMPLE_COUNT 200
#define MAX_ITERATIONS 2000

//these we need!
#include <stdlib.h>
#include <math.h>
#include "TinyNN.h"

//flavor!
//#define MEASURE_TIME
//#define FANCY
#define OUTPUT

#include <stdio.h>

#ifdef MEASURE_TIME
#include <time.h>
#include <stdio.h>

void tickDiff(struct timespec* start, struct timespec* end, struct timespec* diff)
{

    if ((end->tv_nsec-start->tv_nsec)<0) {
    	diff->tv_sec = end->tv_sec-start->tv_sec-1;
    	diff->tv_nsec = 1000000000+end->tv_nsec-start->tv_nsec;
    } else {
    	diff->tv_sec = end->tv_sec-start->tv_sec;
    	diff->tv_nsec = end->tv_nsec-start->tv_nsec;
    }
}

#define TIC struct timespec* a = (struct timespec*) malloc(sizeof(struct timespec)); \
	struct timespec* b = (struct timespec*) malloc(sizeof(struct timespec)); \
	struct timespec* diff = (struct timespec*) malloc(sizeof(struct timespec)); \
	clock_gettime(CLOCK_REALTIME, a);

#define TOC clock_gettime(CLOCK_REALTIME, b); \
	tickDiff(a,b,diff); \
	printf("Process took = %.9Lf s\n",(long double)(diff->tv_sec + 0.000000001*diff->tv_nsec) ); \
	free(a); \
	free(b); \
	free(diff);
#endif

#ifdef FANCY
#include <stdio.h>
#endif

#ifdef OUTPUT
#include <stdio.h>
#endif

/*
 * the derivative of the sigmoid function
 */
inline NETWORK_DATA_TYPE g_deriv(NETWORK_DATA_TYPE a)
{
	NETWORK_DATA_TYPE sigmoid = 1.0 / (1.0 + exp(-1.0*a));

	return sigmoid*(1.0-sigmoid);
}

/*
 * the sigmoid function
 */
inline NETWORK_DATA_TYPE g(NETWORK_DATA_TYPE a)
{
	return 1.0 / (1.0 + exp(-1.0*a));
}

/*
 * samples the sin function from 'start' to 'end' with 'size' equidistant steps
 */
void sampleSine(NETWORK_DATA_TYPE* input, NETWORK_DATA_TYPE* output, NETWORK_DATA_TYPE start, NETWORK_DATA_TYPE end, unsigned int size)
{
	NETWORK_DATA_TYPE step = (end-start)/(NETWORK_DATA_TYPE)size;

	unsigned int i;
	for(i=0; i<size; ++i)
	{
		input[i] = start + step * (NETWORK_DATA_TYPE)i;
		output[i] = sin(input[i]);
	}
}

/*
 * fills the array with values from [-0.5,0.5]
 */
void fillArrayRand(NETWORK_DATA_TYPE* array, unsigned int size)
{
	unsigned int i;
	for(i=0; i<size; ++i)
	{
		array[i] = (NETWORK_DATA_TYPE)rand()/(NETWORK_DATA_TYPE)RAND_MAX - 0.5;
	}
}

/*
 * fills the array with 0
 */
void fillArrayNull(NETWORK_DATA_TYPE* array, unsigned int size)
{
	unsigned int i;
	for(i=0; i<size; ++i)
	{
		array[i] = (NETWORK_DATA_TYPE)0;
	}
}

int train(NETWORK_DATA_TYPE eta, NETWORK_DATA_TYPE error_threshold, NETWORK_DATA_TYPE bias, NETWORK_DATA_TYPE alpha)
{

	NETWORK_DATA_TYPE total_error = error_threshold + 1.0;//why?

	//data
	NETWORK_DATA_TYPE input[SAMPLE_COUNT];
	NETWORK_DATA_TYPE output[SAMPLE_COUNT];

	//NN container for NEURON_COUNT neurons
	NETWORK_DATA_TYPE NN_output = 0;
	NETWORK_DATA_TYPE weights_layer0_1[NEURON_COUNT*2]; //don't forget the bias :-) {PSSSTT: format = [input_w, bias_w, input_w, bias_w...]}
	NETWORK_DATA_TYPE weights_layer1_2[NEURON_COUNT];

	NETWORK_DATA_TYPE delta_weights_layer0_1[NEURON_COUNT*2]; //don't forget the bias :-) {PSSSTT: format = [input_w, bias_w, input_w, bias_w...]}
	NETWORK_DATA_TYPE delta_weights_layer1_2[NEURON_COUNT];

	NETWORK_DATA_TYPE delta_layer_2, delta_layer_1;
	NETWORK_DATA_TYPE a;
	NETWORK_DATA_TYPE g_layer_1[NEURON_COUNT];
	NETWORK_DATA_TYPE g_deriv_layer_1[NEURON_COUNT];
	NETWORK_DATA_TYPE delta_w_layer_2_1;
	NETWORK_DATA_TYPE delta_w_layer_1_0;

	//init the network
	fillArrayRand(weights_layer0_1,NEURON_COUNT*2);//random values for the weight 0->1
	fillArrayRand(weights_layer1_2,NEURON_COUNT);//random values for the weights 1->2

	fillArrayNull(delta_weights_layer0_1,NEURON_COUNT*2);
	fillArrayNull(delta_weights_layer1_2,NEURON_COUNT);

	//create the training data
	sampleSine(input, output, 0, 3.14, SAMPLE_COUNT);

	//train the network
	unsigned int iteration_count = 0;

	#ifdef MEASURE_TIME
		TIC
	#endif


	while(total_error > error_threshold && iteration_count < MAX_ITERATIONS)
	{
		total_error = 0;//hmmm....

		//present the data to the NN
		unsigned int i;
		for(i=0; i<SAMPLE_COUNT; ++i)
		{
			//propagate the sample forward
			//layer 0 -> 1 ... and layer 1->2
			unsigned int j;
			for(j=0; j<NEURON_COUNT; ++j)
			{
				a = input[i] * weights_layer0_1[2*j] + bias * weights_layer0_1[2*j+1];
				g_layer_1[j] = g(a);
				g_deriv_layer_1[j] = g_deriv(a);

				NN_output += weights_layer1_2[j] * g_layer_1[j];
			}

			//calculate the errors
			delta_layer_2 = NN_output - output[i];
			total_error += delta_layer_2*delta_layer_2;

			//backpropagate the deltas
			//layer 2->1 ... and 1->0
			for(j=0; j<NEURON_COUNT; ++j)
			{
				delta_layer_1 = g_deriv_layer_1[j] * weights_layer1_2[j] * delta_layer_2;
				delta_w_layer_2_1 = delta_layer_2 * g_layer_1[j];

				//?
				weights_layer1_2[j] -= delta_w_layer_2_1 * eta + alpha*delta_weights_layer1_2[j] ;
				delta_weights_layer1_2[j] = delta_w_layer_2_1 * eta + alpha*delta_weights_layer1_2[j] ;

				delta_w_layer_1_0 = delta_layer_1 * input[i];
				weights_layer0_1[2*j] -= delta_w_layer_1_0 * eta + alpha*delta_weights_layer0_1[2*j];
				delta_weights_layer0_1[2*j] = delta_w_layer_1_0 * eta + alpha*delta_weights_layer0_1[2*j];

				delta_w_layer_1_0 = delta_layer_1 * bias;
				weights_layer0_1[2*j+1] -= delta_w_layer_1_0 * eta + alpha*delta_weights_layer0_1[2*j+1];
				delta_weights_layer0_1[2*j+1] = delta_w_layer_1_0 * eta + alpha*delta_weights_layer0_1[2*j+1];
			}

			//update the weights (uppsi, did it already... but how?)
			#ifdef FANCY
			printf("sample error %f, should %f is %f\n",delta_layer_2,output[i], NN_output);
			#endif

			//prepare for the next nerve wrecking round
			NN_output = 0;
		}

		#ifdef FANCY
		printf("Total error %f\n\n",total_error);
		#endif

		++iteration_count;

		fillArrayNull(delta_weights_layer0_1,NEURON_COUNT*2);
		fillArrayNull(delta_weights_layer1_2,NEURON_COUNT);

		#ifdef OUTPUT
		printf("iteration %d Total error %f\n",iteration_count,total_error);
		#endif
	}

	#ifdef MEASURE_TIME
			TOC
	#endif

	return EXIT_SUCCESS;
}

int main()
{

	train(0.005, 0.01, 1.0, 0.4);

	return EXIT_SUCCESS;
}
