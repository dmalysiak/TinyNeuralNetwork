
#ifndef TINY_NN_H_
#define TINY_NN_H_

#define NETWORK_DATA_TYPE double

extern inline NETWORK_DATA_TYPE g_deriv(NETWORK_DATA_TYPE a);

/*
 * the sigmoid function
 */
extern inline NETWORK_DATA_TYPE g(NETWORK_DATA_TYPE a);

/*
 * samples the sin function from 'start' to 'end' with 'size' equidistant steps
 */
void sampleSine(NETWORK_DATA_TYPE* input, NETWORK_DATA_TYPE* output, NETWORK_DATA_TYPE start, NETWORK_DATA_TYPE end, unsigned int size);

/*
 * fills the array with values from [-0.5,0.5]
 */
void fillArrayRand(NETWORK_DATA_TYPE* array, unsigned int size);

#endif
