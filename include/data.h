#ifndef DATA_H_
#define DATA_H_

#include "include.h"

#define NO_OF_TRAINING_SAMPLES 60000
#define SIZE_OF_ONE_TRAINING_SAMPLE 794
#define SIZE_OF_IMAGE 784
#define SIZE_OF_HIDDEN1 100
#define SIZE_OF_HIDDEN2 90
#define SIZE_OF_HIDDEN3 80
#define SIZE_OF_HIDDEN4 70
#define SIZE_OF_HIDDEN5 60
#define SIZE_OF_LABELS 10
#define LEARN_RATE 0.01f
#define EPOCHS 10
#define SAMPLES_PER_BATCH 60000

float* AllocateDatabaseToMemory();
void FreeAllocatedMemory(float**);

#endif
