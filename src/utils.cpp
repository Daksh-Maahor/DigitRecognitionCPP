#include "../include/utils.h"

void PrintFloatArray(float* array, int size)
{
    for (int i = 0; i < size; ++i)
    {
        std::cout << array[i] << ", ";
    }
    std::cout << std::endl;
}

float Sigmoid(float a)
{
    return 1 / (1 + expf(-a));
}

int ArgMax(float* array, int size)
{
    int idx = 0;
    float MAX = array[0];
    for (int i = 0; i < size; ++i)
    {
        if (array[i] > MAX)
        {
            MAX = array[i];
            idx = i;
        }
    }

    return idx;
}
