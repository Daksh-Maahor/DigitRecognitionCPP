#include "../include/data.h"

float* AllocateDatabaseToMemory() {
    std::ifstream file("res/data/data.csv");
    if (!file.is_open())
    {
        std::cerr << "Error opening file" << std::endl;
        std::cin.get();
    }

    float* data;
    data = (float*)(malloc(sizeof(float) * (NO_OF_TRAINING_SAMPLES * SIZE_OF_ONE_TRAINING_SAMPLE)));

    std::string line;
    int row = 0;

    while (std::getline(file, line) && row < NO_OF_TRAINING_SAMPLES)
    {
        std::stringstream ss(line);
        std::string cell;
        int col = 0;

        while (std::getline(ss, cell, ',') && col < SIZE_OF_ONE_TRAINING_SAMPLE)
        {
            float d = std::stof(cell);
            if (col < 784)
            {
                d /= 255;
            }
            data[row * SIZE_OF_ONE_TRAINING_SAMPLE + col] = d;
            col++;
        }
        row++;
    }

    file.close();

    std::cout << "Data Acquired" << std::endl;

    return data;
}

void FreeAllocatedMemory(float** data)
{
    free(*data);
    std::cout << "Data Freed" << std::endl;
}
