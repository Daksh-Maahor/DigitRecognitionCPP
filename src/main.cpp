#include "../include/data.h"
#include "../include/utils.h"

#include "../include/include.h"

int main() {
    float* data = AllocateDatabaseToMemory();

    float weights_input_hidden1[SIZE_OF_HIDDEN1][SIZE_OF_IMAGE];
    float biases_input_hidden1[SIZE_OF_HIDDEN1];

    float weights_hidden1_hidden2[SIZE_OF_HIDDEN2][SIZE_OF_HIDDEN1];
    float biases_hidden1_hidden2[SIZE_OF_HIDDEN2];

    float weights_hidden2_hidden3[SIZE_OF_HIDDEN3][SIZE_OF_HIDDEN2];
    float biases_hidden2_hidden3[SIZE_OF_HIDDEN3];

    float weights_hidden3_hidden4[SIZE_OF_HIDDEN4][SIZE_OF_HIDDEN3];
    float biases_hidden3_hidden4[SIZE_OF_HIDDEN4];

    float weights_hidden4_hidden5[SIZE_OF_HIDDEN5][SIZE_OF_HIDDEN4];
    float biases_hidden4_hidden5[SIZE_OF_HIDDEN5];

    float weights_hidden5_output[SIZE_OF_LABELS][SIZE_OF_HIDDEN5];
    float biases_hidden5_output[SIZE_OF_LABELS];

    // initialise weights
    for (int i = 0; i < SIZE_OF_HIDDEN1; ++i)
    {
        for (int j = 0; j < SIZE_OF_IMAGE; ++j)
        {
            weights_input_hidden1[i][j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
        }
    }

    for (int i = 0; i < SIZE_OF_HIDDEN2; ++i)
    {
        for (int j = 0; j < SIZE_OF_HIDDEN1; ++j)
        {
            weights_hidden1_hidden2[i][j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
        }
    }


    for (int i = 0; i < SIZE_OF_HIDDEN3; ++i)
    {
        for (int j = 0; j < SIZE_OF_HIDDEN2; ++j)
        {
            weights_hidden2_hidden3[i][j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
        }
    }

    for (int i = 0; i < SIZE_OF_HIDDEN4; ++i)
    {
        for (int j = 0; j < SIZE_OF_HIDDEN3; ++j)
        {
            weights_hidden3_hidden4[i][j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
        }
    }

    for (int i = 0; i < SIZE_OF_HIDDEN5; ++i)
    {
        for (int j = 0; j < SIZE_OF_HIDDEN4; ++j)
        {
            weights_hidden4_hidden5[i][j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
        }
    }

    for (int i = 0; i < SIZE_OF_LABELS; ++i)
    {
        for (int j = 0; j < SIZE_OF_HIDDEN5; ++j)
        {
            weights_hidden5_output[i][j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
        }
    }

    // initialize biases
    for (int i = 0; i < SIZE_OF_HIDDEN1; ++i)
    {
        biases_input_hidden1[i] = 0;
    }

    for (int i = 0; i < SIZE_OF_HIDDEN2; ++i)
    {
        biases_hidden1_hidden2[i] = 0;
    }

    for (int i = 0; i < SIZE_OF_HIDDEN3; ++i)
    {
        biases_hidden2_hidden3[i] = 0;
    }

    for (int i = 0; i < SIZE_OF_HIDDEN4; ++i)
    {
        biases_hidden3_hidden4[i] = 0;
    }

    for (int i = 0; i < SIZE_OF_HIDDEN5; ++i)
    {
        biases_hidden4_hidden5[i] = 0;
    }

    for (int i = 0; i < SIZE_OF_LABELS; ++i)
    {
        biases_hidden5_output[i] = 0;
    }

    // initialize parameters
    int correct = 0;

    // learn
    for (int i = 0; i < EPOCHS; ++i)
    {
        std::cout << "Epoch : " << i << std::endl;
        for (int j = 0; j < SAMPLES_PER_BATCH; ++j)
        {
            int idx = std::rand() % NO_OF_TRAINING_SAMPLES;

            float img[SIZE_OF_IMAGE];
            float label[SIZE_OF_LABELS];

            // copy data of images and labels in images and labels
            for (int k = 0; k < SIZE_OF_IMAGE; ++k)
            {
                img[k] = data[idx * SIZE_OF_ONE_TRAINING_SAMPLE + k];
            }

            for (int k = 0; k < SIZE_OF_LABELS; ++k)
            {
                label[k] = data[idx * SIZE_OF_ONE_TRAINING_SAMPLE + SIZE_OF_IMAGE + k];
            }


            //PrintFloatArray(label, SIZE_OF_LABELS);
            //PrintFloatArray(img, SIZE_OF_IMAGE);

            // forward propagation : input -> hidden1
            float h1[SIZE_OF_HIDDEN1];

            for (int k = 0; k < SIZE_OF_HIDDEN1; ++k)
            {
                // evaluate h1[k]

                float h = 0;
                for (int a = 0; a < SIZE_OF_IMAGE; ++a)
                {
                    h += weights_input_hidden1[k][a] * img[a];
                }
                h += biases_input_hidden1[k];

                h1[k] = Sigmoid(h);
            }

            // forward propagation : hidden1 -> hidden2
            float h2[SIZE_OF_HIDDEN2];

            for (int k = 0; k < SIZE_OF_HIDDEN2; ++k)
            {
                // evaluate h2[k]

                float h = 0;
                for (int a = 0; a < SIZE_OF_HIDDEN1; ++a)
                {
                    h += weights_hidden1_hidden2[k][a] * h1[a];
                }
                h += biases_hidden1_hidden2[k];

                h2[k] = Sigmoid(h);
            }

            // forward propagation : hidden2 -> hidden3
            float h3[SIZE_OF_HIDDEN3];

            for (int k = 0; k < SIZE_OF_HIDDEN3; ++k)
            {
                // evaluate h3[k]

                float h = 0;
                for (int a = 0; a < SIZE_OF_HIDDEN2; ++a)
                {
                    h += weights_hidden2_hidden3[k][a] * h2[a];
                }
                h += biases_hidden2_hidden3[k];

                h3[k] = Sigmoid(h);
            }

            // forward propagation : hidden3 -> hidden4
            float h4[SIZE_OF_HIDDEN4];

            for (int k = 0; k < SIZE_OF_HIDDEN4; ++k)
            {
                // evaluate h4[k]

                float h = 0;
                for (int a = 0; a < SIZE_OF_HIDDEN3; ++a)
                {
                    h += weights_hidden3_hidden4[k][a] * h3[a];
                }
                h += biases_hidden3_hidden4[k];

                h4[k] = Sigmoid(h);
            }

            // forward propagation : hidden4 -> hidden5
            float h5[SIZE_OF_HIDDEN5];

            for (int k = 0; k < SIZE_OF_HIDDEN5; ++k)
            {
                // evaluate h5[k]

                float h = 0;
                for (int a = 0; a < SIZE_OF_HIDDEN4; ++a)
                {
                    h += weights_hidden4_hidden5[k][a] * h4[a];
                }
                h += biases_hidden4_hidden5[k];

                h5[k] = Sigmoid(h);
            }

            // forward propagation : hidden5 -> output
            float o[SIZE_OF_LABELS];
            for (int k = 0; k < SIZE_OF_LABELS; ++k)
            {
                // evaluate o[k]

                float o_val = 0;
                for (int a = 0; a < SIZE_OF_HIDDEN5; ++a)
                {
                    o_val += weights_hidden5_output[k][a] * h5[a];
                }
                o_val += biases_hidden5_output[k];

                o[k] = Sigmoid(o_val);
            }
            //PrintFloatArray(o, SIZE_OF_LABELS);


            // Calculate error
            float delta_o[SIZE_OF_LABELS];
            for (int k = 0; k < SIZE_OF_LABELS; k++)
            {
                delta_o[k] = o[k] - label[k];
            }

            // Calculate cost
            float e = 0;
            for (int k = 0; k < SIZE_OF_LABELS; k++)
            {
                e += delta_o[k] * delta_o[k];
            }
            e /= SIZE_OF_LABELS;

            // check correctness
            if (ArgMax(o, SIZE_OF_LABELS) == ArgMax(label, SIZE_OF_LABELS))
            {
                correct++;
                //std::cout << "Correct Guess" << std::endl;
            }

            // back propagation : output -> hidden5
            for (int row = 0; row < SIZE_OF_LABELS; ++row)
            {
                for (int col = 0; col < SIZE_OF_HIDDEN5; ++col)
                {
                    weights_hidden5_output[row][col] += -LEARN_RATE * delta_o[row] * h1[col];
                }
                biases_hidden5_output[row] += -LEARN_RATE * delta_o[row];
            }

            // Calculate delta_h5
            float delta_h5[SIZE_OF_HIDDEN5];
            for (int row = 0; row < SIZE_OF_HIDDEN5; ++row)
            {
                float delta_h_val = 0;
                for (int col = 0; col < SIZE_OF_LABELS; ++col)
                {
                    delta_h_val += weights_hidden5_output[col][row] * delta_o[col];
                }
                delta_h5[row] = delta_h_val * h5[row] * (1 - h5[row]);
            }

            // back propagation : hidden5 -> hidden4
            for (int row = 0; row < SIZE_OF_HIDDEN5; ++row)
            {
                for (int col = 0; col < SIZE_OF_HIDDEN4; ++col)
                {
                    weights_hidden4_hidden5[row][col] += -LEARN_RATE * delta_h5[row] * h4[col];
                }
                biases_hidden4_hidden5[row] += -LEARN_RATE * delta_h5[row];
            }

            // Calculate delta_h4
            float delta_h4[SIZE_OF_HIDDEN4];
            for (int row = 0; row < SIZE_OF_HIDDEN4; ++row)
            {
                float delta_h_val = 0;
                for (int col = 0; col < SIZE_OF_HIDDEN5; ++col)
                {
                    delta_h_val += weights_hidden4_hidden5[col][row] * delta_h5[col];
                }
                delta_h4[row] = delta_h_val * h4[row] * (1 - h4[row]);
            }

            // back propagation : hidden4 -> hidden3
            for (int row = 0; row < SIZE_OF_HIDDEN4; ++row)
            {
                for (int col = 0; col < SIZE_OF_HIDDEN3; ++col)
                {
                    weights_hidden3_hidden4[row][col] += -LEARN_RATE * delta_h4[row] * h3[col];
                }
                biases_hidden3_hidden4[row] += -LEARN_RATE * delta_h4[row];
            }

            // Calculate delta_h3
            float delta_h3[SIZE_OF_HIDDEN3];
            for (int row = 0; row < SIZE_OF_HIDDEN3; ++row)
            {
                float delta_h_val = 0;
                for (int col = 0; col < SIZE_OF_HIDDEN4; ++col)
                {
                    delta_h_val += weights_hidden3_hidden4[col][row] * delta_h4[col];
                }
                delta_h3[row] = delta_h_val * h3[row] * (1 - h3[row]);
            }

            // back propagation : hidden3 -> hidden2
            for (int row = 0; row < SIZE_OF_HIDDEN3; ++row)
            {
                for (int col = 0; col < SIZE_OF_HIDDEN2; ++col)
                {
                    weights_hidden2_hidden3[row][col] += -LEARN_RATE * delta_h3[row] * h2[col];
                }
                biases_hidden2_hidden3[row] += -LEARN_RATE * delta_h3[row];
            }

            // Calculate delta_h2
            float delta_h2[SIZE_OF_HIDDEN2];
            for (int row = 0; row < SIZE_OF_HIDDEN2; ++row)
            {
                float delta_h_val = 0;
                for (int col = 0; col < SIZE_OF_HIDDEN3; ++col)
                {
                    delta_h_val += weights_hidden2_hidden3[col][row] * delta_h3[col];
                }
                delta_h2[row] = delta_h_val * h2[row] * (1 - h2[row]);
            }

            // back propagation : hidden2 -> hidden1
            for (int row = 0; row < SIZE_OF_HIDDEN2; ++row)
            {
                for (int col = 0; col < SIZE_OF_HIDDEN1; ++col)
                {
                    weights_hidden1_hidden2[row][col] += -LEARN_RATE * delta_h2[row] * h1[col];
                }
                biases_hidden1_hidden2[row] += -LEARN_RATE * delta_h2[row];
            }

            // Calculate delta_h1
            float delta_h1[SIZE_OF_HIDDEN1];
            for (int row = 0; row < SIZE_OF_HIDDEN1; ++row)
            {
                float delta_h_val = 0;
                for (int col = 0; col < SIZE_OF_HIDDEN2; ++col)
                {
                    delta_h_val += weights_hidden1_hidden2[col][row] * delta_h2[col];
                }
                delta_h1[row] = delta_h_val * h1[row] * (1 - h1[row]);
            }

            // back propagation : hidden1 -> input
            for (int row = 0; row < SIZE_OF_HIDDEN1; ++row)
            {
                for (int col = 0; col < SIZE_OF_IMAGE; ++col)
                {
                    weights_input_hidden1[row][col] += -LEARN_RATE * delta_h1[row] * img[col];
                }
                biases_input_hidden1[row] += -LEARN_RATE * delta_h1[row];
            }

            //std::cout << "\n\n" << std::endl;

        }

        std::cout << "Accuracy : " << (std::float_t(correct * 10000) / SAMPLES_PER_BATCH) / 100 << std::endl;
        correct = 0;
    }

    std::cout << "\n\n\n\n\n";
    std::cout << "Weights from input to hidden1\n" << std::endl;
    std::cout << "[";
    for (int i = 0; i < SIZE_OF_HIDDEN1; ++i)
    {
        std::cout << "[";
        for (int j = 0; j < SIZE_OF_IMAGE; ++j)
        {
            std::cout << weights_input_hidden1[i][j] << ", ";
        }
        std::cout << "]," << std::endl;
    }
    std::cout << "]\n\n" << std::endl;

    std::cout << "Biases for hidden1\n" << std::endl;
    std::cout << "[";
    for (int i = 0; i < SIZE_OF_HIDDEN1; ++i)
    {
        std::cout << biases_input_hidden1[i] << ", ";
    }
    std::cout << "]\n\n" << std::endl;

    std::cout << "\n\n\n\n\n";
    std::cout << "Weights from hidden1 to hidden2\n" << std::endl;
    std::cout << "[";
    for (int i = 0; i < SIZE_OF_HIDDEN2; ++i)
    {
        std::cout << "[";
        for (int j = 0; j < SIZE_OF_HIDDEN1; ++j)
        {
            std::cout << weights_hidden1_hidden2[i][j] << ", ";
        }
        std::cout << "]," << std::endl;
    }
    std::cout << "]\n\n" << std::endl;

    std::cout << "Biases for hidden2\n" << std::endl;
    std::cout << "[";
    for (int i = 0; i < SIZE_OF_HIDDEN2; ++i)
    {
        std::cout << biases_hidden1_hidden2[i] << ", ";
    }
    std::cout << "]\n\n" << std::endl;

    std::cout << "\n\n\n\n\n";
    std::cout << "Weights from hidden2 to hidden3\n" << std::endl;
    std::cout << "[";
    for (int i = 0; i < SIZE_OF_HIDDEN3; ++i)
    {
        std::cout << "[";
        for (int j = 0; j < SIZE_OF_HIDDEN2; ++j)
        {
            std::cout << weights_hidden2_hidden3[i][j] << ", ";
        }
        std::cout << "]," << std::endl;
    }
    std::cout << "]\n\n" << std::endl;

    std::cout << "Biases for hidden3\n" << std::endl;
    std::cout << "[";
    for (int i = 0; i < SIZE_OF_HIDDEN3; ++i)
    {
        std::cout << biases_hidden2_hidden3[i] << ", ";
    }
    std::cout << "]\n\n" << std::endl;

    std::cout << "\n\n\n\n\n";
    std::cout << "Weights from hidden3 to hidden4\n" << std::endl;
    std::cout << "[";
    for (int i = 0; i < SIZE_OF_HIDDEN4; ++i)
    {
        std::cout << "[";
        for (int j = 0; j < SIZE_OF_HIDDEN3; ++j)
        {
            std::cout << weights_hidden3_hidden4[i][j] << ", ";
        }
        std::cout << "]," << std::endl;
    }
    std::cout << "]\n\n" << std::endl;

    std::cout << "Biases for hidden4\n" << std::endl;
    std::cout << "[";
    for (int i = 0; i < SIZE_OF_HIDDEN4; ++i)
    {
        std::cout << biases_hidden3_hidden4[i] << ", ";
    }
    std::cout << "]\n\n" << std::endl;

    std::cout << "\n\n\n\n\n";
    std::cout << "Weights from hidden4 to hidden5\n" << std::endl;
    std::cout << "[";
    for (int i = 0; i < SIZE_OF_HIDDEN5; ++i)
    {
        std::cout << "[";
        for (int j = 0; j < SIZE_OF_HIDDEN4; ++j)
        {
            std::cout << weights_hidden4_hidden5[i][j] << ", ";
        }
        std::cout << "]," << std::endl;
    }
    std::cout << "]\n\n" << std::endl;

    std::cout << "Biases for hidden5\n" << std::endl;
    std::cout << "[";
    for (int i = 0; i < SIZE_OF_HIDDEN5; ++i)
    {
        std::cout << biases_hidden4_hidden5[i] << ", ";
    }
    std::cout << "]\n\n" << std::endl;

    std::cout << "\n\n\n\n\n";
    std::cout << "Weights from hidden5 to output\n" << std::endl;
    std::cout << "[";
    for (int i = 0; i < SIZE_OF_LABELS; ++i)
    {
        std::cout << "[";
        for (int j = 0; j < SIZE_OF_HIDDEN5; ++j)
        {
            std::cout << weights_hidden5_output[i][j] << ", ";
        }
        std::cout << "]," << std::endl;
    }
    std::cout << "]\n\n" << std::endl;

    std::cout << "Biases for output\n" << std::endl;
    std::cout << "[";
    for (int i = 0; i < SIZE_OF_LABELS; ++i)
    {
        std::cout << biases_hidden5_output[i] << ", ";
    }
    std::cout << "]\n\n" << std::endl;

    // write to file

    std::ofstream wf("res/data/neural_data.dat", std::ios::out | std::ios::binary);

    if (!wf)
    {
        std::cout << "Can not open file." << std::endl;
        return 1;
    }
    
    // byte 0-3 : how many layers (including input and hidden)
    int numlayers = 7;
    wf.write((char*) &numlayers, sizeof(int));

    // write all weights and biases
    for (int i = 0; i < SIZE_OF_HIDDEN1; i++)
    {
        for (int j = 0; j < SIZE_OF_IMAGE; j++)
        {
            float data = weights_input_hidden1[i][j];

            wf.write((char*) &data, sizeof(float));
        }
    }

    for (int i = 0; i < SIZE_OF_HIDDEN1; i++)
    {
        float data = biases_input_hidden1[i];
        wf.write((char*) &data, sizeof(float));
    }

    for (int i = 0; i < SIZE_OF_HIDDEN2; i++)
    {
        for (int j = 0; j < SIZE_OF_HIDDEN1; j++)
        {
            float data = weights_hidden1_hidden2[i][j];

            wf.write((char*) &data, sizeof(float));
        }
    }

    for (int i = 0; i < SIZE_OF_HIDDEN2; i++)
    {
        float data = biases_hidden1_hidden2[i];
        wf.write((char*) &data, sizeof(float));
    }

    for (int i = 0; i < SIZE_OF_HIDDEN3; i++)
    {
        for (int j = 0; j < SIZE_OF_HIDDEN2; j++)
        {
            float data = weights_hidden2_hidden3[i][j];

            wf.write((char*) &data, sizeof(float));
        }
    }

    for (int i = 0; i < SIZE_OF_HIDDEN3; i++)
    {
        float data = biases_hidden2_hidden3[i];
        wf.write((char*) &data, sizeof(float));
    }

    for (int i = 0; i < SIZE_OF_HIDDEN4; i++)
    {
        for (int j = 0; j < SIZE_OF_HIDDEN3; j++)
        {
            float data = weights_hidden3_hidden4[i][j];

            wf.write((char*) &data, sizeof(float));
        }
    }

    for (int i = 0; i < SIZE_OF_HIDDEN4; i++)
    {
        float data = biases_hidden3_hidden4[i];
        wf.write((char*) &data, sizeof(float));
    }

    for (int i = 0; i < SIZE_OF_HIDDEN5; i++)
    {
        for (int j = 0; j < SIZE_OF_HIDDEN4; j++)
        {
            float data = weights_hidden4_hidden5[i][j];

            wf.write((char*) &data, sizeof(float));
        }
    }

    for (int i = 0; i < SIZE_OF_HIDDEN5; i++)
    {
        float data = biases_hidden4_hidden5[i];
        wf.write((char*) &data, sizeof(float));
    }

    for (int i = 0; i < SIZE_OF_LABELS; i++)
    {
        for (int j = 0; j < SIZE_OF_HIDDEN5; j++)
        {
            float data = weights_hidden5_output[i][j];

            wf.write((char*) &data, sizeof(float));
        }
    }

    for (int i = 0; i < SIZE_OF_LABELS; i++)
    {
        float data = biases_hidden5_output[i];
        wf.write((char*) &data, sizeof(float));
    }

    wf.close();

    if (!wf.good())
    {
        std::cout << "Error occurred at writing time!" << std::endl;
        return 1;
    }

    // Free the allocated memory
    FreeAllocatedMemory(&data);
    std::cin.get();

    return 0;
}
