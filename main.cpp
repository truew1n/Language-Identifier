#include <iostream>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <limits>
#include <vector>
#include <cstring>
#include <iomanip>
#include <windows.h>
#include <unordered_map>

using namespace std;

#define MF_E 2.71828182845904523

typedef struct input_t {
    string lang;
    float letter_freq[26]; // Unipolar float [0.0f;1.0f]
} input_t;

typedef vector<input_t> trainset_t;

#define ssize(arr) (sizeof(arr)/sizeof(arr[0]))

void get_trainset(std::string filepath, trainset_t *trainset, unordered_map<string, uint8_t> *truth_table)
{
    FILE *file = fopen(filepath.c_str(), "rb");
    if(!file) exit(-1);

    bool is_first_comma = true;
    input_t input = {};
    uint32_t title_length = 0;
    uint8_t lang_counter = 0;

    char c = 0;
    do {
        c = fgetc(file);
        switch(c) {
            case ',': {
                if(is_first_comma) {
                    if(truth_table) {
                        if (truth_table->find(input.lang) == truth_table->end()) {
                            (*truth_table)[input.lang] = lang_counter;
                            lang_counter++;
                        }
                    }
                    is_first_comma = false;
                }
                break;
            }
            case '\r': {
                break;
            }
            case EOF:  {
                if(input.lang == "") break;
            }
            case '\n': {
                for(uint32_t i = 0; i < ssize(input.letter_freq); ++i) {
                    input.letter_freq[i] /= (float)title_length;
                }
                trainset->push_back(input);
                is_first_comma = true;
                input = {};
                title_length = 0;
                break;
            }
            default: {
                if(is_first_comma) {
                    input.lang += c;
                } else {
                    bool is_cap = (c > 0x40 && c < 0x5B);
                    c = (c + 0x20) * is_cap + c * !is_cap;
                    if(c > 0x60 && c < 0x7B) {
                        input.letter_freq[c - 0x61] += 1.0f;
                        title_length++;
                    }
                }
                break;
            }
        }
    } while(c != EOF);

    fclose(file);
}

input_t parse_input_line(string input_line)
{
    input_t input = {};
    uint32_t title_length = 0;
    for(int32_t i = 0; i < input_line.size(); ++i) {
        char c = input_line.at(i);
        bool is_cap = (c > 0x40 && c < 0x5B);
        c = (c + 0x20) * is_cap + c * !is_cap;
        if(c > 0x60 && c < 0x7B) {
            input.letter_freq[c - 0x61] += 1.0f;
            title_length++;
        }
    }
    for(uint32_t i = 0; i < ssize(input.letter_freq); ++i) {
        input.letter_freq[i] /= (float)title_length;
    }
    return input;
}

float sigmoidf(float x)
{
    return 1.0f / (1.0f - powf(MF_E, - x));
}

float calc_cost(trainset_t *trainset, unordered_map<string, uint8_t> *truth_table, float *weights, uint64_t input_size, float *biases, uint64_t output_size)
{
    float cost = 0.0f;
    uint32_t num_samples = 0;

    for (const input_t& input : *trainset)
    {
        if (ssize(input.letter_freq) != input_size || truth_table->find(input.lang) == truth_table->end())
            continue;

        uint8_t truth_index = (*truth_table)[input.lang];
        ++num_samples;

        float partial_cost = 0.0f;
        for (uint32_t j = 0; j < output_size; ++j)
        {
            float y = 0.0f;
            for (uint32_t i = 0; i < input_size; ++i)
            {
                y += input.letter_freq[i] * weights[i + j * input_size];
            }
            y = sigmoidf(y - biases[j]);
            partial_cost += (truth_index == j) ? ((1.0f - y) * (1.0f - y)) : (y * y);
        }
        partial_cost /= output_size;
        cost += partial_cost;
    }

    cost /= (float)(num_samples);
    return cost;
}

void update(float learning_rate, trainset_t *trainset, unordered_map<string, uint8_t> *truth_table, float *weights, uint32_t input_size, float *biases, uint32_t output_size) {
    for (const input_t& input : *trainset) {
        if (ssize(input.letter_freq) != input_size || truth_table->find(input.lang) == truth_table->end())
            continue;

        uint8_t truth_index = (*truth_table)[input.lang];

        // Calculate Q-values for each action (output neuron)
        float max_q_value = -INFINITY;
        for (uint32_t j = 0; j < output_size; ++j) {
            float q_value = 0.0f;
            for (uint32_t i = 0; i < input_size; ++i) {
                q_value += input.letter_freq[i] * weights[i + j * input_size];
            }
            q_value = sigmoidf(q_value - biases[j]);

            // Update max Q-value
            if (q_value > max_q_value)
                max_q_value = q_value;

            // Update weights and biases based on Q-learning update rule
            float td_target = (truth_index == j) ? 1.0f : 0.0f;
            float td_error = td_target - q_value;
            for (uint32_t i = 0; i < input_size; ++i) {
                weights[i + j * input_size] += learning_rate * td_error * input.letter_freq[i];
            }
            biases[j] -= learning_rate * td_error;
        }
    }
}


float rand_float()
{
    return rand() / (float) RAND_MAX;
}

float fill_weights_biases(float *weights, uint64_t weights_size, float *biases, uint64_t biases_size)
{
    for(uint32_t i = 0; i < weights_size; ++i) {
        weights[i] = rand_float();
    }
    for(uint32_t i = 0; i < biases_size; ++i) {
        biases[i] = rand_float();
    }
}

float print_weights_biases(float *weights, uint64_t input_size, float *biases, uint64_t biases_size)
{
    for(uint32_t j = 0; j < biases_size; ++j) {
        for(uint32_t i = 0; i < input_size; ++i) {
            cout << weights[i] << " ";
        }
        cout << endl;
    }
    for(uint32_t i = 0; i < biases_size; ++i) {
        cout << biases[i] << endl;
    }
}

string classify(input_t *input, unordered_map<string, uint8_t> *truth_table, float *weights, uint64_t input_size, float *biases, uint64_t output_size)
{
    if(ssize(input->letter_freq) != input_size) return "";

    float *outputs = (float *) malloc(sizeof(float) * output_size);

    uint32_t max_j = 0;
    uint32_t max_activ = 0.0f;
    for(uint32_t j = 0; j < output_size; ++j) {
        float y = 0.0f;
        for(uint32_t i = 0; i < input_size; ++i) {
            y += input->letter_freq[i] * weights[i + j * input_size];
        }
        y = sigmoidf(y - biases[j]);
        if(y > max_activ) {
            max_j = j;
        }
    }

    for(pair<string, uint8_t> truth_pair : *truth_table) {
        if(truth_pair.second == max_j) return truth_pair.first;
    }
    return "";
}

int main(void)
{
    unordered_map<string, uint8_t> truth_table;
    trainset_t trainset;
    get_trainset("lang.train.csv", &trainset, &truth_table);

    trainset_t testset;
    get_trainset("lang.test.csv", &testset, nullptr);

    uint32_t input_size = 26;
    uint32_t output_size = truth_table.size();
    float *weights = (float *) malloc(sizeof(float) * input_size * output_size);
    float *biases = (float *) malloc(sizeof(float) * output_size);
    srand(72);
    fill_weights_biases(weights, input_size * output_size, biases, output_size);
    
    /*
    Best Rate = 0.049999997019767761 Min Cost = 0.88674169778823853
    Best Rate = 0.059999994933605194 Min Cost = 0.30343541502952576
    Best Rate = 0.069999992847442627 Min Cost = 0.16549840569496155
    Best Rate = 0.089999988675117493 Min Cost = 0.10434745997190475
    Best Rate = 0.27000001072883606 Min Cost = 0.070754334330558777
    Best Rate = 0.31999996304512024 Min Cost = 0.054637093096971512
    Best Rate = 0.32999995350837708 Min Cost = 0.030290525406599045
    Best Rate = 0.52999979257583618 Min Cost = 0.022562174126505852
    Best Rate = 0.56999975442886353 Min Cost = 0.010844515636563301
    Best Rate = 0.78999954462051392 Min Cost = 0.0099486857652664185
    Best Rate = 0.90999943017959595 Min Cost = 0.0052288179285824299
    Best Rate = 1.7199987173080444 Min Cost = 0.005014380905777216
    Best Rate = 2.2199983596801758 Min Cost = 0.0025209642481058836
    Best Rate = 3.549997091293335 Min Cost = 0.0012626989046111703
    Best Rate = 6.1100449562072754 Min Cost = 1.0731064321589656e-006
    Best Rate = 10.080135345458984 Min Cost = 7.8977899420351605e-007
    */

    float learning_rate = 0.90999943017959595f;
    float cost = 0.0f;
    for(int32_t i = 0; i < 4000; ++i) {
        update(learning_rate, &trainset, &truth_table, weights, input_size, biases, output_size);
        cost = calc_cost(&trainset, &truth_table, weights, input_size, biases, output_size);
    }
    cout << "Cost = " << cost << endl;

    float accuracy = 0.0f;
    for(input_t input : testset) {
        string classified = classify(&input, &truth_table, weights, input_size, biases, output_size);
        if(classified == input.lang) accuracy += 1.0f;
    }
    accuracy /= (float)testset.size();
    cout << "Accuracy = " << accuracy << endl;

    const int BUFFER_SIZE = 4096;
    char input_buffer[BUFFER_SIZE];
    
    while (true) {
        std::cout << "Enter a title: ";
        std::fgets(input_buffer, BUFFER_SIZE, stdin);
        

        if (std::strlen(input_buffer) > 0 && input_buffer[std::strlen(input_buffer) - 1] == '\n') {
            input_buffer[std::strlen(input_buffer) - 1] = '\0';
        }
        
        if (std::strcmp(input_buffer, "exit") == 0) {
            std::cout << "Exiting program..." << std::endl;
            break;
        }
        
        input_t input = parse_input_line(input_buffer);
        std::cout << "Classification: " << classify(&input, &truth_table, weights, input_size, biases, output_size) << std::endl; // You need to implement classify_input function
    }

    return 0;
}