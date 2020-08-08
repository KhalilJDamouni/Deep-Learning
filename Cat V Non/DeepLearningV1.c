#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

typedef struct Node
{
    unsigned char values[64][64][3];
    bool class;
} Node;

Node** load_train_set(void);
Node** load_test_set(void);
void flatten(Node** train_set,  Node** test_set);
double* sigmoid(double z[209]);

double flatten_train_set_x[12288][209];
double flatten_train_set_y[1][209];
double flatten_test_set_x[12288][50];
double flatten_test_set_y[1][50];
double cost_function(double A[209], double Y[1][209]);
double* propagate(double* w, double b, double X[12288][209], double Y[1][209]);
double** optimize(double* w, double b, double X[12288][209], double Y[1][209], int num_iterations, double learning_rate);
int* predict(double* w, double b, double X[12288][209]);
void model(int num_iterations, double learning_rate);

int main(void)
{
    Node** train_set = load_train_set();
    Node** test_set = load_test_set();
    
    flatten(train_set, test_set);

    //DEBUG
    for(int i = 0; i < 12288; ++i)
    {
        printf("%f ", flatten_train_set_x[0][i]);
    }

    model(20, 0.5);


    return 0;
}

void model(int num_iterations, double learning_rate)
{
    double* w = malloc(sizeof(double) * 12288);
    double b = 0;

    optimize(w, b, flatten_train_set_x, flatten_train_set_y, num_iterations, learning_rate);

    int* prediction_test;
    prediction_test = predict(w, b, flatten_train_set_x);

    int total = 0;

    for(int m = 0; m < 209; ++m)
        if(prediction_test[m] == flatten_train_set_y[0][m])
            total += 1;
    
    double precentage_right = total / 209;

    printf("Train Accuracy: %f\n", precentage_right);

}

int* predict(double* w, double b, double X[12288][209])
{
    int* prediction = malloc(sizeof(int) * 209);

    double* A;
    
    double z[209];

    for(int m = 0; m < 209; ++m)
    {
        int sum = b;

        for(int i = 0; i < 12288; ++i)
        {
           sum += w[i] * X[i][m];
        }

        z[m] = 209;
    }

    A = sigmoid(z);

    for(int m = 0; m < 209; ++m)
    {
        if(A[m] <= 0.5)
            prediction[m] = 0;
        else
            prediction[m] = 1;
    }

    return prediction;
}

double** optimize(double* w, double b, double X[12288][209], double Y[1][209], int num_iterations, double learning_rate)
{
    double dw[12288];
    double db;
    double cost;

    for(int iteration = 0; iteration < num_iterations; ++iteration)
    {
        double* prop = propagate(w, b, X, Y);

        cost = prop[0];
        db = prop[1];
        for(int i = 0; i < 12288; ++i)
            dw[i] = prop[i + 2];

        b = b - (learning_rate * db);
        for(int i = 0; i < 12288; ++i)
            w[i] = w[i] - (learning_rate * dw[i]);
        
        printf("\nCost: %f\n", cost);

        for(int i = 0; i < 12288; ++i)
            printf("%f ",w[i]);

    }

    double** output = malloc(sizeof(double));

    return output;
}

double* propagate(double* w, double b, double X[12288][209], double Y[1][209])
{

    double* output = malloc(sizeof(double) * (12288 + 2));

    //w.T * X + b
    double z[209];

    for(int i = 0; i < 209; ++i)
    {
        double dot = b;
        for(int a = 0; a < 12288; ++a)
        {
            dot += X[a][i] * w[a];
        }
        z[i] = dot;
    }

    double* A;
    A = sigmoid(z);
    
    output[0] = cost_function(A, Y);
    
    
    double db = 0;
    double dw = 0;
    for(int i = 0; i < 209; ++i)
    {
        db += (A[i] - Y[1][i]);
    }
    db = (1 / 209) * db;

    output[1] = db;

    for(int i = 0; i < 12288; ++i)
    {
        double sum = 0;
        for(int a = 0; a < 209; ++a)
        {
            sum += X[i][a] * (A[a] - Y[1][a]);
        }
        sum = (1 / 209) * sum;
        output[i + 2] = sum;
    }

    //Output Format: 
    //[0] - cost
    //[1] - db
    //[2:end] - dw
    for(int i = 0; i < 12288; ++i)
        printf("%f ", output[i + 2]);
        
    return output;
}

double cost_function(double A[209], double Y[1][209])
{

    double sum = 0;

    for(int i = 0; i < 209; ++i)
    {
        sum += (Y[0][i] * log10(A[i])) + (1 - Y[0][i]) * log10(1 - A[i]);
    }
    sum = -1 * (1 / 209) * sum;

    return sum;
}

double* sigmoid(double z[209])
{
    //Input: 1 x something array

    double* output = malloc(209 * sizeof(double));
    
    for(int i = 0; i < 209; ++i)
    {
        output[i] = 1 / (1 + exp(-1 * z[i]));
    }

    return output;
}

void flatten(Node** train_set,  Node** test_set)
{
    //TRAIN SET
    for(int m = 0; m < 209; ++m)
    {
        unsigned char* position = *(train_set[m]->values[0]);
        for(int index = 0; index < 12288; ++index)
        {
            flatten_train_set_x[index][m] = (*position++);
            flatten_train_set_x[index][m] /= 255;
        }
        
        flatten_train_set_y[0][m] = train_set[m]->class;
    }
    
    //TEST SET
    for(int m = 0; m < 50; ++m)
    {
        unsigned char* position = *(test_set[m]->values[0]);
        for(int index = 0; index < 12288; ++index)
        {
            flatten_test_set_x[index][m] = (*position++);
            flatten_test_set_x[index][m] /= 255;
        }
        
        flatten_test_set_y[0][m] = test_set[m]->class;
    }
    
}

Node** load_train_set(void)
{
    Node** train_set = malloc(sizeof(Node*) * 209);
    
    char num[10];

    for(int i = 0; i < 209; ++i)
    {
        //printf("x_train_set #%i\n", i);
        Node* current_entry = malloc(sizeof(Node));

        char path[50] = "Plain_Files/Train/x/";
        sprintf(num, "%i", i);
        strncat(path, num, 50);
        char txt[] = ".txt";
        strncat(path, txt, 50);
        //printf("%s\n", path);
        FILE *x_train_set_input = fopen(path, "r");

        char current_line[258]; 
        fgets(current_line, 258, x_train_set_input);
        
        int current_value = atoi(strtok(current_line, " "));
        
        for(int a = 0; a < 3; ++a)
        {   
            fgets(current_line, 258, x_train_set_input);
            fgets(current_line, 258, x_train_set_input);
            for(int b = 0; b < 64; ++b)
            {
                fgets(current_line, 258, x_train_set_input);
                current_value = atoi(strtok(current_line, " "));
                for(int c = 0; c < 64; ++c)
                {  
                    current_entry->values[c][b][a] = current_value;
                    //printf("%i ", current_value);

                    if(c == 63 && a == 2 && b == 63)
                        break;
                    current_value = atoi(strtok(NULL, " "));
                }
            }
        }

        fclose(x_train_set_input);

        //Create PATH
        char path2[50] = "Plain_Files/Train/y/";
        strncat(path2, num, 50);
        strncat(path2, txt, 50);
        //printf("%s\n", path2);
        FILE *y_train_set_input = fopen(path2, "r");


        //Get Class
        fgets(current_line, 258, y_train_set_input);
        current_value = atoi(strtok(current_line, " "));
        current_entry->class = current_value;

        fclose(y_train_set_input);

        train_set[i] = current_entry;

    }

    return train_set;
}

Node** load_test_set(void)
{
    Node** test_set = malloc(sizeof(Node*) * 50);
    
    char num[10];

    for(int i = 0; i < 50; ++i)
    {
        //printf("x_test_set #%i\n", i);
        Node* current_entry = malloc(sizeof(Node));

        char path[50] = "Plain_Files/Test/x/";
        sprintf(num, "%i", i);
        strncat(path, num, 50);
        char txt[] = ".txt";
        strncat(path, txt, 50);
        //printf("%s\n", path);
        FILE *x_test_set_input = fopen(path, "r");

        char current_line[258]; 
        fgets(current_line, 258, x_test_set_input);
        
        int current_value = atoi(strtok(current_line, " "));
        
        for(int a = 0; a < 3; ++a)
        {   
            fgets(current_line, 258, x_test_set_input);
            fgets(current_line, 258, x_test_set_input);
            for(int b = 0; b < 64; ++b)
            {
                fgets(current_line, 258, x_test_set_input);
                current_value = atoi(strtok(current_line, " "));
                for(int c = 0; c < 64; ++c)
                {  
                    current_entry->values[c][b][a] = current_value;
                    //printf("%i ", current_value);

                    if(c == 63 && a == 2 && b == 63)
                        break;
                    current_value = atoi(strtok(NULL, " "));
                }
            }
        }

        fclose(x_test_set_input);

        //Create PATH
        char path2[50] = "Plain_Files/Test/y/";
        strncat(path2, num, 50);
        strncat(path2, txt, 50);
        //printf("%s\n", path2);
        FILE *y_test_set_input = fopen(path2, "r");


        //Get Class
        fgets(current_line, 258, y_test_set_input);
        current_value = atoi(strtok(current_line, " "));
        current_entry->class = current_value;

        fclose(y_test_set_input);

        test_set[i] = current_entry;

    }

    return test_set;
}