#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

//V2 of Cat V NonCat Identifier
//Im Making this general Debug
//Adds GMP Library (Maybe move this to third)
//Linear Algebra will be third Version
//GMP Needed because I think all the values are going to 0 and need more precision
//Plus Various Cleanups.

typedef struct Node
{
    unsigned char values[64][64][3];
    bool class;
} Node;

Node** load_train_set(void);
Node** load_test_set(void);

double*** flatten(Node** train_set, Node** test_set);
double* sigmoid(double* z);

double flatten_train_set_x[12288][209];
double flatten_train_set_y[1][209];
double flatten_test_set_x[12288][50];
double flatten_test_set_y[1][50];
double cost_function(double* A, double** Y);
double* propagate(double* w, double* b, double** X, double** Y);
void optimize(double* w, double* b, double** X, double** Y, int num_iterations, double learning_rate);
int* predict(double* w, double* b, double** X);
void model(double** X_train, double** Y_train, double** X_test, double** Y_test ,int num_iterations, double learning_rate);



void model(double** X_train, double** Y_train, double** X_test, double** Y_test ,int num_iterations, double learning_rate)
{
    //printf("Model\n");
    double* w = malloc(sizeof(double) * 12288);
    double* b = malloc(sizeof(double));

    optimize(w, b, X_train, Y_train, num_iterations, learning_rate);

    int* prediction_test;
    prediction_test = predict(w, b, X_train);

    /*printf("\nFINAL WEIGHTS: ");
    for(int i = 0; i < 3; ++i)
        printf("%f\t", w[i]);
    printf("%f\n", b[0]);*/

    //for(int i = 0; i < 209; ++i)
    //    printf("%i\n", prediction_test[i]);

    //TRAIN ACCURACY
    int total = 0;
    for(int m = 0; m < 209; ++m)
        if(prediction_test[m] == Y_train[m][0])
            total += 1;

    //printf("Total: %i\n", total);

    double precentage_right = (double)total / 209;

    printf("Train Accuracy: %0.2f%%\n", precentage_right*100);

    //TEST ACCURACY
    prediction_test = predict(w, b, X_test);
    
    total = 0;
    for(int m = 0; m < 50; ++m)
        if(prediction_test[m] == Y_test[m][0])
            total += 1;

    //printf("Total: %i\n", total);

    precentage_right = (double)total / 50;

    printf("Test  Accuracy: %0.2f%%\n", precentage_right * 100);

}

int* predict(double* w, double* b, double** X)
{
    //printf("predict\n");
    int size = 0;
    while(X[size])
        ++size; 
    //printf("%i\n", size);
    
    int* prediction = malloc(sizeof(int) * size);

    double* A;
    
    double* z = malloc(sizeof(double) * size);

    for(int m = 0; m < size; ++m)
    {
        double sum = b[0];

        for(int i = 0; i < 12288; ++i)
        {
           sum = sum + w[i] * X[m][i];
        }

        z[m] = sum;
    }

   /* for(int i = 0; i < 209; ++i)
    {
        printf("z[%i]: %0.25f\t", i, z[i]);
    }*/

    A = sigmoid(z);
    /*for(int i = 0; i < 209; ++i)
    {
        //printf("A[%i]: %0.25f\t", i, A[i]);
    }*/

    for(int m = 0; m < size; ++m)
    {
        if(A[m] <= 0.5)
            prediction[m] = 0;
        else
            prediction[m] = 1;
    }

    return prediction;
}

void optimize(double* w, double* b, double** X, double** Y, int num_iterations, double learning_rate)
{
    //printf("Optimize\n");
    double dw[12288];
    double db;
    double cost;

    for(int iteration = 0; iteration < num_iterations; ++iteration)
    {
        printf("Iteration %i\n", iteration);
        double* prop = propagate(w, b, X, Y);

        cost = prop[0];
        db = prop[1];
        for(int i = 0; i < 12288; ++i)
            dw[i] = prop[i + 2];

        b[0] = b[0] - (learning_rate * db);
        for(int i = 0; i < 12288; ++i)
            w[i] = w[i] - (learning_rate * dw[i]);
        
        //printf("\nCost: %f\n", cost);

        /*for(int i = 0; i < 3; ++i)
            printf("w[%i]: %f\t",i,w[i]);
        printf("b: %f\n", b[0]);*/
    }

}

double* propagate(double* w, double* b, double** X, double** Y) //HERE
{
    //printf("propogate\n");
    double* output = malloc(sizeof(double) * (12288 + 2));

    int size = 0;
    while(X[size])
        ++size;

    double* z = malloc(sizeof(double) * size);

    for(int m = 0; m < size; ++m)
    {
        z[m] = b[0];
        for(int i = 0; i < 12288; ++i)
        {
            z[m] = z[m] + (X[m][i] * w[i]);
        }
    }

    double* A;

    A = sigmoid(z);

    //Cost
    output[0] = cost_function(A, Y);

    //db
    double sum = 0;
    for(int m = 0; m < size; ++m)
    {   
        sum = sum + (A[m] - Y[m][0]);
    }
    //printf("output[1]: %f\n", sum);
    //printf("output[1]: %f\n", output[1] = (1 / (double)(size + 1)) * sum);
    output[1] = (1 / (double)(size + 1)) * sum;

    //dw
    for(int i = 0; i < 12288; ++i)
    {
        output[i + 2] = b[0];
        double sum = 0;
        for(int m = 0; m < 209; ++m)
        {
            sum = sum + X[m][i] * (A[m] - Y[m][0]);
        }
        output[i + 2] = (1 / (double)(size + 1)) * sum;
    }
    /*for(int i = 0; i < 3; ++i)
    {
        printf("dw[%i]: %f\n", i, output[i + 2]);
    }*/
    //printf("db: %f\t", output[1]);
    //[0] - cost
    //[1] - db
    //[2:end] - dw
    //for(int i = 0; i < 12288; ++i)
    //    printf("%f ", output[i + 2]);

    return output;

}

double cost_function(double* A, double** Y)
{
    //printf("cost_function\n");
    double sum = 0;
    int i = 0;
    while(Y[i])
    {
        //printf("Sum: %f\t", sum);
        sum = sum + (Y[i][0] * log(A[i])) + ((double)1 - Y[i][0]) * log((double)1 - A[i]);
        ++i;
    }
    
    sum = -1 * (1 / (double)(i + 1)) * sum;

    return sum;
}

double* sigmoid(double* z)
{
    //printf("sigmoid\n");
    //Input: 1 x something array

    double* output = malloc(209 * sizeof(double));
    int i = 0;
    while(i < 209) //FIX
    {
        
        output[i] = 1 / (1 + exp(-1 * z[i]));
        ++i;
    }

    return output;
}

double*** flatten(Node** train_set, Node** test_set)
{
    //printf("Flatten\n");
    //Format:
        //Output[0] = "Train Set X"
        //Output[1] = "Train Set Y"
        //Output[2] = "Test Set X"
        //Output[3] = "Test Set Y"

        //Each list terminated with \0 for iteration purposes

    int train_size = 0;
    while(train_set[train_size])
        ++train_size;
    int test_size = 0;
    while(test_set[test_size])
        ++test_size;

    double*** output = malloc(sizeof(double**) * (train_size + test_size));

    //Grab Train Set
    double** train_set_x = malloc(sizeof(double*) * (train_size + 1));
    double** train_set_y = malloc(sizeof(double*) * (train_size + 1));
    
    train_set_x[train_size] = (double*)'\0';
    train_set_y[train_size] = (double*)'\0';

    for(int m = 0; m < train_size; ++m)
    {   
        double* shape = malloc(sizeof(double) * 12288);
        double* class = malloc(sizeof(double));
        
        unsigned char* position = *(train_set[m]->values[0]);
        
        for(int index = 0; index < 12288; ++index)
        {
            shape[index] = (*position++);
            shape[index] /= 255;
        }

        train_set_x[m] = shape;
        class[0] = (double)train_set[m]->class;
        train_set_y[m] = class;
    }

    output[0] = train_set_x;
    output[1] = train_set_y;

    //Grab Test Set
    double** test_set_x = malloc(sizeof(double*) * (test_size + 1));
    double** test_set_y = malloc(sizeof(double*) * (test_size + 1));
    
    test_set_x[test_size] = (double*)'\0';
    test_set_y[test_size] = (double*)'\0';

    for(int m = 0; m < test_size; ++m)
    {   
        double* shape = malloc(sizeof(double) * 12288);
        double* class = malloc(sizeof(double));
        
        unsigned char* position = *(test_set[m]->values[0]);
        
        for(int index = 0; index < 12288; ++index)
        {
            shape[index] = (*position++);
            shape[index] /= 255;
        }

        test_set_x[m] = shape;
        class[0] = (double)test_set[m]->class;
        test_set_y[m] = class;
    }

    output[2] = test_set_x;
    output[3] = test_set_y;

    return output;
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

int main(void)
{
    Node** train_set_full = load_train_set();
    Node** test_set_full = load_test_set();

    double*** flattened;
    flattened = flatten(train_set_full, test_set_full);
    
    //Unpack Flattened Structure
    double** train_set_x;
    double** train_set_y;
    double** test_set_x;
    double** test_set_y;

    train_set_x = flattened[0];    
    train_set_y = flattened[1];    
    test_set_x = flattened[2];    
    test_set_y = flattened[3];    

    //Debug
    
    int i = 0;
    /*
    while(train_set_y[i]) 
    {
        printf("%i: %f\n", i, train_set_y[i][0]);
        ++i;
    }
    i = 0;
    while(test_set_y[i])
    {
        printf("%i: %f\n", i, test_set_y[i][0]);
        ++i;
        
    }*/
    
    model(train_set_x, train_set_y, test_set_x, test_set_y, 10, 0.5);


    return 0;
}