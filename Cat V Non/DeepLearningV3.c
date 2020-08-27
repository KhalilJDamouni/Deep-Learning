#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <gsl/gsl_blas.h>

//V3 of Cat V NonCat Identifier
//Adding GSL_BLAS for linear algebra operations

typedef struct Node
{
    unsigned char values[64][64][3];
    bool class;
} Node;

//Functions Defs
gsl_vector* sigmoid(gsl_vector* z); 
gsl_vector** propagate(gsl_vector* w, double* b, gsl_matrix* X, gsl_vector* Y);
void optimize(gsl_vector* w, double* b, gsl_matrix* X, gsl_vector* Y, int num_iterations, double learning_rate);
gsl_vector* predict(gsl_vector* w, double* b, gsl_matrix* X);
double calculate_cost(gsl_vector* Y, gsl_vector* A);
double*** flatten(Node** train_set, Node** test_set);
Node** load_train_set(void);
Node** load_test_set(void);
void function_testing(void);


gsl_vector* sigmoid(gsl_vector* z) //Works
{
    //printf("Sigmoid\n");
    gsl_vector_scale(z, -1);
    
    //Have to exp element wise
    for(int index = 0; index < z->size; ++index)
        gsl_vector_set(z, index, exp(gsl_vector_get(z, index)));

    gsl_vector_add_constant(z, 1);
    gsl_vector* s = gsl_vector_alloc(z->size);
    gsl_vector_set_all(s, 1);
    gsl_vector_div(s, z); //s = 1, s <- 1 / z;


    return s;
}

double calculate_cost(gsl_vector* Y, gsl_vector* A)
{
    //printf("Calculate Cost\n");
    gsl_vector* subbed_A = gsl_vector_alloc(A->size);
    gsl_vector* subbed_Y = gsl_vector_alloc(A->size);
    gsl_vector* logged_A = gsl_vector_alloc(A->size);
    gsl_vector* copied_Y = gsl_vector_alloc(Y->size);

    gsl_vector_memcpy(copied_Y, Y);
    gsl_vector_set_all(subbed_A , 1);
    gsl_vector_sub(subbed_A, A);

    gsl_vector_set_all(subbed_Y, 1);
    gsl_vector_sub(subbed_Y, Y);
    
    for(int i = 0; i < A->size; ++i)
    {
        gsl_vector_set(subbed_A, i, log(gsl_vector_get(subbed_A, i)));
        gsl_vector_set(logged_A, i, log(gsl_vector_get(A, i)));
    }
    gsl_vector_mul(copied_Y, logged_A); 
    gsl_vector_mul(subbed_Y, subbed_A);
    gsl_vector_add(copied_Y, subbed_Y);
    double* cost = malloc(sizeof(double));
    gsl_vector* temp = gsl_vector_alloc(copied_Y->size);
    gsl_vector_set_all(temp, 1);
    gsl_blas_ddot(copied_Y, temp, cost);
    cost[0] *= (double)-1 * (double)1 / A->size;
    return cost[0];
}

gsl_vector** propagate(gsl_vector* w, double* b, gsl_matrix* X, gsl_vector* Y)
{
    //printf("Propagate\n");
    gsl_vector** output = malloc(sizeof(gsl_vector*) * 2);

    gsl_vector* B = gsl_vector_alloc(X->size2); //Make B
    gsl_vector_set_all(B, b[0]);
    gsl_blas_dgemv(CblasTrans, 1.0, X, w, 1.0, B);

    gsl_vector* A = sigmoid(B);    
    
    //cost
    double cost = calculate_cost(Y, A);
    //printf("Cost: %f\n", cost);

    //dw = (1 / m) * np.dot(X,(A - Y).T)
    gsl_vector_sub(A, Y); //Stored in A
    gsl_vector* C = gsl_vector_calloc(X->size1);
    gsl_blas_dgemv(CblasNoTrans, 1.0, X, A, 1.0, C);
    gsl_vector_scale(C, (double)1/X->size2);

    //db
    double* db = malloc(sizeof(double));
    db[0] = 0;
    gsl_vector* temp = gsl_vector_alloc(A->size);
    gsl_vector_set_all(temp, 1);
    gsl_blas_ddot(A, temp, db);
    db[0] *= ((double)1 / (double)X->size2);

    output[0] = C;
    output[1] = gsl_vector_alloc(1);
    gsl_vector_set(output[1], 0, db[0]);
    
    return output;
}

void optimize(gsl_vector* w, double* b, gsl_matrix* X, gsl_vector* Y, int num_iterations, double learning_rate)
{
    //printf("Optimize\n");
    gsl_vector** prop;
    gsl_vector* dw;

    for(int i = 0; i < num_iterations; ++i)
    {
        //printf("Iteration: %i\n", i);
        
        prop = propagate(w, b, X, Y);
        dw   = prop[0];
        double db = gsl_vector_get(prop[1], 0);
        gsl_vector_scale(dw, learning_rate);
        gsl_vector_sub(w, dw);
        b[0] = b[0] - (learning_rate * db);        
    }

    return;
}

gsl_vector* predict(gsl_vector* w, double* b, gsl_matrix* X)
{
    //printf("Predict\n");
    gsl_vector* y_prediction = gsl_vector_calloc(X->size2);
    
    gsl_vector* B = gsl_vector_calloc(X->size2); //Make B
    gsl_vector_set_all(B, b[0]);
    gsl_blas_dgemv(CblasTrans, 1.0, X, w, 1.0, B);

    gsl_vector* A = sigmoid(B); 
    
    for(int i = 0; i < A->size; ++i)
    {
        if((double)gsl_vector_get(A, i) <= (double)0.5)
        {
            gsl_vector_set(y_prediction, i, 0);
        }
        else
        {   
            gsl_vector_set(y_prediction, i, 1);
        }
    }

    return y_prediction;
}

void model(gsl_matrix* X_train, gsl_vector* Y_train, gsl_matrix* X_test, gsl_vector* Y_test, int num_iterations, double learning_rate)
{
    printf("Model\n");
    gsl_vector* w = gsl_vector_calloc(12288);
    double* b = malloc(sizeof(double));
    b[0] = 0;

    optimize(w, b, X_train, Y_train, num_iterations, learning_rate);

    gsl_vector* prediction_test = predict(w, b, X_train);
    

    //Train Accuracy
    int total = 0;
    for(int i = 0; i < prediction_test->size; ++i)
        if(gsl_vector_get(prediction_test, i) == gsl_vector_get(Y_train, i))
            ++total;

    printf("\nTrain Accuracy: %0.2f%%\n", (double)total / prediction_test->size * 100);

    prediction_test = predict(w, b, X_test);

    //Test Accuracy
    total = 0;
    for(int i = 0; i < prediction_test->size; ++i)
        if((int)gsl_vector_get(prediction_test, i) == (int)gsl_vector_get(Y_test, i))
            ++total;

    printf("Test Accuracy: %0.2f%%\n", (double)total / prediction_test->size * 100);
    
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

void function_testing(void)
{
    //Sigmoid #Works
    gsl_vector* s = gsl_vector_alloc(2);
    gsl_vector_set(s, 0, 0); gsl_vector_set(s, 1, 2);
    s = sigmoid(s);
    printf("%g %g\n\n", gsl_vector_get(s, 0), gsl_vector_get(s, 1));   

    //Propogate #Works
    gsl_vector* w = gsl_vector_alloc(2);
    gsl_vector_set(w, 0, 1); gsl_vector_set(w, 1, 2);
    double* b = malloc(sizeof(double));
    b[0] = 2;
    gsl_matrix* X = gsl_matrix_alloc(2, 3);
    gsl_matrix_set(X, 0, 0, 1); gsl_matrix_set(X, 0, 1, 2); gsl_matrix_set(X, 0, 2, -1); gsl_matrix_set(X, 1, 0, 3); gsl_matrix_set(X, 1, 1, 4); gsl_matrix_set(X, 1, 2, -3.2);
    gsl_vector* Y = gsl_vector_alloc(3);
    gsl_vector_set(Y, 0, 1); gsl_vector_set(Y, 1, 0); gsl_vector_set(Y, 2, 1); 

    gsl_vector** output = propagate(w, b, X, Y);
    printf("%g %g\n", gsl_vector_get(output[0], 0), gsl_vector_get(output[0], 1));   
    printf("%g\n\n", gsl_vector_get(output[1], 0));   
    /*

    //Optimize #Works
    optimize(w, b, X, Y, 100, 0.009);
    printf("%g %g\n", gsl_vector_get(w, 0), gsl_vector_get(w, 1));   
    printf("%g\n\n", b[0]);

    //Predict #Works
    gsl_vector_set(w, 0, 0.1124579); gsl_vector_set(w, 1, 0.23106775);
    b[0] = -0.3;
    gsl_matrix_set(X, 0, 0, 1); gsl_matrix_set(X, 0, 1, -1.1); gsl_matrix_set(X, 0, 2, -3.2); gsl_matrix_set(X, 1, 0, 1.2); gsl_matrix_set(X, 1, 1, 2); gsl_matrix_set(X, 1, 2, 0.1);
    gsl_vector* y_pred = gsl_vector_alloc(3);
    y_pred = predict(w, b, X);
    printf("%g %g %g\n", gsl_vector_get(y_pred, 0), gsl_vector_get(y_pred, 1), gsl_vector_get(y_pred, 2));  */ 

    //Percent Right
    /*gsl_vector* prediction_test = gsl_vector_calloc(3);
    gsl_vector* Y_train = gsl_vector_alloc(3);
    gsl_vector_set(Y_train, 0, 1); gsl_vector_set(Y_train, 1, 1); gsl_vector_set(Y_train, 2, 1);*/

    //Train Accuracy
    /*int total = 0;
    for(int i = 0; i < prediction_test->size; ++i)
        if(gsl_vector_get(prediction_test, i) == gsl_vector_get(Y_train, i))
            ++total;

    printf("Train Accuracy: %0.2f%%\n", (double)total / prediction_test->size * 100);*/
}

int main(void)
{
    //function_testing();
    
    //return 0;

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

    int size = 0;
    while(train_set_x[size])
        ++size;
    //printf("%i\n", size);
    gsl_matrix* train_set_x_matrix = gsl_matrix_alloc(12288, size);
    for(int i = 0; i < size; ++i)
        for(int d = 0; d < 12288; ++d)
            gsl_matrix_set(train_set_x_matrix, d, i, (double)train_set_x[i][d]);

    gsl_vector* train_set_y_vector = gsl_vector_alloc(size);
    for(int i = 0; i < size; ++i)
    {
        //printf("%f ", train_set_y[i][0]);
        gsl_vector_set(train_set_y_vector, i, train_set_y[i][0]);
        //printf("%g\n", gsl_vector_get(train_set_y_vector, i));
    }
    size = 0;
    while(test_set_x[size])
        ++size;
    //printf("%i\n", size);
    gsl_matrix* test_set_x_matrix = gsl_matrix_alloc(12288, size);
    for(int i = 0; i < size; ++i)
        for(int d = 0; d < 12288; ++d)
            gsl_matrix_set(test_set_x_matrix, d, i, ((double)test_set_x[i][d]));

    gsl_vector* test_set_y_vector = gsl_vector_alloc(size);
    for(int i = 0; i < size; ++i)
        gsl_vector_set(test_set_y_vector, i, test_set_y[i][0]);


    //Start Clock
    clock_t begin = clock();
    model(train_set_x_matrix, train_set_y_vector, test_set_x_matrix, test_set_y_vector, 2000, 0.005);
   
    //Calculate Time
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Runtime: %0.2fs\n", time_spent);

    return 0;
}