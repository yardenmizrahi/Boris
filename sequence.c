#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "mpi.h"
#include <stdbool.h>
#define COLS 5

double computeX(double x1, double x2, double t) {
    return ((x2 - x1) / 2.0) * sin(t * M_PI / 2.0) + (x2 + x1) / 2.0;
}

double computeY(double x, double a, double b) {
    return a * x + b;
}
double euclideanDistance(double x1, double y1, double x2, double y2) {
    return sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));
}

int satisfiesProximityCriteria(int currentIndex, double t, double D, int N, double* x1, double* x2, double* a, double* b , int K) {
    int count = 0;

    for (int i = 0; i < N; i++) {
        if (i != currentIndex) {
            double distance = euclideanDistance(
                computeX(x1[currentIndex], x2[currentIndex], t),
                computeY(computeX(x1[currentIndex], x2[currentIndex], t), a[currentIndex], b[currentIndex]),
                computeX(x1[i], x2[i], t),
                computeY(computeX(x1[i], x2[i], t), a[i], b[i])
            );

            if (distance < D) {
                count++;
            }
        }
    }

    return count >= K; // returns 1 (true) if the criteria is satisfied, otherwise 0 (false)
}


int main(int argc, char* argv[]){
	int my_rank; /* rank of process */
	int num_proc;       /* number of processes */
	MPI_Status status ;   /* return status for receive */
	
	FILE *file;
	int N; /* number of points */
	int K; /* minimal number of points to satisfy the Proximity Criteria */
	double D; /* distance */
	int TCount; /* given count of t values */
	
	double* x1; /* all x1 ordered by index */
	double* x2; /* all x2 ordered by index */
	double* a; /* all a ordered by index */
	double* b; /* all b ordered by index */
	int* id; /* all b ordered by index */
	
	double** matrix; /* all values from the input.txt file except the first row and the id's col */
	int i, j; /* for loops */
	
	double* t; /* all t calculations accordig to the formula */
	double temp_x, temp_y, temp_current_x, temp_current_y; /* temporary points for the main calculation */
	double temp_d; /* temporary distance for each point */
	int* temp_result; /* temporary result - PC points */
	int** all_results; /* all of the results */
	int counter; /* to count whether there are at least k or 3 point that satisfing PC */
	
	double start, end, elapsed_time; /* for time measurment */

	
	/* start up MPI */	
	MPI_Init(&argc, &argv);
	
	/* find out process rank */
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); 
	
	/* find out number of processes */
	MPI_Comm_size(MPI_COMM_WORLD, &num_proc); 
	
	start = MPI_Wtime();
	
	/* read input.txt */
	file = fopen("input.txt", "r"); //open
    	if (file == NULL) {
        	fprintf(stderr, "Error opening the file\n");
       		exit(1);
    	}
    	
    	/* read the first row */
    	if (fscanf(file, "%d %d %lf %d", &N, &K, &D, &TCount) != 4) {
        	fprintf(stderr, "Error reading the first line\n");
        	fclose(file);
        	exit(1);
    	}
    	
    	// Print the values of N, K, D, and TCount
  //  	printf("N = %d, K = %d, D = %lf, TCount = %d\n", N, K, D, TCount);
    	
    	/* Read the matrix */
    	matrix = (double**)malloc(N * sizeof(double*));
    	if(matrix == NULL) {
    		fprintf(stderr, "Memory allocation failed for rows!\n");
        	exit(1);
    	}
    	
    	for(i = 0; i < N; i++) {
        	// For each row pointer, allocate memory for COLS columns
        	matrix[i] = (double *)malloc(COLS * sizeof(double));
        	if(matrix[i] == NULL) {
            		fprintf(stderr, "Memory allocation failed for columns at row %d!\n", i);
           		exit(1);
        	}
        	// Read values from the file for this row
        	for(j = 0; j < COLS; j++) {
            		if (fscanf(file, "%lf", &(matrix[i][j])) != 1) {
                		fprintf(stderr, "Error reading value at row %d, column %d from input.txt\n", i, j);
                		exit(1);
            		}
        	}
    	}

    	// Close the file
    	fclose(file);
    	
    	//built x1, x2, a, b arrays
    	x1 = (double*)malloc(N * sizeof(double));
    	x2 = (double*)malloc(N * sizeof(double));
    	a = (double*)malloc(N * sizeof(double));
    	b = (double*)malloc(N * sizeof(double));
    	id = (int*)malloc(N * sizeof(double));
    	
    	for(i = 0; i < COLS; i++) {
    		int c = 0;
    		for(j = 0; j < N; j++) {
	    		if(i == 0) {
	    			id[c] = matrix[j][i];
	    			}
    			if(i == 1) {
    				x1[c] = matrix[j][i];
    			}
    			if(i == 2) {
    				x2[c] = matrix[j][i];
    			}
    			if(i == 3) {
    				a[c] = matrix[j][i];
    			}
    			if(i == 4) {
    				b[c] = matrix[j][i];
    			}
    			c++;
    		}
	}
    	
    	//built array t
    	t = (double*)malloc((TCount+1)*sizeof(double));
    	for(i = 0; i < TCount + 1; i++) {
    		t[i] = 2.0 * i / TCount - 1;
    	}
    		
    	FILE* outputFile = fopen("output.txt", "w");

	for (int i = 0; i <= TCount; i++) {
	    double t = 2.0 * i / TCount - 1.0;

	    int satisfiedPointsCount = 0;
	    int satisfiedPoints[3]; // To store the IDs of points that satisfy the proximity criteria.

	    for (int j = 0; j < N; j++) {
		if (satisfiesProximityCriteria(j, t, D, N, x1, x2, a, b, K)) {
		    satisfiedPoints[satisfiedPointsCount] = id[j];
		    satisfiedPointsCount++;

		    if (satisfiedPointsCount == 3) {
		        fprintf(outputFile, "Points %d, %d, %d satisfy Proximity Criteria at t = %lf\n",
		                satisfiedPoints[0], satisfiedPoints[1], satisfiedPoints[2], t);
		        break; // break out of the loop as soon as we find 3 points
		    }
		}
	    }
	}

	if (ftell(outputFile) == 0) { // If nothing is written yet, no points were found.
	    fprintf(outputFile, "No points satisfy the Proximity Criteria.\n");
	}

	fclose(outputFile);


	
	end = MPI_Wtime();
	elapsed_time = end - start;
	
	printf("MPI_Wtime measured to be: %.9f\n", elapsed_time);
	fflush(stdout);
    	

	/* shut down MPI */
	MPI_Finalize(); 
	
	
	return 0;
}

