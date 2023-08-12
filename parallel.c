#include <mpi.h>
#include <omp.h>
#include <stdio.h>

#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "myProto.h"


int main(int argc, char* argv[]){
	/*
	MPI+OpenMP+CUDA Integration

	*/
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
	
	double start_time, end_time, time; /* for time measurment */

	
	/* start up MPI */	
	MPI_Init(&argc, &argv);
	
	/* find out process rank */
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); 
	
	/* find out number of processes */
	MPI_Comm_size(MPI_COMM_WORLD, &num_proc); 
	
	
	if(my_rank == MASTER)
	{
		start_time = MPI_Wtime();
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
	    		    	
	    	MPI_Send(&N, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
		MPI_Send(&K, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
		MPI_Send(&D, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
		MPI_Send(&TCount, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
	    	
	    	// Print the values of N, K, D, and TCount
	   // 	printf("N = %d, K = %d, D = %f, TCount = %d\n", N, K, D, TCount);
	    	
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
	    		
	    	//build x1, x2, a, b arrays
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
	    	
	    	MPI_Send(id, N, MPI_INT, 1, 0, MPI_COMM_WORLD);
	    	MPI_Send(x1, N, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
		MPI_Send(x2, N, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
		MPI_Send(a, N, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
		MPI_Send(b, N, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);

	    	// Close the file
	    	fclose(file);

	}
	if(my_rank == 1) {
		MPI_Recv(&N, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(&K, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(&D, 1, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(&TCount, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status);
		
		//build x1, x2, a, b arrays
		id = (int*)malloc(N * sizeof(double));
	    	MPI_Recv(id, N, MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status);	
	    	x1 = (double*)malloc(N * sizeof(double));
	    	MPI_Recv(x1, N, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD, &status);
	    	x2 = (double*)malloc(N * sizeof(double));
	    	MPI_Recv(x2, N, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD, &status);
	    	a = (double*)malloc(N * sizeof(double));
	    	MPI_Recv(a, N, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD, &status);
	    	b = (double*)malloc(N * sizeof(double));
	    	MPI_Recv(b, N, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD, &status);
    	}
    	
    	
    //	printf("I am process %d and I know that N = %d, K = %d, D = %lf, TCount = %d\n", my_rank, N, K, D, TCount);
    	int part = TCount / 2;
    		
    	FILE* outputFile = fopen("output.txt", "w");

#pragma omp parallel for num_threads(4) private(j)
	for (int i = 0; i <= part; i++) {
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
	
	int* results[3];
	double* t_results;
	
	// On each process - perform a second half of its task with CUDA
	if (computeOnGPU(results, t_results, part + 1, D, N, K, TCount, x1, x2, a, b, id) != 0)
		MPI_Abort(MPI_COMM_WORLD, __LINE__);
		
	for (int i = 0; i < N; i++) {
		if(results[i][0] != 0) { // check if there's a message at this index
			fprintf(outputFile, "Points %d, %d, %d satisfy Proximity Criteria at t = %lf\n",
				results[i][0], results[i][1], results[i][2], t_results[i]);
		}
	}

	if (ftell(outputFile) == 0) { // If nothing is written yet, no points were found.
	    fprintf(outputFile, "No points satisfy the Proximity Criteria.\n");
	}

	fclose(outputFile);
	if(my_rank == MASTER) {
		end_time = MPI_Wtime();
		time = end_time - start_time;
		printf("MPI_Wtime measured to be: %.9f\n", time);
		fflush(stdout);
	}



	/* shut down MPI */
	MPI_Finalize(); 
	
	
	return 0;
}

