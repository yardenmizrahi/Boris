#include <mpi.h>
#include <omp.h>
#include <stdio.h>

#include <stdlib.h>
#include <math.h>
#include <string.h>


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


