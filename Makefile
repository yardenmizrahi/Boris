build:
	mpicxx -fopenmp -c parallel.c -o main.o -lm
	mpicxx -fopenmp -c cFunctions.c -o cFunctions.o -lm
	mpicxx -fopenmp -o mpiOpemMP  main.o cFunctions.o 

clean:
	rm -f *.o ./mpiOpemMP

run:
	mpiexec -np 2 ./mpiOpemMP

runOn2:
	mpiexec -np 2 -machinefile  mf  -map-by  node  ./mpiOpemMP





