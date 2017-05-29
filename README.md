Compilation instructions:
-----------------------------------------------------------------
For OpenMP
----------
1.) g++ -fopenmp -std=c++11 main.cpp 
2.) ./a.out (-h for help)

-----------------------------------------------------------------
For MPI
-----------
1.) mpicc -o mpi_example main.cpp
 mpicc main.cpp -lstdc++ -lm -fopenmp
mpic++ -o mpi main.cpp -fopenmp -std=c++11

2.) mpirun -np <num_procs> mpi_example


seq 263


# concurrent-nbody
# concurrent-nbody
