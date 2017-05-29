#include <stdlib.h>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <string.h>
#include <iterator>
#include <iomanip>
#include <omp.h>
#include <mpi.h>
#include "vector2.h"
#include "common.h"
#include <iostream>
#include <chrono>
using namespace std;

#define ROOT 0

/*
 * Commit particle masses and positions to file in CSV format
 */
void PersistPositions(const string &p_strFilename, std::vector<Particle> &p_bodies)
{
    cout << "Writing to file: " << p_strFilename << endl;

    ofstream output(p_strFilename.c_str());

    if (output.is_open())
    {
        for (int j = 0; j < p_bodies.size(); j++)
        {
            output << 	p_bodies[j].Mass << ", " <<
                   p_bodies[j].Position.Element[0] << ", " <<
                   p_bodies[j].Position.Element[1] << endl;
        }

        output.close();
    }
    else
        cerr << "Unable to persist data to file:" << p_strFilename << endl;

}

/*
 * Sequential functions
 */
void ComputeForcesSeq(std::vector<Particle> &p_bodies, float p_gravitationalTerm, float p_deltaT)
{
    Vector2 direction,
            force, acceleration;

    float distance;

    for (size_t j = 0; j < p_bodies.size(); ++j)
    {
        Particle &p1 = p_bodies[j];

        force = 0.f, acceleration = 0.f;

        for (size_t k = 0; k < p_bodies.size(); ++k)
        {
            if (k == j) continue;

            Particle &p2 = p_bodies[k];
            direction = p2.Position - p1.Position;
            distance = std::max<float>( 0.5f * (p2.Mass + p1.Mass), direction.Length() );
            force += direction / (distance * distance * distance) * p2.Mass;
        }
        acceleration = force * p_gravitationalTerm;
        p1.Velocity += acceleration * p_deltaT;
    }
}
void MoveBodiesSeq(std::vector<Particle> &p_bodies, float p_deltaT)
{
    for (size_t j = 0; j < p_bodies.size(); ++j)
    {
        p_bodies[j].Position += p_bodies[j].Velocity * p_deltaT;
    }
}

/*
 * OMP functions
 */
void ComputeForcesPar(std::vector<Particle> &p_bodies, float p_gravitationalTerm, float p_deltaT, int thread_count)
{
    Vector2 direction,
            force, acceleration;

    float distance;
    omp_lock_t writelock;
    omp_init_lock(&writelock);

    int j,k;
    int n = p_bodies.size();
#pragma omp parallel for private(direction, distance, force) num_threads(thread_count)
    for (j = 0; j < n; ++j)
    {
        Particle &p1 = p_bodies[j];
        force = 0.f, acceleration = 0.f;
        for (k = 0; k < n; ++k)
        {
            if (k == j) continue;
            Particle &p2 = p_bodies[k];

            // Compute direction vector
            direction = p2.Position - p1.Position;
            // Limit distance term to avoid singularities
            distance = max<float>( 0.5f * (p2.Mass + p1.Mass), direction.Length() );
            // Accumulate force
//            omp_set_lock(&writelock);
#pragma omp critical
            force += direction / (distance * distance * distance) * p2.Mass;
//            omp_unset_lock(&writelock);
        }
#pragma omp critical
        acceleration = force * p_gravitationalTerm;
//        omp_set_lock(&writelock);
        p1.Velocity += acceleration * p_deltaT;
//        omp_unset_lock(&writelock);
    };
//omp_destroy_lock(&writelock);
//#pragma omp barrier
}
void MoveBodiesPar(std::vector<Particle> &p_bodies, float p_deltaT)
{
    for (size_t j = 0; j < p_bodies.size(); ++j)
    {
        p_bodies[j].Position += p_bodies[j].Velocity * p_deltaT;
    }
}

/*
 * MPI functions
 */
void ComputeForcesMPI(std::vector<Particle> &p_bodies,int start,int end,float gTerm,float p_deltaT)
{
    Vector2 direction,
            force, acceleration;
    int i,j;
    float distance;

    for (i=start;i<end;++i)
    {
        Particle &p1 = (p_bodies[i]);
        force = 0.f, acceleration = 0.f;

        for (j=0;j<p_bodies.size();++j)
        {
            if (i==j) continue;
            Particle &p2 = p_bodies[j];

            direction = &p2.Position - &p1.Position;
            distance = max<float>( 0.5f * (p2.Mass + p1.Mass), direction.Length() );
            force += direction / (distance * distance * distance) * p2.Mass;
        }
        acceleration = force * gTerm;
        p1.Velocity += acceleration*p_deltaT;
    }
}
void MoveBodiesMPI(std::vector<Particle>&bodies, float p_deltaT)
{
    for (size_t j = 0; j < bodies.size(); ++j)
    {
        bodies[j].Position += bodies[j].Velocity * p_deltaT;
    }
}

/*
 * Hybrid functions
 */
void ComputeForcesHybrid(std::vector<Particle> &p_bodies,int start,int end,
                         float gTerm,float p_deltaT,int thread_count)
{
    Vector2 direction,
            force, acceleration;

    int i,j;
    float distance;
#pragma omp parallel for private(direction, distance, force) num_threads(thread_count)
    for (i=start;i<end;++i)
    {
        Particle &p1 = (p_bodies[i]);
        force = 0.f, acceleration = 0.f;

        for (j=0;j<p_bodies.size();++j)
        {
            if (i==j) continue;
            Particle &p2 = p_bodies[j];

            direction = &p2.Position - &p1.Position;
            distance = max<float>( 0.5f * (p2.Mass + p1.Mass), direction.Length() );
#pragma omp critical
            force += direction / (distance * distance * distance) * p2.Mass;
        }
#pragma omp critical
        p1.Velocity += acceleration*p_deltaT;
        acceleration = force * gTerm;
    }
}
void MoveBodiesHybrid(std::vector<Particle>&bodies,float p_deltaT)
{
#pragma omp parallel for schedule(static)
    for (size_t j = 0; j < bodies.size(); ++j)
    {
        bodies[j].Position += bodies[j].Velocity * p_deltaT;
    }
}

void SequentialRun(std::vector<Particle> &bodies, int maxIteration, float gTerm, float deltaT,
                   char* output_files, int count)
{
    std::stringstream fileOutput;
    std::cout<<"Sequential run: "<<std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i =0; i<count; i++) {
        for (int iteration = 0; iteration < maxIteration; ++iteration) {
            if (iteration % 500 == 0) {
                cout << "iteration: " << iteration << "\n";
            }
            ComputeForcesSeq(bodies, gTerm, deltaT);
            MoveBodiesSeq(bodies, deltaT);

            if (strcmp("n", output_files)) {

                fileOutput.str(std::string());
                fileOutput << "nbody_" << iteration << ".txt";
                PersistPositions(fileOutput.str(), bodies);
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    long dur = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count()/count;
    std::cout<<"Average time taken for sequential: "
             <<dur/1000000000.0
             <<" secs."<<std::endl;
}

void OmpRun(std::vector<Particle> &bodies, int maxIteration, float gTerm, float deltaT,
            int thread_count, char* output_files, int count)
{
    std::stringstream fileOutput;
    std::cout<<"OMP run: "<<std::endl;
    double start = omp_get_wtime();
    for (int i=0;i<count;i++)
    {
//        bodies = c.read_file("input_"+ss.str()+".txt");
        for (int iteration = 0; iteration < maxIteration; ++iteration)
        {
            if(iteration%500 == 0){
                cout<<"iteration: "<<iteration<<"\n";
            }
            ComputeForcesPar(bodies, gTerm, deltaT, thread_count);
            MoveBodiesPar(bodies, deltaT);

            if(strcmp("n", output_files)){
                fileOutput.str(string());
                fileOutput << "nbody_" << iteration << ".txt";
                PersistPositions(fileOutput.str(), bodies);
            }
        }
    }
    double end = omp_get_wtime();
    printf("Average time taken for OMP: %f secs\n", ((end-start)/count));
}

void MPIRun(int argc,char **argv,int rank,int size,int particle_count,
           float gTerm,float deltaT,std::vector<Particle> bodies,
           int maxIteration,Common c,char* output_files,int count,int chunk,
           MPI_Datatype MPI_ParticleType){

    std::stringstream fileOutput;

    unsigned bodiesSize;
    if(rank == ROOT)
    {
        //write info to say that we are in MPI run
        std::cout<<"MPI run: "<<std::endl;
        //get size of the bodies array
        bodiesSize = bodies.size();
        /*
         * Send out the correct size
         */
        MPI_Bcast(&bodiesSize, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        /*
         * Size the bodies vector, and broadcast its contents
         */
        MPI_Bcast(&bodies.front(), bodiesSize, MPI_ParticleType, 0, MPI_COMM_WORLD);
    }else{
        MPI_Bcast(&bodiesSize, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        MPI_Bcast(&bodies.front(), bodiesSize, MPI_ParticleType, 0, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    for (int i=0;i<count;i++)
    {
        for (int iteration = 0; iteration < maxIteration; ++iteration)
        {
            if(iteration%500 == 0 && rank == ROOT){
                cout<<"iteration: "<<iteration<<"\n";
            }
            ComputeForcesMPI(bodies,(rank*chunk),((rank+1)*chunk),gTerm,deltaT);
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Allgather(MPI_IN_PLACE, chunk,
                            MPI_ParticleType, &bodies.front(), chunk,
                            MPI_ParticleType, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
            MoveBodiesMPI(bodies,deltaT);
            if(strcmp("n", output_files) && rank==0){
                fileOutput.str(string());
                fileOutput << "nbody_" << iteration << ".txt";
                PersistPositions(fileOutput.str(), bodies);
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();
    if(rank == ROOT)
        printf("Average time taken for MPI: %f secs.\n", ((end-start)/count));
}

void HybridRun(int argc,char **argv,int rank,int size,int particle_count,
               int thread_count,float gTerm,float deltaT,std::vector<Particle> bodies,
               int maxIteration,Common c,char* output_files,int count, int chunk,
               MPI_Datatype MPI_ParticleType)
{
    std::stringstream fileOutput;
    unsigned bodiesSize;
    if(rank == 0)
    {
        std::cout<<"Hybrid run: "<<std::endl;
        bodiesSize = bodies.size();
        /*
         * Send out the correct size
         */
        MPI_Bcast(&bodiesSize, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        /*
         * Size the bodies vector, and broadcast its contents
         */
        MPI_Bcast(&bodies.front(), bodiesSize, MPI_ParticleType, 0, MPI_COMM_WORLD);
    }else{
        MPI_Bcast(&bodiesSize, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        MPI_Bcast(&bodies.front(), bodiesSize, MPI_ParticleType, 0, MPI_COMM_WORLD);
    }
    double start = MPI_Wtime();
    for (int i=0;i<count;i++)
    {
        for (int iteration = 0; iteration < maxIteration; ++iteration)
        {
            if(iteration%500 == 0){
                cout<<"iteration: "<<iteration<<"\n";
            }
            ComputeForcesHybrid(bodies,(rank*chunk),((rank+1)*chunk),gTerm,deltaT,thread_count);
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Allgather(MPI_IN_PLACE, chunk,
                          MPI_ParticleType, &bodies.front(), chunk,
                          MPI_ParticleType, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
            MoveBodiesHybrid(bodies,deltaT);
            if(strcmp("n", output_files)){
                fileOutput.str(string());
                fileOutput << "nbody_" << iteration << ".txt";
                PersistPositions(fileOutput.str(), bodies);
            }
        }
    }
    double end = MPI_Wtime();

    if(rank == ROOT)
        printf("Average time taken for Hybrid: %f secs.\n", ((end-start)/count));
}
int main(int argc, char **argv)
{
    //declarations needed
    Common c;
    const int maxIteration = 1000;
    const float deltaT = 1.5f;
    const float gTerm = 20.f;
    int count = 1;
    stringstream fileOutput;

    //Display help menu
    if(c.display_help_menu(argc, argv)) exit(0);

    //Get required vars
    int particle_count = c.read_int(argc, argv, "-n", 10);
    int thread_count = c.read_int(argc, argv, "-t", 12);
    char* output_files = c.read_string(argc, argv, "-o", (char*)"n");
    int runs = c.read_int(argc, argv, "-r", 4);

    std::ostringstream ss;
    ss<<particle_count;

    //This is the vector to be filled, and its size
    std::vector<Particle> bodies = c.read_file("input_"+ss.str()+".txt");

    //MPI
    int size, rank;
    //Check if MPI could be initialised
    if(MPI_Init(&argc, &argv) != MPI_SUCCESS)
    {
        printf("MPI couldn't be initialized");
        MPI_Finalize();
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int chunk = particle_count/size;

    /*
     * Create an MPI struct for the Particle
     */
    const int    nItems=3;
    int          blocklengths[nItems] = {1, 1, 1};
    MPI_Datatype types[nItems] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
    MPI_Datatype MPI_ParticleType;
    MPI_Aint     offsets[nItems];
    offsets[0] = offsetof(Particle, Position);
    offsets[1] = offsetof(Particle, Velocity);
    offsets[2] = offsetof(Particle, Mass);
    MPI_Type_create_struct(nItems, blocklengths, offsets, types, &MPI_ParticleType);
    MPI_Type_commit(&MPI_ParticleType);


    switch (runs){
        case 0:
            SequentialRun(bodies,maxIteration,gTerm,deltaT,output_files,count);
            break;
        case 1:
            OmpRun(bodies,maxIteration,gTerm,deltaT,thread_count,output_files,count);
            break;
        case 2:
            MPIRun(argc,argv,rank,size,particle_count,gTerm,
                   deltaT,bodies,maxIteration,c,output_files,
                   count,chunk,MPI_ParticleType);
            break;
        case 3:
            HybridRun(argc,argv,rank,size,particle_count,thread_count,
                      gTerm,deltaT,bodies,maxIteration,c,output_files,
                      count,chunk,MPI_ParticleType);
            break;
        case 4:
            SequentialRun(bodies,maxIteration,gTerm,
                          deltaT,output_files,count);
            bodies = c.read_file("input_"+ss.str()+".txt");
            OmpRun(bodies,maxIteration,gTerm,deltaT,
                   thread_count,output_files,count);
            bodies = c.read_file("input_"+ss.str()+".txt");
            MPIRun(argc,argv,rank,size,particle_count,gTerm,
                   deltaT,bodies,maxIteration,c,output_files,count,chunk,MPI_ParticleType);
            bodies = c.read_file("input_"+ss.str()+".txt");
            HybridRun(argc,argv,rank,size,particle_count,thread_count,
                      gTerm,deltaT,bodies,maxIteration,c,output_files,count,chunk,MPI_ParticleType);
            break;
        default:
            break;
    }

    MPI_Type_free(&MPI_ParticleType);
    MPI_Finalize();
    return 0;
}