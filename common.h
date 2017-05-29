#pragma once
#include <string.h>
#include <stdlib.h>
using namespace std;


/*
 * Constant definitions for field dimensions, and particle masses
 */
const int fieldWidth = 1000;
const int fieldHalfWidth = fieldWidth >> 1;
const int fieldHeight = 1000;
const int fieldHalfHeight = fieldHeight >> 1;

const float minBodyMass = 2.5f;
const float maxBodyMassVariance = 5.f;

/*
 * Particle structure
 */
struct Particle
{
    Vector2 Position;
    Vector2 Velocity;
    double	Mass;

    Particle(){}
    Particle(double Mass, double xPos, double yPos)
            : Position(xPos, yPos)
            , Velocity( 0, 0)
            , Mass(Mass)
    { }
};

class Common
{

public:
    int find_option(int argc, char **argv, const char *option){
        for( int i = 1; i < argc; i++ )
            if( strcmp( argv[i], option ) == 0 )
                return i;
        return -1;
    }
    int read_int( int argc, char **argv, const char *option, int default_value ){
        int iplace = find_option( argc, argv, option );
        if( iplace >= 0 && iplace < argc-1 )
            return atoi( argv[iplace+1] );
        return default_value;
    }
    char* read_string(int argc, char** argv, const char *option, char *default_value){
        int iplace = find_option(argc, argv, option);
        if( iplace >= 0 && iplace < argc-1)
            return argv[iplace+1];
        return default_value;
    }
    template<typename Out>
    void split(const std::string &s, char delim, Out result){
        std::stringstream ss;
        ss.str(s);
        std::string item;
        while (std::getline(ss, item, delim)) {
            *(result++) = item;
        }
    }
    std::vector<std::string> split(const std::string &s, char delim){
        std::vector<std::string> elems;
        split(s, delim, std::back_inserter(elems));
        return elems;
    }

    vector<Particle> read_file(const std::string &file_name){
        string line;
        ifstream file_(file_name.c_str());
        vector<Particle> bodies;
        if(file_.is_open())
        {
            while(getline(file_, line))
            {
                double scale = 0.000001;
                vector<string> particle = split(line, ',');
                double mass = floor(::atof(particle[0].c_str()) / scale + 0.5) * scale;
                double xPos = floor(::atof(particle[1].c_str()) / scale + 0.5) * scale;
                double yPos = floor(::atof(particle[2].c_str()) / scale + 0.5) * scale;
                Particle body(mass, xPos, yPos);
                bodies.push_back(body);
            }
            file_.close();
        }
        return bodies;
    }
    int display_help_menu(int argc, char **argv){
        if( find_option( argc, argv, "-h" ) >= 0 )
        {
            printf( "Options:\n" );
            printf( "-h to see this help\n" );
            printf( "-n <int> to set the number of particles(64, 1024, 4096, 16384)\n" );
            printf( "-t <int> to specify the thread number\n" );
            printf( "-o <char> to specify whether output logs are required(y/n)\n" );
            printf( "-r <char> to specify which version to run: 0(seq), 1(omp), 2(mpi), 3(hybrid), 4(all)\n" );
            return 1;
        }
    }
};
