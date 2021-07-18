#include <iostream>
#include <fstream>
#include <thread>
#include <queue>
#include <semaphore>
#include <vector>
#include <optional>
#include <chrono>
#include "image.hpp"
#include "monitor.hpp"
#include "argparse.hpp"

template < class T >
std::optional<T> front_and_pop( Monitor<std::queue<T>> & mon ) {
    auto lock = mon.safe();
    if ( lock->empty() ){
        return std::nullopt;
    }
    else {
        auto value = lock->front();
        lock->pop();
        return value;
    }
}

long long execute( int threads, std::string source,std::optional<std::string> output);

int main ( int argc, char **argv ) {
    auto program = argparse::ArgumentParser( argv[0]);
    int threads;
    std::string source;
    std::optional<std::string> output;
    program.add_argument("image")
        .help("source image");
    program.add_argument("-j", "--threads")
        .help("number of threads")
        .action([](const std::string& value) { return std::stoi(value); })
        .default_value(1);
    program.add_argument("-o", "--output")
        .help("validation output")
        .action([]( const std::string& value) { return std::optional(value); })
        .default_value(std::optional<std::string>(std::nullopt));
    try {
        program.parse_args( argc, argv );

    }
    catch (const std::runtime_error& err) {
        std::cout << err.what() << std::endl;
        std::cout << program;
        exit(0);
    }
    

    try {
        threads = program.get<int>("--threads");
        source = program.get<std::string>("image");
        output = program.get<std::optional<std::string>>("--output");
    }
    catch( const std::logic_error& err ) {
        std::cout << err.what() << std::endl;
        std::cout << program;
        exit(0);
    }


    auto time = execute( threads, source, output );
    std::cout << "Took " << time << " µs\n";
    return 0;
}

long long execute( int NUM_THREADS, std::string source, std::optional<std::string> output ) {

    auto rowsMonitor = Monitor(std::queue<std::span<int>>());
    auto outputMonitor = Monitor(std::vector<std::vector<int>>());

    auto threads = std::vector<std::thread>();

    auto image = Image(source);
        auto start = std::chrono::high_resolution_clock::now();

    rowsMonitor.unsafeLock();

    for ( int i = 0, l = image.getHeight(); i < l; ++i  )
        rowsMonitor.unsafe().push( image[i] );
    
    for ( int  i = 0 ; i < NUM_THREADS; ++i ) {
        // logica thread
        auto thread = std::thread([&rowsMonitor, &outputMonitor, i]() {
            int count = 0;
            std::optional<std::span<int>> _input;
            
            while ( _input = front_and_pop(rowsMonitor)) {
                count ++;
                auto input = std::vector<int>( );
                for ( auto i : _input.value() ) {
                    input.push_back( (i & 0xFF) > 128 );
                }

                auto runs = std::vector<int>( ); {
                    int run = 1;
                    for ( int i = 1, size = input.size() ; i < size; ++i ) {
                        while ( i < size-1 && input[i] == input[i+1]  ) {
                            run++;
                            i++;
                        }
                        runs.push_back(run);
                        run = 1;
                    }
                }
                // normalizza
                {
                    int rif, min = INT_MAX, max = INT_MIN; for (int i = 1, size = runs.size() -1; i < size; ++i ) {
                        min = std::min( runs[i], min);
                        max = std::max( runs[i], max);
                    }
                    rif = max - min;

                    for ( int i = 0, size = runs.size(); i < size; ++i ) {
                        if ( i == 0 || i == size - 1) {
                            runs[i] = 1;
                        }
                        else {
                            auto el = runs[i] - min ;
                            el = 
                                el < 1 + rif / 4
                                    ? 1
                                    : el < 1 + rif / 2
                                        ? 2
                                        : el < 1 + rif / 4 + rif / 2
                                            ? 3
                                            : el <= 1 + rif
                                                ? 4
                                                : 1; 
                            //std::cout << i << ':' << runs[i] << ':' << el << ' ';

                            runs[i] = el;
                        }
                    }
                }
                auto output = std::vector<int>(  );
                
                for ( int i = 0, size = runs.size(); i < size; ++i ) {
                    int run = runs[i];
                    int value = i & 1;
                    for ( int j = 0; j < run; ++j )
                        output.push_back(value);
                }
                //std::cout << '\n';
                outputMonitor->push_back( std::move(output));


            }
        });

        threads.push_back( std::move( thread )); 
    }

    rowsMonitor.unsafeRelease();
    for ( auto &t : threads ) t.join();
	auto end = std::chrono::high_resolution_clock::now();
    
    long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    if ( output ) {
        std::string path = output.value();
        std::ofstream file(path);
        auto data = outputMonitor.unsafe();
        for ( auto &row : data ) {
            for ( int i = 0, size = row.size(); i < size; ++i ) {
                file << row[i];
                if ( i < size - 1 ) {
                    file << ' ';
                }
            }
            file << "\n";
        }
        
        file.close();
    }

    return microseconds; 

}