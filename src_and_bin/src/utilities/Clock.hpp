#ifndef _CLOCK_HPP
#define _CLOCK_HPP

#include <iostream>
#include <iomanip>
#include <chrono>

class Clock {
protected:
    std::chrono::high_resolution_clock::time_point startTime;
public:
    Clock() {
        tick();
    }

    void tick() {
        startTime = std::chrono::high_resolution_clock::now();
    }

    double tock() {
        std::chrono::high_resolution_clock::time_point endTime = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::duration<double>>(endTime - startTime).count();
    }

    // Print elapsed time with newline
    void ptock() {
        double elapsed = tock();
        std::cout << "Took " << elapsed << " seconds" << std::endl;
    }
};

#endif
