#pragma once

#include <chrono>
#include <string>
#include <iostream>

class MeasureExecutionTime
{
    private:
        const std::chrono::steady_clock::time_point begin;
        const std::string caller;
    public:
        MeasureExecutionTime(const std::string& caller):caller(caller),begin(std::chrono::steady_clock::now()){}
        ~MeasureExecutionTime(){
            const auto duration=std::chrono::steady_clock::now()-begin;
            std::cout << "execution time for " << caller << " is "<<std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()<<" ms\n";
        }
};

#ifndef MEASURE_FUNCTION_EXECUTION_TIME
#define MEASURE_FUNCTION_EXECUTION_TIME const MeasureExecutionTime measureExecutionTime(__FUNCTION__);
#endif
