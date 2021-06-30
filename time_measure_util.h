#pragma once

#include <chrono>
#include <ratio>
#include <string>
#include <iostream>
#include <tuple>
#include <utility>


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

class time_elapse_aggregator
{
    public:
        std::chrono::duration<size_t,std::ratio<1,1000000000>> duration; 
        std::string function_name;

        time_elapse_aggregator(const std::string caller)
            : function_name(caller)
        {}

        ~time_elapse_aggregator()
        {
            std::cout << "cumulative execution time for ";
            std::cout << function_name;
            std::cout << " is "<< std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() << " ms\n"; 
        }
};

class measure_cumulative_execution_time
{
    private:
        time_elapse_aggregator& time_elapsed;
        const std::chrono::steady_clock::time_point begin;

    public:
        measure_cumulative_execution_time(time_elapse_aggregator& _time_elapsed)
            : time_elapsed(_time_elapsed),
            begin(std::chrono::steady_clock::now())
        {}

        ~measure_cumulative_execution_time()
        {
            const auto duration = std::chrono::steady_clock::now()-begin;
            time_elapsed.duration += duration; 
        }
};

#ifndef MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
#define MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME static time_elapse_aggregator time_elapse_aggregator_object(__func__); measure_cumulative_execution_time measure_cumulative_execution_time_object(time_elapse_aggregator_object);
#endif

#ifndef MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME2
#define MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME2(TIME_ELAPSED_IDENTIFIER) static time_elapse_aggregator time_elapse_aggregator_object(TIME_ELAPSED_IDENTIFIER); measure_cumulative_execution_time measure_cumulative_execution_time_object(time_elapse_aggregator_object);
#endif

