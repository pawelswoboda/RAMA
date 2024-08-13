#pragma once
#include <CLI/CLI.hpp>

struct multicut_solver_options {
    std::string input_file;
    std::string output_sol_file="";
    int max_cycle_length_lb = 5;
    int num_dual_itr_lb = 10;
    int max_cycle_length_primal = 3;
    int num_dual_itr_primal = 5;
    int num_outer_itr_dual = 1;
    float mean_multiplier_mm = 0.05;
    float matching_thresh_crossover_ratio = 0.05;
    float tri_memory_factor = 2.0;
    bool only_compute_lb = false;
    int max_time_sec = -1;
    bool dump_timeline = false;
    bool verbose = true;
    bool sanitize_graph = false;
    // =========== Preprocessor ============//
    bool run_preprocessor = false;
    bool preprocessor_each_step = false;
    float preprocessor_threshold = 0.005;


    multicut_solver_options() { }
    multicut_solver_options(const std::string& solver_type) {
        if(solver_type == "PD")
        {
            std::cout<<"Running solver type PD which offers best compute time versus quality tradeoff."<<std::endl;
        }
        else if(solver_type == "P")
        {
            std::cout<<"Running purely primal solver (better runtime, worse quality)"<<std::endl;
            max_cycle_length_lb = 0;
            num_dual_itr_lb = 0;
            max_cycle_length_primal = 0;
            num_dual_itr_primal = 0;
        }
        else if(solver_type == "PD+")
        {
            std::cout<<"Running PD+ solver (worse runtime, better quality)"<<std::endl;
            max_cycle_length_lb = 5;
            num_dual_itr_lb = 10;
            max_cycle_length_primal = 5;
            num_dual_itr_primal = 10;
        }
        else if(solver_type == "D")
        {
            std::cout<<"Running dual solver to compute only the lower bound."<<std::endl;
            max_cycle_length_lb = 5;
            num_dual_itr_lb = 10;
            num_outer_itr_dual = 5;
            only_compute_lb = true;
        }
        else
            std::runtime_error("invalid solver_type specified.");
    }

    multicut_solver_options(
        const int _max_cycle_length_lb, 
        const int _num_dual_itr_lb, 
        const int _max_cycle_length_primal, 
        const int _num_dual_itr_primal, 
        const int _num_outer_itr_dual,
        const float _mean_multiplier_mm,
        const float _matching_thresh_crossover_ratio,
        const float _tri_memory_factor,
        const float _preprocessor_threshold,
        const bool _only_compute_lb,
        const int _max_time_sec, 
        const bool _dump_timeline = false,
        const bool _sanitize_graph = false,
        const bool _run_preprocessor = false,
        const bool _preprocessor_each_step = false) :
        max_cycle_length_lb(_max_cycle_length_lb), 
        num_dual_itr_lb(_num_dual_itr_lb), 
        max_cycle_length_primal(_max_cycle_length_primal), 
        num_dual_itr_primal(_num_dual_itr_primal), 
        num_outer_itr_dual(_num_outer_itr_dual),
        mean_multiplier_mm(_mean_multiplier_mm),
        matching_thresh_crossover_ratio(_matching_thresh_crossover_ratio),
        tri_memory_factor(_tri_memory_factor),
        preprocessor_threshold(_preprocessor_threshold),
        only_compute_lb(_only_compute_lb),
        max_time_sec(_max_time_sec),
        dump_timeline(_dump_timeline),
        sanitize_graph(_sanitize_graph),
        run_preprocessor(_run_preprocessor),
        preprocessor_each_step(_preprocessor_each_step)
    {}

    int from_cl(int argc, char** argv) {
        CLI::App app{"Solver for multicut problem. "};
        app.add_option("-f,--file,file_pos", input_file, "Path to multicut instance (.txt)")->required()->check(CLI::ExistingPath);
        app.add_option("-o,--out_sol_file", output_sol_file, "Path to save node labeling (.txt)");
        app.add_option("max_cycle_dual", max_cycle_length_lb, "Maximum length of conflicted cycles to consider for initial dual updates. (Default: 5).")->check(CLI::Range(0, 5));
        app.add_option("dual_itr", num_dual_itr_lb, "Number of dual update iterations per cycle. (Default: 10).")->check(CLI::NonNegativeNumber);
        app.add_option("max_cycle_primal", max_cycle_length_primal, "Maximum length of conflicted cycles to consider during primal iterations for reparameterization. (Default: 3).")->check(CLI::Range(0, 5));
        app.add_option("dual_itr_primal", num_dual_itr_primal, "Number of dual update iterations per cycle during primal reparametrization. (Default: 5).")->check(CLI::NonNegativeNumber);
        app.add_option("dual_itr_outer", num_outer_itr_dual, "Number of outer dual iterations for initial dual updates. Larger number detects conflicted cycles again. (Default: 1).")->check(CLI::NonNegativeNumber);
        app.add_option("mean_multiplier_mm", mean_multiplier_mm, "Match the edges which have cost more than mean(pos edges) * mean_multiplier_mm.")->check(CLI::NonNegativeNumber);
        app.add_option("matching_thresh_crossover_ratio", matching_thresh_crossover_ratio, "Ratio of (# contract edges / # nodes ) at which to change from maximum matching based contraction to MST based. "
            "(Default: 0.1). Greater than 1 will always use MST.")->check(CLI::NonNegativeNumber);
        app.add_option("tri_memory_factor", tri_memory_factor, 
            "Average number of triangles per repulsive edge. (Used for memory allocation. Use lesser value in-case of out of memory errors during dual solve). (Default: 2.0).")->check(CLI::PositiveNumber);
        app.add_option("--preprocessor_threshold", preprocessor_threshold, "Minimum fraction of edges found by the preprocessor to warrant the contraction of the graph. (Default: 0.005).") -> check(CLI::Range(0.0,1.0));
        app.add_flag("--only_lb", only_compute_lb, "Only compute the lower bound. (Default: false).");
        app.add_flag("--dump_timeline", dump_timeline, "Return the output of each contraction step. Only use for debugging/visualization purposes. (slow). (Default: false).");
        app.add_flag("--sanitize_graph", sanitize_graph, "If the input graph contains nodes without any edges and thus needs sanitizing. Cluster labels in this case will be -1 for these nodes. (Default: false).");
        app.add_flag("--run_preprocessor", run_preprocessor, "Run a preprocessor, searching for persistency criteria to reduce the effective size of the graph. (Default: false).");
        app.add_flag("--preprocessor_each_step", preprocessor_each_step, "If true, the preprocessor will run in between each cycle of the RAMA algorithm. (slow). (Default: false).");
        try {
            app.parse(argc, argv);
            return -1;
        } catch (const CLI::ParseError &e) {
            return app.exit(e);
        }
    }

    std::string get_string() const
    {
        return std::string("<multicut_solver_options>:") +
            "max_cycle_length_lb: " + std::to_string(max_cycle_length_lb) +
            ", num_dual_itr_lb: " + std::to_string(num_dual_itr_lb) +
            ", num_dual_itr_primal: " + std::to_string(num_dual_itr_primal) +
            ", max_cycle_length_primal: " + std::to_string(max_cycle_length_primal) +
            ", num_outer_itr_dual: " + std::to_string(num_outer_itr_dual) +
            ", mean_multiplier_mm: " + std::to_string(mean_multiplier_mm) +
            ", matching_thresh_crossover_ratio: " + std::to_string(matching_thresh_crossover_ratio) +
            ", tri_memory_factor: " + std::to_string(tri_memory_factor) +
            ", only_compute_lb: " + std::to_string(only_compute_lb) +
            ", max_time_sec: " + std::to_string(max_time_sec) +
            ", sanitize_graph: " + std::to_string(sanitize_graph) +
            ", verbose: " + std::to_string(verbose) +
            ", run_preprocessor: " + std::to_string(run_preprocessor) +
            ", preprocessor_each_step: " + std::to_string(preprocessor_each_step) +
            ", preprocessor_threshold: " + std::to_string(preprocessor_threshold) + "\n";
    }

    void print() const
    {
        std::cout<<this->get_string();
    }
};