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
    float triangle_budget_ratio = 0;
    bool only_compute_lb = false;
    int max_time_sec = -1;
    bool dump_timeline = false;
    bool verbose = false;
    bool sanitize_graph = false;
    std::string long_cycle_method = "bfs";

    multicut_solver_options() { }
    multicut_solver_options(const std::string& solver_type) {
        apply_solver_type(solver_type);
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
        const bool _only_compute_lb,
        const int _max_time_sec,
        const bool _dump_timeline = false,
        const bool _sanitize_graph = false) :
        max_cycle_length_lb(_max_cycle_length_lb),
        num_dual_itr_lb(_num_dual_itr_lb),
        max_cycle_length_primal(_max_cycle_length_primal),
        num_dual_itr_primal(_num_dual_itr_primal),
        num_outer_itr_dual(_num_outer_itr_dual),
        mean_multiplier_mm(_mean_multiplier_mm),
        matching_thresh_crossover_ratio(_matching_thresh_crossover_ratio),
        tri_memory_factor(_tri_memory_factor),
        only_compute_lb(_only_compute_lb),
        max_time_sec(_max_time_sec),
        dump_timeline(_dump_timeline),
        sanitize_graph(_sanitize_graph)
    {}

    int from_cl(int argc, char** argv) {
        CLI::App app{
            "RAMA: Rapid Algorithm for Multicut Problem\n"
            "\n"
            "Solves the multicut (correlation clustering) problem on a graph.\n"
            "Edges with positive costs prefer endpoints in the same cluster;\n"
            "edges with negative costs prefer endpoints in different clusters.\n"
            "\n"
            "Solver types (--solver-type):\n"
            "  PD   Primal-dual (default). Best time/quality tradeoff.\n"
            "  P    Purely primal. Fastest, lower quality.\n"
            "  PD+  Extended primal-dual. Best quality, slower.\n"
            "  D    Dual only. Computes lower bound, no clustering."
        };

        // --- Input / Output ---
        app.add_option("-i,--input", input_file,
            "Path to input graph (.txt in MULTICUT format)")
            ->required()->check(CLI::ExistingPath)
            ->group("Input/Output");
        app.add_option("-o,--output", output_sol_file,
            "Path to save output node labeling (.txt)")
            ->group("Input/Output");

        // --- Solver type preset ---
        std::string solver_type;
        app.add_option("-s,--solver-type", solver_type,
            "Solver preset: P, PD (default), PD+, or D")
            ->check(CLI::IsMember({"P", "PD", "PD+", "D"}))
            ->group("Solver");

        // --- Dual solver options ---
        app.add_option("--max-cycle-length-lb", max_cycle_length_lb,
            "Max cycle length for initial lower-bound computation [0-100]")
            ->check(CLI::Range(0, 100))
            ->group("Dual solver");
        app.add_option("--num-dual-itr-lb", num_dual_itr_lb,
            "Number of message-passing iterations for lower bound")
            ->check(CLI::NonNegativeNumber)
            ->group("Dual solver");
        app.add_option("--num-outer-itr-dual", num_outer_itr_dual,
            "Outer dual iterations (re-detects conflicted cycles)")
            ->check(CLI::NonNegativeNumber)
            ->group("Dual solver");
        app.add_option("--long-cycle-method", long_cycle_method,
            "Method for cycles longer than 5: \"bfs\" or \"mst\"")
            ->check(CLI::IsMember({"bfs", "mst"}))
            ->group("Dual solver");

        // --- Primal solver options ---
        app.add_option("--max-cycle-length-primal", max_cycle_length_primal,
            "Max cycle length for primal reparametrization [0-100]")
            ->check(CLI::Range(0, 100))
            ->group("Primal solver");
        app.add_option("--num-dual-itr-primal", num_dual_itr_primal,
            "Number of message-passing iterations during primal phase")
            ->check(CLI::NonNegativeNumber)
            ->group("Primal solver");
        app.add_option("--mean-multiplier-mm", mean_multiplier_mm,
            "Matching threshold as fraction of mean positive-edge cost")
            ->check(CLI::NonNegativeNumber)
            ->group("Primal solver");
        app.add_option("--matching-thresh-crossover", matching_thresh_crossover_ratio,
            "Contract/node ratio at which to switch from matching to MST contraction (>1 = always MST)")
            ->check(CLI::NonNegativeNumber)
            ->group("Primal solver");

        // --- Solver flags ---
        app.add_flag("--only-lb", only_compute_lb,
            "Only compute the lower bound, skip primal solve")
            ->group("Solver");
        app.add_option("--max-time", max_time_sec,
            "Time limit in seconds (-1 = unlimited)")
            ->group("Solver");
        app.add_flag("--sanitize-graph", sanitize_graph,
            "Handle graphs with isolated nodes (labels will be -1 for those)")
            ->group("Solver");

        // --- Advanced ---
        app.add_option("--tri-memory-factor", tri_memory_factor,
            "Expected triangles per repulsive edge (for memory pre-allocation)")
            ->check(CLI::PositiveNumber)
            ->group("Advanced");
        app.add_option("--triangle-budget-ratio", triangle_budget_ratio,
            "Keep only top fraction of triangles by cycle strength (0 = no limit)")
            ->check(CLI::NonNegativeNumber)
            ->group("Dual solver");

        // --- Debug ---
        app.add_flag("-v,--verbose", verbose,
            "Print detailed solver progress")
            ->group("Debug");
        app.add_flag("--dump-timeline", dump_timeline,
            "Record clustering at each contraction step (slow)")
            ->group("Debug");

        try {
            app.parse(argc, argv);
        } catch (const CLI::ParseError &e) {
            return app.exit(e);
        }

        // Apply solver-type preset first, then let explicit flags override
        if (!solver_type.empty())
            apply_solver_type(solver_type);

        return -1;
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
            ", triangle_budget_ratio: " + std::to_string(triangle_budget_ratio) +
            ", only_compute_lb: " + std::to_string(only_compute_lb) +
            ", max_time_sec: " + std::to_string(max_time_sec) +
            ", sanitize_graph: " + std::to_string(sanitize_graph) +
            ", long_cycle_method: " + long_cycle_method +
            ", verbose: " + std::to_string(verbose) + "\n";
    }

    void print() const
    {
        std::cout<<this->get_string();
    }

private:
    void apply_solver_type(const std::string& solver_type) {
        if (solver_type == "PD")
        {
            // defaults are already PD
        }
        else if (solver_type == "P")
        {
            max_cycle_length_lb = 0;
            num_dual_itr_lb = 0;
            max_cycle_length_primal = 0;
            num_dual_itr_primal = 0;
        }
        else if (solver_type == "PD+")
        {
            max_cycle_length_lb = 5;
            num_dual_itr_lb = 10;
            max_cycle_length_primal = 5;
            num_dual_itr_primal = 10;
        }
        else if (solver_type == "D")
        {
            max_cycle_length_lb = 5;
            num_dual_itr_lb = 10;
            num_outer_itr_dual = 5;
            only_compute_lb = true;
        }
        else
        {
            throw std::runtime_error("Invalid solver type: " + solver_type +
                ". Must be one of: P, PD, PD+, D");
        }
    }
};