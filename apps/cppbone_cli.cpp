// cppbone_cli: command-line front end to the library. Loads an edge list, runs any set of backbone methods,
// filters them, and optionally compares the results (measures, property progression, KS statistics, consensus).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include <io.h>
#define CPPBONE_ISATTY(fd) _isatty(fd)
#else
#include <unistd.h>
#define CPPBONE_ISATTY(fd) isatty(fd)
#endif

#include <args.hxx>
#include <indicators/progress_bar.hpp>

#include "cppbone/cppbone.hpp"

namespace cb = cppbone;
namespace fs = std::filesystem;

namespace {

// ---- registries ---------------------------------------------------------------------------------------------

struct MethodOptions {
    std::uint32_t ecm_seed = 0;
    std::uint32_t modularity_seed = 123;
    double glanb_c = 1;
    bool betweenness_unweighted = false;
    bool betweenness_raw = false;
    bool degree_weighted = false;
    bool noise_corrected_exact = false;
};

struct Method {
    std::string name;
    std::string family;
    std::function<cb::Backbone(const cb::Graph&, const MethodOptions&)> run;
};

const std::vector<Method>& methods() {
    using G = const cb::Graph&;
    using O = const MethodOptions&;
    static const std::vector<Method> m = {
        {"disparity", "statistical", [](G g, O) { return cb::disparity(g); }},
        {"lans", "statistical", [](G g, O) { return cb::lans(g); }},
        {"noise_corrected", "statistical", [](G g, O o) { return cb::noise_corrected(g, !o.noise_corrected_exact); }},
        {"marginal_likelihood", "statistical", [](G g, O) { return cb::marginal_likelihood(g); }},
        {"ecm", "statistical", [](G g, O o) { return cb::ecm(g, o.ecm_seed); }},
        {"betweenness", "structural",
         [](G g, O o) { return cb::betweenness(g, !o.betweenness_unweighted, !o.betweenness_raw); }},
        {"degree", "structural", [](G g, O o) { return cb::degree(g, o.degree_weighted); }},
        {"doubly_stochastic", "structural", [](G g, O) { return cb::doubly_stochastic(g); }},
        {"global_threshold", "structural", [](G g, O) { return cb::global_threshold(g); }},
        {"gspar", "structural", [](G g, O) { return cb::gspar(g); }},
        {"h_backbone", "structural", [](G g, O) { return cb::h_backbone(g); }},
        {"high_salience_skeleton", "structural", [](G g, O) { return cb::high_salience_skeleton(g); }},
        {"maximum_spanning_tree", "structural", [](G g, O) { return cb::maximum_spanning_tree(g); }},
        {"metric_distance_backbone", "structural", [](G g, O) { return cb::metric_distance_backbone(g); }},
        {"ultrametric_distance_backbone", "structural", [](G g, O) { return cb::ultrametric_distance_backbone(g); }},
        {"mlam", "structural", [](G g, O) { return cb::mlam(g); }},
        {"plam", "structural", [](G g, O) { return cb::plam(g); }},
        {"modularity_backbone", "structural", [](G g, O o) { return cb::modularity_backbone(g, o.modularity_seed); }},
        {"pmfg", "structural", [](G g, O) { return cb::pmfg(g); }},
        {"glanb", "hybrid", [](G g, O o) { return cb::glanb(g, o.glanb_c); }},
    };
    return m;
}

const std::vector<std::pair<std::string, cb::Property>>& measures() {
    static const std::vector<std::pair<std::string, cb::Property>> m = {
        {"node_fraction", cb::node_fraction},
        {"edge_fraction", cb::edge_fraction},
        {"weight_fraction", cb::weight_fraction},
        {"density", cb::density},
        {"average_degree", cb::average_degree},
        {"weight_entropy", cb::weight_entropy},
        {"reachability", cb::reachability},
        {"number_connected_components", cb::number_connected_components},
        {"lcc_size", cb::lcc_size},
        {"lcc_node_fraction", [](const cb::Graph&, const cb::Graph& g) { return cb::lcc_node_fraction(g); }},
        {"lcc_edge_fraction", cb::lcc_edge_fraction},
        {"lcc_weight_fraction", cb::lcc_weight_fraction},
        {"diameter", cb::diameter},
        {"average_clustering_coefficient", cb::average_clustering_coefficient},
        {"transitivity", cb::transitivity},
    };
    return m;
}

const std::vector<std::pair<std::string, cb::Distribution>>& distributions() {
    static const std::vector<std::pair<std::string, cb::Distribution>> d = {
        {"degree", [](const cb::Graph& g) { return cb::degrees(g); }},
        {"weighted_degree", [](const cb::Graph& g) { return cb::degrees(g, "weight"); }},
        {"weight", [](const cb::Graph& g) { return cb::weights(g); }},
    };
    return d;
}

// unordered_map: the map type args::MapFlag expects.
const std::unordered_map<std::string, cb::Filter> filters = {
    {"boolean", cb::boolean_filter},
    {"threshold", cb::threshold_filter},
    {"fraction", cb::fraction_filter},
};

// ---- helpers ------------------------------------------------------------------------------------------------

struct UsageError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Flattens repeated and comma-separated values ("-m a,b -m c").
std::vector<std::string> split_list(const std::vector<std::string>& raw) {
    std::vector<std::string> out;
    for (const auto& item : raw) {
        std::stringstream ss(item);
        std::string tok;
        while (std::getline(ss, tok, ','))
            if (!tok.empty()) out.push_back(tok);
    }
    return out;
}

const std::string& name_of(const Method& m) { return m.name; }
template <typename T>
const std::string& name_of(const std::pair<std::string, T>& p) { return p.first; }

// Resolves names against a registry; "all" selects every entry. Keeps registry order, drops duplicates.
template <typename Registry>
std::vector<size_t> select(const Registry& registry, const std::vector<std::string>& names, const std::string& what) {
    std::vector<bool> chosen(registry.size(), false);
    for (const auto& n : names) {
        if (n == "all") {
            chosen.assign(registry.size(), true);
            continue;
        }
        auto it = std::find_if(registry.begin(), registry.end(), [&](const auto& e) { return name_of(e) == n; });
        if (it == registry.end()) throw UsageError("unknown " + what + " '" + n + "' (see --list)");
        chosen[it - registry.begin()] = true;
    }
    std::vector<size_t> idx;
    for (size_t i = 0; i < chosen.size(); ++i)
        if (chosen[i]) idx.push_back(i);
    return idx;
}

std::vector<double> parse_doubles(const std::string& s) {
    std::vector<double> out;
    for (const auto& tok : split_list({s})) {
        size_t pos = 0;
        double v = 0;
        try {
            v = std::stod(tok, &pos);
        } catch (const std::exception&) {
            pos = 0;
        }
        if (pos != tok.size()) throw UsageError("not a number: '" + tok + "'");
        out.push_back(v);
    }
    return out;
}

bool accepts(const cb::Backbone& b, cb::Filter f) {
    const auto& fs = b.compatible_filters();
    return std::find(fs.begin(), fs.end(), f) != fs.end();
}

std::string compatible_list(const cb::Backbone& b) {
    std::string s;
    for (cb::Filter f : b.compatible_filters()) {
        std::string n = cb::filter_name(f);
        n = n.substr(0, n.find("_filter"));
        s += (s.empty() ? "" : ",") + n;
    }
    return s;
}

// Writes to `dir/name`, or to stdout under a "# name" banner when no output directory was given.
void emit(const std::optional<fs::path>& dir, const std::string& name, const std::function<void(std::ostream&)>& w) {
    if (!dir) {
        std::cout << "\n# " << name << '\n';
        w(std::cout);
        return;
    }
    fs::path p = *dir / name;
    std::ofstream out(p);
    if (!out) throw std::runtime_error("cannot write " + p.string());
    w(out);
}

// Progress over the selected methods on stderr; a no-op when disabled.
class Progress {
public:
    Progress(size_t total, bool enabled) : total_(total) {
        if (!enabled || total == 0) return;
        bar_ = std::make_unique<indicators::ProgressBar>(
            indicators::option::BarWidth{40}, indicators::option::Start{"["}, indicators::option::Fill{"="},
            indicators::option::Lead{">"}, indicators::option::Remainder{" "}, indicators::option::End{"]"},
            indicators::option::ShowElapsedTime{true}, indicators::option::ShowPercentage{true},
            indicators::option::MaxProgress{total}, indicators::option::Stream{std::cerr},
            indicators::option::ForegroundColor{indicators::Color::cyan});
    }
    void start(const std::string& what) {
        if (bar_) bar_->set_option(indicators::option::PostfixText{what + " (" + std::to_string(done_ + 1) + "/" +
                                                                   std::to_string(total_) + ")"});
        if (bar_) bar_->print_progress();
    }
    void tick() {
        ++done_;
        if (!bar_) return;
        if (done_ == total_) bar_->set_option(indicators::option::PostfixText{"done"});
        bar_->tick();
    }

private:
    size_t total_, done_ = 0;
    std::unique_ptr<indicators::ProgressBar> bar_;
};

void print_list() {
    std::cout << "methods:\n";
    for (const auto& m : methods()) std::cout << "  " << std::left << std::setw(32) << m.name << m.family << '\n';
    std::cout << "\nfilters:\n";
    for (const char* f : {"boolean", "threshold", "fraction"}) std::cout << "  " << f << '\n';
    std::cout << "\nmeasures (--measure):\n";
    for (const auto& m : measures()) std::cout << "  " << m.first << '\n';
    std::cout << "\ndistributions (--ks):\n";
    for (const auto& d : distributions()) std::cout << "  " << d.first << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    args::ArgumentParser parser(
        "Extract network backbones from a weighted edge list.",
        "Lists accept repeated flags and comma-separated values; 'all' selects every entry.\n"
        "Example: cppbone_cli edges.csv -m disparity,noise_corrected -f threshold -v 0.05 -M all -o out/");
    parser.helpParams.width = 100;
    parser.helpParams.flagindent = 2;
    args::HelpFlag help(parser, "help", "Show this help and exit", {'h', "help"});
    args::Flag list(parser, "list", "List methods, filters, measures and distributions, then exit", {"list"});
    args::CompletionFlag completion(parser, {"complete"});

    args::Group input_group(parser, "Input:");
    args::Positional<std::string> input(input_group, "input", "Edge list CSV");
    args::Flag no_header(input_group, "no-header",
                         "The CSV has no header; columns are picked by index (--source-col/--target-col/--weight-col)",
                         {'H', "no-header"});
    args::ValueFlag<int> source_col(input_group, "N", "Source column, 0-based (default 0, with --no-header)",
                                    {"source-col"}, 0);
    args::ValueFlag<int> target_col(input_group, "N", "Target column, 0-based (default 1, with --no-header)",
                                    {"target-col"}, 1);
    args::ValueFlag<int> weight_col(input_group, "N",
                                    "Weight column, 0-based (default 2, with --no-header); -1 for an unweighted list",
                                    {"weight-col"}, 2);
    args::ValueFlag<std::string> weight_name(input_group, "NAME",
                                             "Header column to use as the weight (default 'weight')", {"weight-name"});
    args::Flag use_lcc(input_group, "lcc", "Keep only the largest connected component", {"lcc"});

    args::Group method_group(parser, "Methods:");
    args::ValueFlagList<std::string> method_names(method_group, "METHOD", "Backbone method(s) to run (default all)",
                                                  {'m', "method"});
    args::ValueFlag<std::uint32_t> ecm_seed(method_group, "SEED", "ecm random seed (default 0)", {"ecm-seed"}, 0);
    args::ValueFlag<std::uint32_t> modularity_seed(method_group, "SEED", "modularity_backbone seed (default 123)",
                                                   {"modularity-seed"}, 123);
    args::ValueFlag<double> glanb_c(method_group, "C", "glanb c parameter (default 1)", {"glanb-c"}, 1);
    args::Flag betweenness_unweighted(method_group, "unweighted", "betweenness: ignore weights",
                                      {"betweenness-unweighted"});
    args::Flag betweenness_raw(method_group, "raw", "betweenness: do not normalize", {"betweenness-raw"});
    args::Flag degree_weighted(method_group, "weighted", "degree: use weighted degree", {"degree-weighted"});
    args::Flag noise_exact(method_group, "exact", "noise_corrected: exact computation instead of the approximation",
                           {"noise-corrected-exact"});

    args::Group filter_group(parser, "Filtering:");
    args::MapFlag<std::string, cb::Filter> filter(filter_group, "FILTER",
                                                  "boolean | threshold | fraction (default boolean)", {'f', "filter"},
                                                  filters, cb::Filter{cb::boolean_filter});
    args::ValueFlag<double> value(filter_group, "X", "Threshold (e.g. a p-value) or fraction of edges to keep",
                                  {'v', "value"});
    args::Flag narrate(filter_group, "narrate", "Print each method's description and filter messages", {"narrate"});

    args::Group compare_group(parser, "Comparison:");
    args::ValueFlagList<std::string> measure_names(compare_group, "MEASURE",
                                                   "Measures of every backbone vs the original network",
                                                   {'M', "measure"});
    args::ValueFlag<std::string> progression(
        compare_group, "X,Y,...", "Measures at each filter value (needs --filter threshold|fraction and --measure)",
        {"progression"});
    args::ValueFlagList<std::string> ks_names(compare_group, "DIST",
                                              "Two-sample KS statistic of each backbone's distribution vs the original",
                                              {"ks"});
    args::Flag consensus(compare_group, "consensus", "Intersection of all filtered backbones", {"consensus"});

    args::Group output_group(parser, "Output:");
    args::ValueFlag<std::string> output(output_group, "DIR",
                                        "Write <method>.scores.csv, <method>.backbone.csv and the comparison tables "
                                        "here (default: summary and tables on stdout)",
                                        {'o', "output"});
    args::Flag no_scores(output_group, "no-scores", "Do not write the per-edge score tables", {"no-scores"});
    args::Flag quiet(output_group, "quiet", "No progress bar", {'q', "quiet"});

    args::Group perf_group(parser, "Performance:");
    args::ValueFlag<int> threads(perf_group, "N", "Worker threads (default: OMP_NUM_THREADS or all cores)",
                                 {'j', "threads"});

    try {
        parser.ParseCLI(argc, argv);
    } catch (const args::Completion& e) {
        std::cout << e.what();
        return 0;
    } catch (const args::Help&) {
        std::cout << parser;
        return 0;
    } catch (const args::Error& e) {
        std::cerr << e.what() << "\n\n" << parser;
        return 2;
    }
    if (list) {
        print_list();
        return 0;
    }

    try {
        if (!input) throw UsageError("missing input CSV");
        const cb::Filter flt = args::get(filter);
        if (flt != cb::boolean_filter && !value && !(progression && !ks_names && !consensus))
            throw UsageError("--filter " + cb::filter_name(flt) + " needs --value");
        if (progression && flt == cb::boolean_filter)
            throw UsageError("--progression needs --filter threshold or fraction");
        if (progression && !measure_names) throw UsageError("--progression needs --measure");
        if (no_header && weight_name) throw UsageError("--weight-name is for files with a header");
        if (!no_header && (source_col || target_col || weight_col))
            throw UsageError("--source-col/--target-col/--weight-col need --no-header");

        auto chosen_methods = select(methods(), method_names ? split_list(args::get(method_names))
                                                             : std::vector<std::string>{"all"}, "method");
        auto chosen_measures = select(measures(), split_list(args::get(measure_names)), "measure");
        auto chosen_dists = select(distributions(), split_list(args::get(ks_names)), "distribution");
        std::vector<double> progression_values = progression ? parse_doubles(args::get(progression))
                                                             : std::vector<double>{};
        const double filter_value = value ? args::get(value) : 0;

        MethodOptions opts;
        opts.ecm_seed = args::get(ecm_seed);
        opts.modularity_seed = args::get(modularity_seed);
        opts.glanb_c = args::get(glanb_c);
        opts.betweenness_unweighted = betweenness_unweighted;
        opts.betweenness_raw = betweenness_raw;
        opts.degree_weighted = degree_weighted;
        opts.noise_corrected_exact = noise_exact;

        std::optional<fs::path> out_dir;
        if (output) {
            out_dir = args::get(output);
            fs::create_directories(*out_dir);
        }

        if (threads) cb::set_num_threads(args::get(threads));

        // ---- load ----
        auto t_load = std::chrono::steady_clock::now();
        cb::Graph g = no_header ? cb::Graph::read_csv(args::get(input), args::get(source_col), args::get(target_col),
                                                      args::get(weight_col))
                                : cb::Graph::read_csv(args::get(input));
        if (weight_name && args::get(weight_name) != "weight") {
            const std::string& w = args::get(weight_name);
            for (const auto& e : g.edges()) {
                double x = g.get(e.id, w);
                if (std::isnan(x)) throw std::runtime_error("column '" + w + "' missing on some edge");
                g.set(e.id, "weight", x);
            }
        }
        if (use_lcc) g = cb::lcc(g);
        std::cout << "network: " << g.number_of_nodes() << " nodes, " << g.number_of_edges() << " edges (loaded in "
                  << std::fixed << std::setprecision(1)
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load).count()
                  << std::defaultfloat << " s, " << cb::num_threads() << " threads)\n";

        // ---- run, filter and write each method; only backbones needed for the comparison are kept ----
        const bool comparing = !chosen_measures.empty() || !chosen_dists.empty() || consensus;
        cb::Compare cmp;
        size_t n_compared = 0;
        struct Row {
            std::string name, note;
            double seconds;
            int nodes = -1, edges = -1;
        };
        std::vector<Row> rows;
        {
            Progress progress(chosen_methods.size(), !quiet && !narrate && CPPBONE_ISATTY(2));
            for (size_t i : chosen_methods) {
                const Method& m = methods()[i];
                progress.start(m.name);
                auto t0 = std::chrono::steady_clock::now();
                std::optional<cb::Backbone> b;
                try {
                    b = m.run(g, opts);
                } catch (const std::exception& e) {
                    std::cerr << "\n" << m.name << " failed: " << e.what() << '\n';
                    progress.tick();
                    continue;
                }
                Row row{m.name, "", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()};
                if (narrate) b->narrate();
                if (accepts(*b, flt)) {
                    cb::Graph filtered = flt(*b, filter_value, narrate, "weight", false);
                    row.nodes = filtered.number_of_nodes();
                    row.edges = filtered.number_of_edges();
                    if (out_dir)
                        emit(out_dir, m.name + ".backbone.csv", [&](std::ostream& o) { filtered.to_csv(o); });
                    if (comparing) {
                        cmp.add_backbone(*b);
                        ++n_compared;
                    }
                } else {
                    row.note = "filter not supported (use " + compatible_list(*b) + ")";
                }
                if (out_dir && !no_scores) emit(out_dir, m.name + ".scores.csv", [&](std::ostream& o) { b->to_csv(o); });
                rows.push_back(row);
                progress.tick();
            }
        }

        std::cout << '\n'
                  << std::left << std::setw(32) << "method" << std::right << std::setw(10) << "time [s]"
                  << std::setw(10) << "nodes" << std::setw(10) << "edges" << "  note\n";
        for (const auto& r : rows) {
            std::cout << std::left << std::setw(32) << r.name << std::right << std::fixed << std::setprecision(3)
                      << std::setw(10) << r.seconds << std::defaultfloat;
            if (r.nodes >= 0)
                std::cout << std::setw(10) << r.nodes << std::setw(10) << r.edges;
            else
                std::cout << std::setw(10) << "-" << std::setw(10) << "-";
            std::cout << "  " << r.note << '\n';
        }

        // ---- comparison ----
        if (comparing) {
            cmp.set_network(g);
            if (n_compared == 0) throw std::runtime_error("no selected method supports the chosen filter");
            for (size_t i : chosen_measures) cmp.add_property(measures()[i].first, measures()[i].second);
            for (size_t i : chosen_dists) cmp.add_distribution(distributions()[i].first, distributions()[i].second);

            if (!progression_values.empty()) {
                cmp.set_filter(flt, progression_values);
                for (const auto& [prop, table] : cmp.properties_progression())
                    emit(out_dir, "progression_" + prop + ".csv", [&](std::ostream& o) { table.to_csv(o); });
            }
            // Graphs are only summarized on stdout; the edge list goes to the output directory.
            auto write_consensus = [&](const cb::Graph& c) {
                std::cout << "\nconsensus: " << c.number_of_nodes() << " nodes, " << c.number_of_edges() << " edges\n";
                if (out_dir) emit(out_dir, "consensus.csv", [&](std::ostream& o) { c.to_csv(o); });
            };
            // Without --value only the progression was asked for (checked above).
            if (flt == cb::boolean_filter || value) {
                cmp.set_filter(flt, std::vector<double>(n_compared, filter_value));
                if (!chosen_measures.empty())
                    emit(out_dir, "measures.csv", [&](std::ostream& o) { cmp.properties().to_csv(o); });
                if (!chosen_dists.empty()) {
                    auto ks = cmp.distribution_ks_statistic(true, std::vector<bool>(n_compared, consensus));
                    emit(out_dir, "ks.csv", [&](std::ostream& o) { ks.ks_statistics.to_csv(o); });
                    if (ks.consensual_backbone) write_consensus(*ks.consensual_backbone);
                } else if (consensus) {
                    write_consensus(cmp.consent());
                }
            }
        }
        if (out_dir) std::cout << "\nwritten to " << out_dir->string() << '\n';
    } catch (const UsageError& e) {
        std::cerr << "error: " << e.what() << "\n(run with --help for usage)\n";
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
