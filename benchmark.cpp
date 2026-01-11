#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <numeric>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include "SLIC_Algorithm_AoS_Sequential.h"
#include "SLIC_Algorithm_SoA_Sequential.h"
#include "SLIC_Algorithm_AoS_Parallel.h"
#include "SLIC_Algorithm_SoA_Parallel.h"
#include "SLIC_common.h"

struct BenchmarkConfig {
    int num_runs = 5;
    int warm_up_runs = 1;
    std::vector<int> threads = {1, 2, 4, 8, 16};
    std::vector<int> chunks = {1, 10, 50, 100};
    int K = 200;
    int m = 10;
    int iterations = 10;
};

struct Result {
    double mean;
    double std_dev;
};

// Write on CSV file
void log_to_csv(std::ofstream& csv, const std::string& algo, int threads,
                const std::string& sched, int chunk, const Result& res,bool tile) {
    csv << algo << "," << tile << "," << threads << "," << sched << "," << chunk << ","
        << res.mean << "," << res.std_dev << "\n";
    csv.flush(); // Forza la scrittura su disco subito
}

// Convert enum OpenMP in string
std::string sched_to_str(omp_sched_t s) {
    if (s == omp_sched_static) return "static";
    if (s == omp_sched_dynamic) return "dynamic";
    if (s == omp_sched_guided) return "guided";
    return "auto";
}

// Exec N runs and calculate statistics (mean, stddev)
Result measure_performance(SLIC_Algorithm* algo, int threads, omp_sched_t sched, int chunk, const BenchmarkConfig& cfg) {
    if (algo->is_parallel()) {
        omp_set_num_threads(threads);
        omp_set_schedule(sched, chunk);
    }

    std::vector<double> times;
    times.reserve(cfg.num_runs);

    // Warm up of the cache
    for(int i=0; i<cfg.warm_up_runs; i++) {
        algo->clear();
        algo->run();
    }

    // Measure of the real executing time
    for (int i = 0; i < cfg.num_runs; i++) {
        algo->clear();

        double start = omp_get_wtime();
        algo->run();
        double end = omp_get_wtime();

        times.push_back((end - start) * 1000.0); // ms
    }

    //  Calculate the statistics
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = sum / times.size();

    double sq_sum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0);
    double std_dev = std::sqrt(sq_sum / times.size() - mean * mean);

    return {mean, std_dev};
}

//
void optimize_algorithm(SLIC_Algorithm* algo, const BenchmarkConfig& cfg, std::ofstream& csv) {
    std::cout << "\n=== TUNING: " << algo->get_name() << " ===" << std::endl;

    // Checks if it is sequential and log performance.
     if (!algo->is_parallel()) {
        Result res = measure_performance(algo, 1, omp_sched_static, 0, cfg);
        std::cout << "Sequential Time: " << res.mean << " ms (+/- " << res.std_dev << ")" << std::endl;
        log_to_csv(csv, algo->get_name(), 1, "N/A", 0, res,false);
        return;
    }

    // Then if parallel, checks for best threads.
    double best_time = 1e9;
    int best_threads = 1;

    std::cout << "--- Fase 1: Scaling Threads ---" << std::endl;
    for (int t : cfg.threads) {
        Result res = measure_performance(algo, t, omp_sched_static, 0, cfg);

        std::cout << "Thr: " << t << " -> " << res.mean << " ms" << std::endl;
        log_to_csv(csv, algo->get_name(), t, "static", 0, res,algo->use_tiling);

        if (res.mean < best_time) {
            best_time = res.mean;
            best_threads = t;
        }
    }
    std::cout << ">>> Best Threads: " << best_threads << std::endl;

    // For the parallel case, want to checks the best scheduling type.
    std::cout << "--- Fase 2: Scheduling & Chunking (con Thr=" << best_threads << ") ---" << std::endl;

    std::vector<omp_sched_t> schedules = {omp_sched_static, omp_sched_dynamic, omp_sched_guided};

    for (auto sched : schedules) {
        for (int chunk : cfg.chunks) {
            Result res = measure_performance(algo, best_threads, sched, chunk, cfg);

            std::cout << "Sched: " << sched_to_str(sched) << " | Chk: " << chunk << " -> " << res.mean << " ms" << std::endl;
            log_to_csv(csv, algo->get_name(), best_threads, sched_to_str(sched), chunk, res,algo->use_tiling);
        }
    }
}

int main() {
    BenchmarkConfig cfg;

    std::string img_path = get_random_image_path(PATH_images);
    if (img_path.empty()) return -1;

    cv::Mat image = cv::imread(img_path);
    if (image.empty()) { std::cerr << "Err: Immagine non trovata!" << std::endl; return -1; }
    cv::Mat image_original = image.clone();
    cv::resize(image, image_original, cv::Size(), 4.0, 4.0, cv::INTER_CUBIC);

    cv::Mat image_lab;
    cv::cvtColor(image_original, image_lab, cv::COLOR_BGR2Lab);

    std::cout << "Immagine caricata: " << img_path << " (" << image.cols << "x" << image.rows << ")" << std::endl;

    // File Output
    std::ofstream csv("../final_benchmark.csv");
    csv << "Algorithm,Tile,Threads,Schedule,Chunk,Mean_ms,StdDev_ms\n";

    SLIC_Algorithm_AoS_Sequential seq_aos(image_lab, cfg.K, cfg.m, cfg.iterations);
    SLIC_Algorithm_SoA_Sequential seq_soa(image_lab, cfg.K, cfg.m, cfg.iterations);

    SLIC_Algorithm_AoS_Parallel par_aos(image_lab, cfg.K, cfg.m, cfg.iterations);
    SLIC_Algorithm_SoA_Parallel par_soa(image_lab, cfg.K, cfg.m, cfg.iterations);

    SLIC_Algorithm_SoA_Parallel par_soa_tiled(image_lab, cfg.K, cfg.m, cfg.iterations);
    par_soa_tiled.set_tiling(true);
    SLIC_Algorithm_AoS_Parallel par_aos_tiled(image_lab, cfg.K, cfg.m, cfg.iterations);
    par_aos_tiled.set_tiling(true);

    // Sequential
    optimize_algorithm(&seq_aos, cfg, csv);
    optimize_algorithm(&seq_soa, cfg, csv);

    // Parallel
    optimize_algorithm(&par_aos, cfg, csv);
    optimize_algorithm(&par_soa, cfg, csv);
    optimize_algorithm(&par_soa_tiled, cfg, csv);
    optimize_algorithm(&par_aos_tiled, cfg, csv);

    std::cout << "\nBenchmark concluso. Dati in 'final_benchmark.csv'" << std::endl;


    // Increase or Decrease the number of superpixel don't affect the benchmark procedure
    // This is because SLIC algorithm has a complexity of O(N).





    return 0;
}