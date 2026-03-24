#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <numeric>
#include <limits>
#include <chrono>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include "SLIC_Algorithm_AoS_Sequential.h"
#include "SLIC_Algorithm_SoA_Sequential.h"
#include "SLIC_Algorithm_AoS_Parallel.h"
#include "SLIC_Algorithm_SoA_Parallel.h"
#include "SLIC_common.h"

namespace os = std::filesystem;

struct BenchmarkConfig {
    int num_runs = 15;
    int warm_up_runs = 5;
    std::vector<int> threads = {1, 2, 4, 8, 16, 32};
    std::vector<int> chunks = {1, 10, 50, 100};
    int K = 1000;
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
        << res.mean << "," << res.std_dev<< "\n";
    csv.flush();
}

// Convert enum OpenMP in string
std::string sched_to_str(omp_sched_t s) {
    if (s == omp_sched_static) return "static";
    if (s == omp_sched_dynamic) return "dynamic";
    if (s == omp_sched_guided) return "guided";
    return "auto";
}


// Exec N runs and calculate statistics (mean, stddev)
// We use this function to measure the performance of an algorithm
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


        double time=(end - start) * 1000.0;
        //std::cout << "Run " << i << ": " << time << " ms" << std::endl;
        times.push_back(time);

    }

    //  Calculate the statistics
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = sum / static_cast<double>(times.size());
    double sq_sum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0);
    double std_dev = std::sqrt(sq_sum / static_cast<double>(times.size()) - mean * mean);

    return {mean, std_dev};
}

void optimize_algorithm(SLIC_Algorithm* algo, const BenchmarkConfig& cfg, std::ofstream& csv,int best_thread_num=8) {
    std::cout << "\n=== TUNING: " << algo->get_name() << " ===" << std::endl;

    // Checks if it is sequential and log performance.
     if (!algo->is_parallel()) {
        Result res = measure_performance(algo, 1, omp_sched_static, 0, cfg);
        std::cout << "Sequential Time: " << res.mean << " ms (+/- " << res.std_dev << ")" << std::endl;
        log_to_csv(csv, algo->get_name(), 1, "N/A", 0, res,false);
        return;
    }

    if (algo->use_tiling) {
        std::cout << "Using Tiling Optimization. \n" << std::endl;
    } else {
        std::cout << "Without Tiling Optimization. \n" << std::endl;
    }


    std::vector<omp_sched_t> schedules = {omp_sched_static, omp_sched_dynamic, omp_sched_guided};

    for (auto sched : schedules) {
        for (int chunk : cfg.chunks) {
            Result res = measure_performance(algo, best_thread_num, sched, chunk, cfg);
            std::cout << "Sched: " << sched_to_str(sched) << " | Chk: " << chunk << " -> " << res.mean << " ms" << std::endl;
            log_to_csv(csv, algo->get_name(), best_thread_num, sched_to_str(sched), chunk, res,algo->use_tiling);
        }
    }
}

int variant_experiment_N_size(const BenchmarkConfig& cfg, const std::string& path_img, int factor_size, std::string csv_name) {
    cv::Mat image = cv::imread(path_img);
    if (image.empty()) { std::cerr << "Err: Image not found!" << std::endl; return -1; }
    cv::Mat image_original = image.clone();
    if (factor_size != 1) {
        cv::resize(image, image_original, cv::Size(), factor_size, factor_size, cv::INTER_CUBIC);
    }
    // Convert to LAB color space
    cv::Mat image_lab;
    cv::cvtColor(image_original, image_lab, cv::COLOR_BGR2Lab);
    cv::imshow("Input Image",image_original);
    cv::waitKey(0);
    std::cout << "Loaded image: " << path_img << " (" << image_original.cols << "x" << image_original.rows << ")" << std::endl;

    //Visualize the result of one algorithm.
    SLIC_Algorithm_SoA_Parallel par_aos(image_lab, cfg.K, cfg.m, cfg.iterations);
    par_aos.run();
    cv::Mat result = par_aos.display_boundaries();
    cv::imshow("Result Image", result);
    cv::waitKey(0);

    //Prepare the CSV file to log the results, one for resize factor
    std::string factor_str = std::to_string(factor_size);
    csv_name += "x" + factor_str + ".csv";
    std::ofstream csv(csv_name);
    csv << "Algorithm, Tile, Threads, Schedule, Chunk, Mean_ms, StdDev_ms \n";
    {
        SLIC_Algorithm_AoS_Sequential seq_aos(image_lab, cfg.K, cfg.m, cfg.iterations);
        optimize_algorithm(&seq_aos, cfg, csv);
    }
    {
        SLIC_Algorithm_SoA_Sequential seq_soa(image_lab, cfg.K, cfg.m, cfg.iterations);
        optimize_algorithm(&seq_soa, cfg, csv);
    }
    {
        SLIC_Algorithm_AoS_Parallel par_aos_notiled(image_lab, cfg.K, cfg.m, cfg.iterations);
        optimize_algorithm(&par_aos_notiled, cfg, csv);
    }
    {
        SLIC_Algorithm_SoA_Parallel par_soa_notiled(image_lab, cfg.K, cfg.m, cfg.iterations);
        optimize_algorithm(&par_soa_notiled, cfg, csv);
    }
    {
        SLIC_Algorithm_SoA_Parallel par_soa_tiled(image_lab, cfg.K, cfg.m, cfg.iterations);
        par_soa_tiled.set_tiling(true);
        optimize_algorithm(&par_soa_tiled, cfg, csv);
    }
    {
        SLIC_Algorithm_AoS_Parallel par_aos_tiled(image_lab, cfg.K, cfg.m, cfg.iterations);
        par_aos_tiled.set_tiling(true);
        optimize_algorithm(&par_aos_tiled, cfg, csv);
    }

    std::cout << "\nBenchmark finished. Data in the file: "<< csv_name << "\n" << std::endl;
    return 0;
}

void get_avg_time_num_thread(const std::string& alg, const BenchmarkConfig& cfg) {
    std::string imgs_path = PATH_images;
    std::vector<double>mean_times;
    std::vector<std::string> valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"};
    std::cout << "STRUCTURE TYPE:" << alg << std::endl;
    for (int t : cfg.threads) {
        std::vector<double> times;
        std::cout<< "\n--- Testing with " << t << " threads ---" << std::endl;
        for (const auto& entry : os::directory_iterator(imgs_path)) {
            std::string img_path = entry.path().string();
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            bool is_valid = false;
            for (const auto& valid_ext : valid_extensions) {
                if (ext == valid_ext) {
                    is_valid = true;
                    break;
                }
            }

            if (!is_valid) continue;

            if (img_path.empty()) continue;

            cv::Mat image = cv::imread(img_path);
            if (image.empty()) {
                std::cerr << "Error: Impossible to load the  image from " << img_path << std::endl;
                return;
            }
            cv::Mat image_lab;
            cv::cvtColor(image, image_lab, cv::COLOR_BGR2Lab);
            SLIC_Algorithm* algo = nullptr;
            if (alg== "aos") {
                algo = new SLIC_Algorithm_AoS_Parallel(image_lab, cfg.K, cfg.m, cfg.iterations);
            }else {
                algo = new SLIC_Algorithm_SoA_Parallel(image_lab, cfg.K, cfg.m, cfg.iterations);
            }

            omp_set_num_threads(t);

            omp_set_schedule(omp_sched_static, 0);


            Result res = measure_performance(algo, t, omp_sched_static, 0, cfg);
            times.push_back(res.mean);
            //std::cout << "Time (mean): " << res.mean << " ms" << std::endl;

            delete algo;
        }
        std::cout << std::endl;
        // Calculate average time for this number of threads
        double sum = std::accumulate(times.begin(), times.end(), 0.0);
        double mean = sum / static_cast<double>(times.size());
        mean_times.push_back(mean);
    }
    std::cout << std::endl;
    // Print the best time and with the corresponding number of threads
    double min_time = std::numeric_limits<double>::max();
    int best_threads = 0;
    for (int i = 0; i < cfg.threads.size(); i++) {
        std::cout << "Threads: " << cfg.threads[i] << " -> Avg Time: " << mean_times[i] << " ms" << std::endl;
        if (mean_times[i] < min_time) {
            min_time = mean_times[i];
            best_threads = i;
        }
    }
    std::cout << "\n>>> Best Threads: " << cfg.threads[best_threads] << " with Avg Time: " << min_time << " ms\n" << std::endl;
    // We want to save all the average times in a CSV file
    //Create a directory where to save the results, one csv for algorithm
    if (!os::exists("../all_benchmark_results/num_thread_experiments")) {
        os::create_directory("../all_benchmark_results/num_thread_experiments");
    }

    std::string csv_name = "benchmark_avg_time_threads_";
    csv_name = "../all_benchmark_results/num_thread_experiments/" + csv_name + alg + ".csv";
    std::ofstream csv(csv_name);
    if (!csv.is_open()) {
        std::cerr << "Errore: Impossibile aprire il file " << csv_name << std::endl;
        return;
    }
    csv << "Threads, Avg_Time_ms\n";
    for (int i=0; i<cfg.threads.size(); i++) {
        csv << cfg.threads[i] << "," << mean_times[i] << "\n";
    }
}

//This function is to verify the complexity of the SLIC algorithm
// Doesn't matter the layout of the data structure
void get_time_for_complexity(int num_factor, const BenchmarkConfig& cfg, int num_thread) {
    std::string img_path = get_random_image_path(PATH_images);
    SLIC_Algorithm *algo = nullptr;
    cv::Mat resize_image;

    if (img_path.empty()) return;

    std::string folder_path_str = "../all_benchmark_results/complexity_experiment";
    os::path folder_path(folder_path_str);


    if (!os::exists(folder_path)) {
        if (!os::create_directories(folder_path)) {
            std::cerr << "Errore: Impossibile creare la directory " << folder_path << std::endl;
            return;
        }
    }

    cv::Mat image = cv::imread(img_path);
    cv::Mat image_lab;
    cv::cvtColor(image, image_lab, cv::COLOR_BGR2Lab);

    os::path csv_file_path = folder_path / "benchmark_complexity.csv";

    std::ofstream csv(csv_file_path.string());
    if (!csv.is_open()) {
        std::cerr << "Errore: Impossibile aprire/creare il file CSV in: " << csv_file_path << std::endl;
        return;
    }


    csv << "Image_Size_Factor, Mean_Time_ms, Parallel\n";
    csv.flush();
    for (int i=1; i<= num_factor; i++) {
        cv::resize(image_lab, resize_image, cv::Size(), i, i);
        std::cout<< "Testing with image size factor: " << i << " (" << resize_image.cols << "x" << resize_image.rows << ")" << std::endl;
        algo= new SLIC_Algorithm_SoA_Parallel(resize_image, cfg.K, cfg.m, cfg.iterations);
        omp_set_num_threads(num_thread);
        omp_set_schedule(omp_sched_static, 0);
        Result res = measure_performance(algo, num_thread, omp_sched_static, 0, cfg);
        std::cout << "Image Size Factor: " << i << " -> Time (mean): " << res.mean << " ms" << std::endl;
        csv << i << "," << res.mean << "," << 1 <<"\n";
        csv.flush();
        delete algo;
    }
    std::cout<< "\nNow testing the sequential version...\n" << std::endl;
    for (int i=1; i<= num_factor; i++) {
        cv::resize(image_lab, resize_image, cv::Size(), i, i);
        std::cout<< "Testing with image size factor: " << i << " (" << resize_image.cols << "x" << resize_image.rows << ")" << std::endl;
        algo= new SLIC_Algorithm_SoA_Sequential(resize_image, cfg.K, cfg.m, cfg.iterations);
        Result res = measure_performance(algo, num_thread, omp_sched_static, 0, cfg);
        std::cout << "Image Size Factor: " << i << " -> Time (mean): " << res.mean << " ms" << std::endl;
        csv << i << "," << res.mean << "," << 0 <<"\n";
        csv.flush();
        delete algo;
    }
    csv.close();

    }

// Funzione per ottenere le prime N immagini dalla cartella
std::vector<std::string> get_first_N_images(const std::string& directory_path, int N) {
    std::vector<std::string> image_paths;
    std::vector<std::string> valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif"};

    try {
        if (!os::exists(directory_path)) {
            std::cerr << "Directory non trovata: " << directory_path << std::endl;
            return image_paths;
        }

        for (const auto& entry : os::directory_iterator(directory_path)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                // Converti estensione in lowercase per confronto sicuro
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

                for (const auto& valid_ext : valid_extensions) {
                    if (ext == valid_ext) {
                        image_paths.push_back(entry.path().string());
                        break;
                    }
                }
            }
            if (image_paths.size() >= N) break; // Ci fermiamo appena ne abbiamo N
        }
    } catch (const std::exception& e) {
        std::cerr << "Errore nella lettura directory: " << e.what() << std::endl;
    }

    std::cout << "Trovate " << image_paths.size() << " immagini per il benchmark." << std::endl;
    return image_paths;
}

void run_averaged_benchmark(const BenchmarkConfig& cfg, int num_images_to_test, bool tile, bool parallel, bool reduction=true) {
    std::vector<std::string> images = get_first_N_images(PATH_images, num_images_to_test);

    if (images.empty()) {
        std::cerr << "No image found for the benchmark!" << std::endl;
        return;
    }

    std::vector<cv::Size> resolutions = {
        cv::Size(640, 480),   // VGA
        cv::Size(1280, 720),  // HD
        cv::Size(1920, 1080), // Full HD
    };

    std::vector<omp_sched_t> schedules = {omp_sched_static, omp_sched_dynamic, omp_sched_guided};
    std::vector<int> chunks = {1, 10, 50, 100};

    if (!parallel) {
        schedules = {omp_sched_static};
        chunks = {0};
    }


    std::string filename = "../all_benchmark_results/benchmark_experiments/avg_bench";
    filename += (parallel ? "_parallel" : "_sequential");
    if (parallel) {
        filename += (tile ? "_tiled" : "_notiled");
        if (reduction) {
            filename += "_reduction";
        } else {
            filename += "_atomic";
        }
    }
    filename += ".csv";

    std::ofstream csv(filename);
    if (!csv.is_open()) {
        std::cerr << "Errore: Impossibile aprire il file " << filename << std::endl;
        return;
    }
    csv << "Resolution,Num_Pixels,Schedule,Chunk,AoS_Mean_ms,AoS_StdDev_ms,SoA_Mean_ms,SoA_StdDev_ms\n";
    int fixed_threads = 8;

    for (const auto& res : resolutions) {
        long num_pixels = res.width * res.height;
        std::cout << "\n=== Testing Resolution: " << res.width << "x" << res.height << " ===" << std::endl;

        for (auto sch : schedules) {
            for (auto chunk : chunks) {
                double sum_time_aos = 0.0;
                double sum_time_soa = 0.0;

                double sq_sum_time_aos = 0.0;
                double sq_sum_time_soa = 0.0;

                std::cout << "  Config: " << (parallel ? sched_to_str(sch) : "Seq")
                          << " | Chunk: " << chunk << " | Images: " << std::flush;

                int processed_count = 0;

                for (const auto& img_path : images) {
                    cv::Mat raw_img = cv::imread(img_path);
                    if (raw_img.empty()) continue;

                    cv::Mat image, image_lab;
                    cv::resize(raw_img, image, res);
                    cv::cvtColor(image, image_lab, cv::COLOR_BGR2Lab);

                    double time_aos = 0;
                    double time_soa = 0;

                    if (parallel) {
                        // --- AoS Parallel ---
                        SLIC_Algorithm* aos_par = new SLIC_Algorithm_AoS_Parallel(image_lab, cfg.K, cfg.m, cfg.iterations, reduction);
                        if (tile) aos_par->set_tiling(true);
                        Result res_aos = measure_performance(aos_par, fixed_threads, sch, chunk, cfg);
                        time_aos = res_aos.mean; // Prendiamo il tempo medio di questa immagine
                        delete aos_par;

                        // --- SoA Parallel ---
                        SLIC_Algorithm* soa_par = new SLIC_Algorithm_SoA_Parallel(image_lab, cfg.K, cfg.m, cfg.iterations, reduction);
                        if (tile) soa_par->set_tiling(true);
                        Result res_soa = measure_performance(soa_par, fixed_threads, sch, chunk, cfg);
                        time_soa = res_soa.mean;
                        delete soa_par;

                    } else {
                        // --- Sequential ---
                        //std::cout<< "Sequential AoS \n"<< std::endl;
                        SLIC_Algorithm* aos_seq = new SLIC_Algorithm_AoS_Sequential(image_lab, cfg.K, cfg.m, cfg.iterations);
                        if (tile) aos_seq->set_tiling(true);
                        Result res_aos = measure_performance(aos_seq, 1, omp_sched_static, 0, cfg);
                        time_aos = res_aos.mean;
                        delete aos_seq;
                        //std::cout<< "Sequential SoA \n"<< std::endl;
                        SLIC_Algorithm* soa_seq = new SLIC_Algorithm_SoA_Sequential(image_lab, cfg.K, cfg.m, cfg.iterations);
                        if (tile) soa_seq->set_tiling(true);
                        Result res_soa = measure_performance(soa_seq, 1, omp_sched_static, 0, cfg);
                        time_soa = res_soa.mean;
                        delete soa_seq;
                    }
                    sum_time_aos += time_aos;
                    sq_sum_time_aos += (time_aos * time_aos);

                    sum_time_soa += time_soa;
                    sq_sum_time_soa += (time_soa * time_soa);

                    processed_count++;
                    std::cout << "#" << std::flush;
                }


                if (processed_count > 0) {
                    double final_avg_aos = sum_time_aos / processed_count;
                    double final_avg_soa = sum_time_soa / processed_count;

                    double var_aos = (sq_sum_time_aos / processed_count) - (final_avg_aos * final_avg_aos);
                    double var_soa = (sq_sum_time_soa / processed_count) - (final_avg_soa * final_avg_soa);

                    if (var_aos < 0) var_aos = 0;
                    if (var_soa < 0) var_soa = 0;
                    double std_aos = std::sqrt(var_aos);
                    double std_soa = std::sqrt(var_soa);

                    std::string res_name = std::to_string(res.width) + "x" + std::to_string(res.height);
                    std::string sched_name = parallel ? sched_to_str(sch) : "sequential";


                    csv << res_name << ","
                        << num_pixels << ","
                        << sched_name << ","
                        << chunk << ","
                        << final_avg_aos << ","
                        << std_aos << ","
                        << final_avg_soa << ","
                        << std_soa << "\n";


                    std::cout << " -> Done. (AoS: " << (int)final_avg_aos << "ms)(SoA:"<<(int)final_avg_soa<< "ms)" << std::endl;
                }
            }
        }
    }
    std::cout << "\nBenchmark completato! File salvato: " << filename << std::endl;
}

void profile_AllPhases(const BenchmarkConfig& cfg, SLIC_Algorithm* algo) {

    std::string filepath = "../all_benchmark_results/amdahl_law_experiment";

    if (os::exists(filepath) == false) {
        os::create_directory(filepath);
    }

    filepath += "/amdahl_law_experiment.csv";
    std::ofstream csv(filepath);
    if (!csv.is_open()) {
        std::cerr << "Errore: Impossibile aprire il file " << filepath << std::endl;
        return;
    }

    csv << "Initialization_ms, Assignment_ms, Update_ms, EnforceConnectivity_ms, FinalUpdate_ms, Total_ms\n";



    // --- WARM UP ---
    for (int i = 0; i < cfg.warm_up_runs; i++) {
        algo->Initialization();
        for (int j = 0; j < 10; j++) {
            algo->iteration();
            algo->update_centroids();
        }
        int K = algo->EnforceConnectivity();
        algo->set_K(K);
        algo->update_centroids();
        algo->clear();
        algo->set_K(cfg.K);
    }

    // --- PROFILING ---
    using std::chrono::high_resolution_clock;
    using std::chrono::duration;

    auto start_total = high_resolution_clock::now();

    // 1. Initialization
    auto start_init = high_resolution_clock::now();
    algo->Initialization();
    auto end_init = high_resolution_clock::now();
    double time_init = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end_init - start_init).count();

    // 2. Iteration (Assignment) & Update Loop
    double time_assign = 0.0;
    double time_update = 0.0;

    for (int i = 0; i < 10; i++) {
        // Assignment phase
        auto start_assign = high_resolution_clock::now();
        algo->iteration();
        auto end_assign = high_resolution_clock::now();
        time_assign += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end_assign - start_assign).count(); // Accumulo!

        // Update phase
        auto start_update = high_resolution_clock::now();
        algo->update_centroids();
        auto end_update = high_resolution_clock::now();
        time_update += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end_update - start_update).count();
    }

    // 3. Enforce Connectivity
    auto start_enhance = high_resolution_clock::now();
    int K = algo->EnforceConnectivity();
    auto end_enhance = high_resolution_clock::now();
    double time_enhance = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end_enhance - start_enhance).count();

    algo->set_K(K);
    auto start_final = high_resolution_clock::now();
    algo->update_centroids();
    auto end_final = high_resolution_clock::now();
    double time_final_update = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end_final - start_final).count();

    auto end_total = high_resolution_clock::now();
    double time_total = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end_total - start_total).count();

    std::cout << "\n=== SLIC Execution Time Breakdown ===" << std::endl;
    std::cout << "1. Initialization:      " << time_init << " ms (" << (time_init / time_total) * 100.0 << " %)" << std::endl;
    std::cout << "2. Assignment (10x):    " << time_assign << " ms (" << (time_assign / time_total) * 100.0 << " %)" << std::endl;
    std::cout << "3. Update (10x):        " << time_update << " ms (" << (time_update / time_total) * 100.0 << " %)" << std::endl;
    std::cout << "4. Enforce Connect.:    " << time_enhance << " ms (" << (time_enhance / time_total) * 100.0 << " %)" << std::endl;
    std::cout << "5. Final Update:        " << time_final_update << " ms (" << (time_final_update / time_total) * 100.0 << " %)" << std::endl;
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "TOTAL TIME:             " << time_total << " ms" << std::endl;
    std::cout << "=====================================\n" << std::endl;

    csv << time_init <<"," << time_assign << "," << time_update << "," << time_enhance << "," <<time_final_update <<
        "," << time_total << "\n";
    csv.close();

    // Reset for next runs
    algo->set_K(cfg.K);
    algo->clear();
}

void display_evolution(SLIC_Algorithm* algo) {
    int old_k= algo->k();
    algo->Initialization();
    for (int i = 0; i < 10; i++) {
        algo->iteration();
        algo->update_centroids();
    }
    int K = algo->EnforceConnectivity();
    algo->set_K(K);
    algo->update_centroids();
    cv::Mat img1 = algo->display_boundaries();
    algo->set_K(old_k);
    algo->clear();
}


int main() {

    BenchmarkConfig cfg;

    if (os::exists("../all_benchmark_results")==false) {
        os::create_directory("../all_benchmark_results");
    }
    /*
    get_avg_time_num_thread("aos",cfg);
    get_avg_time_num_thread("soa",cfg);
    get_time_for_complexity(6, cfg, 8);

    // After this experiment we can see that the best number of threads is 8 for both the data layouts.
    std::string img_path = get_random_image_path(PATH_images);
    if (img_path.empty()) return -1;
    // Experiment with different image sizes
    // Increase or Decrease the number of superpixel don't affect the benchmark procedure
    // This is because SLIC algorithm has a complexity of O(N).
    if (!os::exists("../all_benchmark_results/benchmark_experiments")) {
        os::create_directory("../all_benchmark_results/benchmark_experiments");
    }

    //Sequential
    std::cout << "\n--- Sequential Benchmark ---\n" << std::endl;
    run_averaged_benchmark(cfg, 3,false,false);

    std::cout << "\n--- Parallel Benchmark without Tiling and with Reduction ---\n" << std::endl;
    //Parallel without Tiling and with reduction
    run_averaged_benchmark(cfg, 3,false,true);

    std::cout << "\n--- Parallel Benchmark without Tiling and with Atomic --- \n" << std::endl;
    //Parallel without tiling and with atomic
    run_averaged_benchmark(cfg,3,false,true,false);

    std::cout << "\n--- Parallel Benchmark with Tiling and with Reduction ---\n" << std::endl;
    //Parallel with Tiling and with reduction
    run_averaged_benchmark(cfg, 3,true,true);

    std::cout << "\n--- Parallel Benchmark with Tiling and with Atomic --- \n" << std::endl;
    //Parallel with Tiling and with atomic
    run_averaged_benchmark(cfg,3,true,true,false);
*/
    //Amhdal Law to see if we can parallelize the Enforce Connectivity function
    cv::Mat raw_image = cv::imread(PATH_example);
    cv::Mat image, image_lab;
    cv::resize(raw_image, image, cv::Size(), 4, 4, cv::INTER_CUBIC);
    cv::cvtColor(image, image_lab, cv::COLOR_BGR2Lab);
    SLIC_Algorithm_AoS_Sequential seq_aos(image_lab, cfg.K, cfg.m, cfg.iterations);

    SLIC_Algorithm_SoA_Parallel par_aos(image_lab, cfg.K, cfg.m, cfg.iterations);
    //profile_AllPhases(cfg, &seq_aos);
    cv::imshow("Image 1", image);
    cv::waitKey(0);

    display_evolution( &seq_aos);

    cv::imshow("Image 2", image);
    cv::waitKey(0);
    display_evolution(&par_aos);

    return 0;
}