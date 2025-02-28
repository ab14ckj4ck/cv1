#include <dirent.h>
#include <sys/stat.h>

#include <fstream>
#include <iostream>
#include <vector>

#include "algorithms.h"
#include "opencv2/opencv.hpp"
#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"

// #define FULL_VERSION 1
// #define FINAL_RUN 1
// #define GENERATE_REF 1

#define RST "\x1B[0m"
#define KRED "\x1B[31m"
#define KGRN "\x1B[32m"

#define FRED(x) KRED x RST
#define FGRN(x) KGRN x RST

#define BOLD(x) "\x1B[1m" x RST

#if GENERATE_REF || FINAL_RUN
struct reference
{
    cv::Mat *mat;
    cv::Point *point;
    int *value_px;
    float *value_real;

    reference(cv::Mat *m) : mat(m), point(nullptr), value_px(nullptr), value_real(nullptr) {}

    reference(cv::Point *pt) : mat(nullptr), point(pt), value_px(nullptr), value_real(nullptr) {}

    reference(int *val) : mat(nullptr), point(nullptr), value_px(val), value_real(nullptr) {}

    reference(float *val) : mat(nullptr), point(nullptr), value_px(nullptr), value_real(val) {}
};
#endif

//===============================================================================
// Configuration
//-------------------------------------------------------------------------------
// TODO:
//  - Nothing!
//  - Do not change anything here
//===============================================================================
struct Config
{
    // gaussian blur
    int kernel_size_gauss = 0;
    float sigma_gauss = 0.f;

    // canny
    int canny_lower = 0;
    int canny_upper = 0;

    // morph operations
    int kernel_size_morph = 0;
};

//===============================================================================
// make_directory()
//-------------------------------------------------------------------------------
// TODO:
//  - Nothing!
//  - Do not change anything here
//===============================================================================
void make_directory(const char *path)
{
#if defined(_WIN32)
    _mkdir(path);
#else
    mkdir(path, 0777);
#endif
}

//===============================================================================
// is_path_existing()
//-------------------------------------------------------------------------------
// TODO:
//  - Nothing!
//  - Do not change anything here
//===============================================================================
bool is_path_existing(const char *path)
{
    struct stat buffer
    {
    };
    return (stat(path, &buffer)) == 0;
}

#if GENERATE_REF
//===============================================================================
// generate_ref()
//-------------------------------------------------------------------------------
// TODO:
//  - Nothing!
//  - Do not change anything here
//===============================================================================
void generate_ref(std::string path, reference ref)
{
    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    if (ref.mat != nullptr)
    {
        fs << "image" << *ref.mat;
    }
    else if (ref.point != nullptr)
    {
        fs << "image" << *ref.point;
    }
    else if (ref.value_px != nullptr)
    {
        fs << "image" << *ref.value_px;
    }
    else if (ref.value_real != nullptr)
    {
        fs << "image" << *ref.value_real;
    }
}
#endif

#if FINAL_RUN
//===============================================================================
// get_ref_image()
//-------------------------------------------------------------------------------
// TODO:
//  - Nothing!
//  - Do not change anything here
//===============================================================================
void get_ref_image(std::string ref_directory, std::string name, reference ref)
{
    struct dirent *entry;
    DIR *dir = opendir((ref_directory).c_str());
    while ((entry = readdir(dir)) != NULL)
    {
        std::string entry_name = entry->d_name;
        if (entry_name.find(name) != std::string::npos)
        {
            std::string full_path = ref_directory + name;
            cv::FileStorage fs(full_path, cv::FileStorage::READ);
            if (ref.mat != nullptr)
            {
                fs["image"] >> *ref.mat;
            }
            else if (ref.point != nullptr)
            {
                fs["image"] >> *ref.point;
            }
            else if (ref.value_px != nullptr)
            {
                fs["image"] >> *ref.value_px;
            }
            else if (ref.value_real != nullptr)
            {
                fs["image"] >> *ref.value_real;
            }
            break;
        }
    }
    closedir(dir);
}
#endif

//===============================================================================
// save_image()
//-------------------------------------------------------------------------------
// TODO:
//  - Nothing!
//  - Do not change anything here
//===============================================================================
void save_image(const std::string &out_directory, const std::string &name, const size_t number, const cv::Mat &image)
{
    std::stringstream number_stringstream;
    number_stringstream << std::setfill('0') << std::setw(2) << number;
    std::string path = out_directory + number_stringstream.str() + "_" + name + ".png";
    cv::imwrite(path, image);
    std::cout << "saving image: " << path << std::endl;
}

//===============================================================================
// save_value()
//-------------------------------------------------------------------------------
// std::ios_base::openmode
//  + std::ios_base::trunc  -> for first call, clear file before write
//  + std::ios_base::app    -> for second call, if you want to append to the file (new line)
// TODO:
//  - Nothing!
//  - Do not change anything here
//===============================================================================
void save_value(const std::string &out_directory, const std::string &name, const size_t number,
                const std::string &field_name, const float value, const int precision,
                const std::ios_base::openmode mode = std::ios_base::trunc)
{
    std::stringstream number_stringstream;
    number_stringstream << std::setfill('0') << std::setw(2) << number;
    std::string path = out_directory + number_stringstream.str() + "_" + name + ".txt";
    std::ofstream file(path, mode);
    if (file.is_open())
    {
        file << field_name << ": " << std::fixed << std::setprecision(precision) << value << std::endl;
        file.close();
        std::cout << "saving value: " << path << std::endl;
    }
    else
    {
        std::cout << BOLD(FRED("[ERROR]")) << " Could not save value to (" << path << ")" << std::endl;
    }
}

//===============================================================================
// run()
//-------------------------------------------------------------------------------
// TODO:
//  - Nothing!
//  - Do not change anything here
//===============================================================================
void run(const cv::Mat &input_image_1, const cv::Mat &input_image_2, const std::string &out_directory, const std::string &ref_directory, Config config)
{
    size_t output_counter = 0;
    int source_image_counter = 0;
    const cv::Mat input_images[] = {input_image_1, input_image_2};

    //=============================================================================
    // Grayscale image
    //=============================================================================
    std::cout << "Step  1 - calculating grayscale images... " << std::endl;
    cv::Mat grayscale_images[2];

    for (cv::Mat input_image : input_images) {
        cv::Mat grayscale = cv::Mat::zeros(input_image.size(), CV_8UC1);
        algorithms::compute_grayscale(input_image, grayscale);

        save_image(out_directory, "grayscale_0" + std::to_string(source_image_counter + 1), ++output_counter, grayscale); 

        #if GENERATE_REF
            generate_ref(
                ref_directory + "0" + std::to_string(output_counter - 1) 
                    + "_grayscale_0" + std::to_string(source_image_counter + 1) + ".json", 
                reference{&grayscale}
            );
        #endif
        #if FINAL_RUN
            get_ref_image(
                ref_directory, 
                "0" + std::to_string(output_counter - 1) 
                    + "_grayscale_0" + std::to_string(source_image_counter + 1) + ".json", 
                reference{&grayscale}
            );
        #endif

        grayscale_images[source_image_counter++] = grayscale;
    }

    source_image_counter = 0;


    //=============================================================================
    // Gaussian blur
    //=============================================================================
    std::cout << "Step 2 - applying gaussian blur... " << std::endl;
    cv::Mat blurred_images[2];

    for(cv::Mat grayscale : grayscale_images) {
        cv::Mat blurred_image = cv::Mat::zeros(input_images[source_image_counter].size(), CV_8UC1);
        algorithms::gaussian_blur(grayscale, config.kernel_size_gauss, config.sigma_gauss, blurred_image);

        save_image(out_directory, "blurred_image_0" + std::to_string(source_image_counter + 1), ++output_counter, blurred_image);

        #if GENERATE_REF
            generate_ref(
                ref_directory + "0" + std::to_string(output_counter - 1) 
                    + "_blurred_image_0" + std::to_string(source_image_counter + 1) + ".json", 
                reference{&blurred_image}
            );
        #endif
        #if FINAL_RUN
            get_ref_image(
                ref_directory, "0" + std::to_string(output_counter - 1) 
                    + "_blurred_image_0" + std::to_string(source_image_counter + 1) + ".json",
                reference{&blurred_image}
            );
        #endif

        blurred_images[source_image_counter++] = blurred_image;
    }

    source_image_counter = 0;


    //=============================================================================
    // Logarithmic transform
    //=============================================================================
    std::cout << "Step  3 - computing logarithmic transform... " << std::endl;
    cv::Mat log_images[2];

    for(cv::Mat blur : blurred_images) {
        cv::Mat log_transform_tmp = cv::Mat::zeros(input_images[source_image_counter].size(), CV_32F);

        cv::Mat blur_tmp = cv::Mat::zeros(input_images[source_image_counter].size(), CV_32F);
        blur.convertTo(blur_tmp, CV_32F);

        algorithms::compute_log_transform(blur_tmp, log_transform_tmp);

        cv::Mat log_transform = cv::Mat::zeros(input_images[source_image_counter].size(), CV_8UC1);
        log_transform_tmp.convertTo(log_transform, CV_8UC1);

        // save intermediate
        save_image(out_directory, "log_transform_0" + std::to_string(source_image_counter + 1), ++output_counter, log_transform);

        #if GENERATE_REF
            generate_ref(
                ref_directory + "0" + std::to_string(output_counter - 1) 
                    + "_log_transform_0" + std::to_string(source_image_counter + 1) + ".json", 
                reference{&log_transform}
            );
        #endif
        #if FINAL_RUN
            get_ref_image(
                ref_directory, "0" + std::to_string(output_counter - 1) 
                    + "_log_transform_0" + std::to_string(source_image_counter + 1) + ".json",
                reference{&log_transform}
            );
        #endif

        log_images[source_image_counter++] = log_transform;
    }

    source_image_counter = 0;


    //=============================================================================
    // Bilateral filter
    //=============================================================================
    std::cout << "Step  4 - applying the bilateral filter... " << std::endl;
    cv::Mat bilateral_filtered_images[2];

    for(cv::Mat log_image : log_images) {
        cv::Mat filtered_image = cv::Mat::zeros(input_images[source_image_counter].size(), CV_8UC1);
        algorithms::apply_bilateral_filter(log_image, filtered_image);

        // save intermediate
        save_image(out_directory, "bilateral_filter_0" + std::to_string(source_image_counter + 1), ++output_counter, filtered_image);

        #if GENERATE_REF
            generate_ref(
                ref_directory + "0" + std::to_string(output_counter - 1) 
                    + "_bilateral_filtered_0" + std::to_string(source_image_counter + 1) + ".json", 
                reference{&filtered_image}
            );
        #endif
        #if FINAL_RUN
            get_ref_image(
                ref_directory, "0" + std::to_string(output_counter - 1) 
                    + "_bilateral_filtered_0" + std::to_string(source_image_counter + 1) + ".json",
                reference{&filtered_image}
            );
        #endif

        bilateral_filtered_images[source_image_counter++] = filtered_image;
    }

    source_image_counter = 0;


    //=============================================================================
    // Canny edge detector
    //=============================================================================
    std::cout << "Step 5 - calculating canny edges... " << std::endl;

    cv::Mat canny_images[2];

    for(cv::Mat filtered_image : bilateral_filtered_images) {

        cv::Mat edges = cv::Mat::zeros(input_images[source_image_counter].size(), CV_8UC1);
        algorithms::canny(filtered_image, config.canny_lower, config.canny_upper, edges);
        save_image(out_directory, "canny_0" + std::to_string(source_image_counter + 1), ++output_counter, edges);

        #if GENERATE_REF
            generate_ref(
                ref_directory + "0" + std::to_string(output_counter - 1) 
                    + "_canny_edges_0" + std::to_string(source_image_counter + 1) + ".json", 
                reference{&edges}
            );
        #endif
        #if FINAL_RUN
            get_ref_image(
                ref_directory, "0" + std::to_string(output_counter - 1) 
                    + "_canny_edges_0" + std::to_string(source_image_counter + 1) + ".json",
                reference{&edges}
            );
        #endif

        canny_images[source_image_counter++] = edges;
    }

    source_image_counter = 0;


    //=============================================================================
    // Morphological closing
    //=============================================================================
    std::cout << "Step  6 - applying morphological operations (DILATE)... " << std::endl;
    cv::Mat dil_images[2];

    for(cv::Mat canny : canny_images) {
        cv::Mat dilated_image = cv::Mat::zeros(canny.size(), CV_8UC1);
        algorithms::apply_morph_operation(canny, config.kernel_size_morph, cv::MORPH_DILATE, dilated_image);
        save_image(out_directory, "dilation_applied_0" + std::to_string(source_image_counter + 1), ++output_counter, dilated_image);

        #if GENERATE_REF
            generate_ref(
                ref_directory + std::to_string(output_counter - 1) 
                    + "_dilated_0" + std::to_string(source_image_counter + 1) + ".json", 
                reference{&dilated_image}
            );
        #endif
        #if FINAL_RUN
            get_ref_image(
                ref_directory, std::to_string(output_counter - 1) 
                    + "_dilated_0" + std::to_string(source_image_counter + 1) + ".json",
                reference{&dilated_image}
            );
        #endif
        
        dil_images[source_image_counter++] = dilated_image;
    }

    source_image_counter = 0;

    std::cout << "Step  7 - applying morphological operations (ERODE)... " << std::endl;
    cv::Mat closed_images[2];

    for(cv::Mat dil : dil_images) {
        cv::Mat closed_image = cv::Mat::zeros(dil.size(), CV_8UC1);
        algorithms::apply_morph_operation(dil, config.kernel_size_morph, cv::MORPH_ERODE, closed_image);
        save_image(out_directory, "closing_applied_0" + std::to_string(source_image_counter + 1), ++output_counter, closed_image);

        #if GENERATE_REF
            generate_ref(
                ref_directory + std::to_string(output_counter - 1) 
                    + "_closed_0" + std::to_string(source_image_counter + 1) + ".json", 
                reference{&closed_image}
            );
        #endif
        #if FINAL_RUN
            get_ref_image(
                ref_directory, std::to_string(output_counter - 1) 
                    + "_closed_0" + std::to_string(source_image_counter + 1) + ".json",
                reference{&closed_image}
            );
        #endif
        
        closed_images[source_image_counter++] = closed_image;
    }

    source_image_counter = 0;


    //=============================================================================
    // Crack matching
    //=============================================================================
    std::cout << "Step  8 - matching the cracks... " << std::endl;
    int row_count = grayscale_images[0].rows + grayscale_images[1].rows;
    int col_count = grayscale_images[0].cols + grayscale_images[1].cols;

    // Testing own distance transform
    cv::Mat dist_transform_img1 = cv::Mat::zeros(closed_images[0].rows, closed_images[0].cols, CV_8UC1);
    cv::Mat dist_transform_img1_norm = cv::Mat::zeros(closed_images[0].rows, closed_images[0].cols, CV_8UC1);

    cv::Mat dist_transform_img2 = cv::Mat::zeros(closed_images[1].rows, closed_images[1].cols, CV_8UC1);
    cv::Mat dist_transform_img2_norm = cv::Mat::zeros(closed_images[1].rows, closed_images[1].cols, CV_8UC1);

    algorithms::L2_distance_transform(closed_images[0], dist_transform_img1);
    cv::normalize(dist_transform_img1, dist_transform_img1_norm, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    save_image(out_directory, "distance_trasform_01", ++output_counter, dist_transform_img1_norm);

    algorithms::L2_distance_transform(closed_images[1], dist_transform_img2);
    cv::normalize(dist_transform_img2, dist_transform_img2_norm, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    save_image(out_directory, "distance_trasform_02", ++output_counter, dist_transform_img2_norm);

    // Bruteforce matching
    cv::Mat matching_frame = cv::Mat::zeros(row_count, col_count, CV_8UC1);
    cv::Mat minimal_distances = cv::Mat::zeros(row_count, col_count, CV_8UC1);
    cv::Mat minimal_distances_norm = cv::Mat::zeros(row_count, col_count, CV_8UC1);
    cv::Mat mask_img1 = cv::Mat::zeros(row_count, col_count, CV_8UC1);
    cv::Mat mask_img2 = cv::Mat::zeros(row_count, col_count, CV_8UC1);

    algorithms::match_cracks(closed_images[0], closed_images[1], matching_frame, 13, minimal_distances, mask_img1, mask_img2);

    cv::normalize(minimal_distances, minimal_distances_norm, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    // save intermediates
    save_image(out_directory, "minimal_distances", ++output_counter, minimal_distances_norm);
    save_image(out_directory, "mask_01", ++output_counter, mask_img1);
    save_image(out_directory, "mask_02", ++output_counter, mask_img2);

    #if GENERATE_REF
        generate_ref(
            ref_directory + std::to_string(output_counter - 5) + "_minimal_distances.json", 
            reference{&minimal_distances}
        );
        generate_ref(
            ref_directory + std::to_string(output_counter - 4) + "_mask_01.json", reference{&mask_img1}
        );
        generate_ref(
            ref_directory + std::to_string(output_counter - 3) + "_mask_02.json", reference{&mask_img2}
        );
    #endif
    #if FINAL_RUN
        get_ref_image(
            ref_directory, std::to_string(output_counter - 5) + "_minimal_distances.json", 
            reference{&minimal_distances}
        );
        get_ref_image(
            ref_directory, std::to_string(output_counter - 4) + "_mask_01.json", reference{&mask_img1}
        );
        get_ref_image(
            ref_directory, std::to_string(output_counter - 3) + "_mask_02.json", reference{&mask_img2}
        );
    #endif


    //=============================================================================
    // Blend original images
    //=============================================================================
    std::cout << "Step 9 - blending the original images... " << std::endl;
    cv::Mat blended_original = cv::Mat::zeros(row_count, col_count, CV_8UC3);
    algorithms::blend_originals(input_images[0], input_images[1], mask_img1, mask_img2, blended_original);
    save_image(out_directory, "blended_original", ++output_counter, blended_original);
    #if GENERATE_REF
        generate_ref(ref_directory + std::to_string(output_counter - 3) + "_blended_original.json", reference{&blended_original});
    #endif
    #if FINAL_RUN
        get_ref_image(ref_directory, std::to_string(output_counter - 3) + "_blended_original.json", reference{&blended_original});
    #endif


    // ==============================================================================
    // Bonus: Custom Canny
    // ==============================================================================
    std::cout << "Bonus  - running custom canny edge detector... " << std::endl;
    for (cv::Mat filtered_image : bilateral_filtered_images) {

        cv::Mat canny_edges = cv::Mat::zeros(filtered_image.size(), CV_8UC1);
        
        cv::Mat gradient_x = cv::Mat::zeros(filtered_image.size(), CV_8UC1);
        cv::Mat gradient_y = cv::Mat::zeros(filtered_image.size(), CV_8UC1);
        cv::Mat gradient_abs = cv::Mat::zeros(filtered_image.size(), CV_8UC1);
        algorithms::compute_gradient(filtered_image, gradient_x, gradient_y, gradient_abs);

        save_image(out_directory+"bonus/", "gradient_x_0" + std::to_string(source_image_counter + 1), ++output_counter, gradient_x);

        #if GENERATE_REF
            generate_ref(ref_directory + std::to_string(output_counter - 3) + "_bonus_gradient_x_" 
                + std::to_string(source_image_counter) + ".json", reference{&gradient_x});
        #endif
        #if FINAL_RUN
            get_ref_image(ref_directory, std::to_string(output_counter - 3) + "_bonus_gradient_x_" 
                + std::to_string(source_image_counter) + ".json", reference{&gradient_x});
        #endif

        save_image(out_directory+"bonus/", "gradient_y_0" + std::to_string(source_image_counter + 1), ++output_counter, gradient_y);

        #if GENERATE_REF
            generate_ref(ref_directory + std::to_string(output_counter - 3) + "_bonus_gradient_y_" 
                + std::to_string(source_image_counter) + ".json", reference{&gradient_y});
        #endif
        #if FINAL_RUN
            get_ref_image(ref_directory, std::to_string(output_counter - 3) + "_bonus_gradient_y_" 
                + std::to_string(source_image_counter) + ".json", reference{&gradient_y});
        #endif

        save_image(out_directory+"bonus/", "gradient_abs_0" + std::to_string(source_image_counter + 1), ++output_counter, gradient_abs);

        #if GENERATE_REF
            generate_ref(ref_directory + std::to_string(output_counter - 3) + "_bonus_gradient_abs_" 
                + std::to_string(source_image_counter) + ".json", reference{&gradient_abs});
        #endif
        #if FINAL_RUN
            get_ref_image(ref_directory, std::to_string(output_counter - 3) + "_bonus_gradient_abs_" 
                + std::to_string(source_image_counter) + ".json", reference{&gradient_abs});
        #endif

        cv::Mat maxima = cv::Mat::zeros(filtered_image.size(), CV_32FC1);
        algorithms::non_maxima_suppression(gradient_abs, gradient_x, gradient_y, maxima);
        
        maxima.convertTo(maxima, CV_8UC1);
        save_image(out_directory+"bonus/", "maxima_0" + std::to_string(source_image_counter + 1), ++output_counter, maxima);
        
        #if GENERATE_REF
            generate_ref(ref_directory + std::to_string(output_counter - 3) + "_bonus_maxima_" 
                + std::to_string(source_image_counter) + ".json", reference{&maxima});
        #endif
        #if FINAL_RUN
            get_ref_image(ref_directory, std::to_string(output_counter - 3) + "_bonus_maxima_" 
                + std::to_string(source_image_counter) + ".json", reference{&maxima});
        #endif

        algorithms::hysteresis(maxima, config.canny_lower, config.canny_upper, canny_edges); 
            
        save_image(out_directory+"bonus/", "canny_0" + std::to_string(source_image_counter + 1), ++output_counter, canny_edges);

        #if GENERATE_REF
            generate_ref(ref_directory + std::to_string(output_counter - 3) + "_bonus_canny_" 
                + std::to_string(source_image_counter) + ".json", reference{&canny_edges});
        #endif
        #if FINAL_RUN
            get_ref_image(ref_directory, std::to_string(output_counter - 3) + "_bonus_canny_" 
                + std::to_string(source_image_counter) + ".json", reference{&canny_edges});
        #endif

        source_image_counter++;
    }

    source_image_counter = 0;
}

//===============================================================================
// execute_testcase()
//-------------------------------------------------------------------------------
// TODO:
//  - Nothing!
//  - Do not change anything here
//===============================================================================
void execute_testcase(const rapidjson::Value &config_data)
{
    //=============================================================================
    // Parse input data
    //=============================================================================
    std::string name = config_data["name"].GetString();
    std::string image_path = config_data["image_1_path"].GetString();
    std::string image_2_path = config_data["image_2_path"].GetString();

    Config config;

    // gaussian blur
    config.sigma_gauss = (float)config_data["sigma_gauss"].GetDouble();
    config.kernel_size_gauss = (int)config_data["kernel_size_gauss"].GetUint();

    // canny
    config.canny_lower = (int)config_data["canny_lower"].GetUint();
    config.canny_upper = (int)config_data["canny_upper"].GetUint();

    // morph operations
    config.kernel_size_morph = (int)config_data["kernel_size_morph"].GetUint();

    //=============================================================================
    // Load input images
    //=============================================================================
    
    // Image 1
    std::cout << BOLD(FGRN("[INFO]")) << " Input image 1: " << image_path << std::endl;

    cv::Mat img = cv::imread(image_path);

    if (!img.data)
    {
        std::cout << BOLD(FRED("[ERROR]")) << " Could not load image (" << image_path << ")" << std::endl;
        throw std::runtime_error("Could not load file");
    }

    // Image 2
    std::cout << BOLD(FGRN("[INFO]")) << " Input image 2: " << image_2_path << std::endl;

    cv::Mat img2 = cv::imread(image_2_path);

    if (!img2.data)
    {
        std::cout << BOLD(FRED("[ERROR]")) << " Could not load image (" << image_2_path << ")" << std::endl;
        throw std::runtime_error("Could not load file");
    }

    //=============================================================================
    // Create output directory
    //=============================================================================
    std::string output_directory = "output/" + name + "/";

    std::cout << BOLD(FGRN("[INFO]")) << " Output path: " << output_directory << std::endl;

    make_directory("output/");
    make_directory(output_directory.c_str());
    // create bonus directory
    make_directory((output_directory + "/bonus/").c_str());

    std::string ref_path = "data/intm/";
    std::string ref_directory = ref_path + name + "/";

#if FINAL_RUN
    if (!is_path_existing(ref_directory.c_str()))
    {
        std::cout << BOLD(FRED("[ERROR]")) << " ref directory does not exist!" << std::endl;
        std::cout << BOLD(FGRN("[INFO]")) << " execute with GENERATE_REF 1 first" << std::endl;
        throw std::runtime_error("Could not load ref files");
    }
    else
    {
        std::cout << "opening ref directory" << std::endl;
    }
#endif

#if GENERATE_REF
    make_directory(ref_path.c_str());
    make_directory(ref_directory.c_str());
#endif

    //=============================================================================
    // Starting default task
    //=============================================================================
    std::cout << "Starting MAIN Task..." << std::endl;
    run(img, img2, output_directory, ref_directory, config);
}

//===============================================================================
// main()
//-------------------------------------------------------------------------------
// TODO:
//  - Nothing!
//  - Do not change anything here
//===============================================================================
int main(int argc, char *argv[])
{
    std::cout << "CV/task1 framework version 1.0" << std::endl;  // DO NOT REMOVE THIS LINE!!!
    std::cout << "===================================" << std::endl;
    std::cout << "               CV Task 1           " << std::endl;
    std::cout << "===================================" << std::endl;

    if (argc != 2)
    {
        std::cout << "Usage: " << argv[0] << " <config-file>" << std::endl;
        return 1;
    }

    std::string path = std::string(argv[1]);
    std::ifstream fs(path);
    if (!fs)
    {
        std::cout << "Error: Failed to open file " << path << std::endl;
        return 2;
    }
    std::stringstream buffer;
    buffer << fs.rdbuf();

    rapidjson::Document doc;
    rapidjson::ParseResult check;
    check = doc.Parse<0>(buffer.str().c_str());

    if (check)
    {
        if (doc.HasMember("testcases"))
        {
            rapidjson::Value &testcases = doc["testcases"];
            for (rapidjson::SizeType i = 0; i < testcases.Size(); i++)
            {
                rapidjson::Value &testcase = testcases[i];
                try
                {
                    execute_testcase(testcase);
                }
                catch (const std::exception &e)
                {
                    std::cout << e.what() << std::endl;
                    std::cout << BOLD(FRED("[ERROR]")) << " Program exited with errors!" << std::endl;
                    return -1;
                }
            }
        }
        std::cout << "Program exited normally!" << std::endl;
    }
    else
    {
        std::cout << "Error: Failed to parse file " << argv[1] << ":" << check.Offset() << std::endl;
        return 3;
    }
    return 0;
}
