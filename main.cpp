
#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>
#include <limits>
#include <cmath>

struct ObjectData
{
    std::vector<double> features;
    std::string label;
};

void saveObjectDatabase(const std::vector<ObjectData> &objectDatabase, const std::string &filename)
{
    std::ofstream file(filename, std::ios::out | std::ios::binary);

    if (file.is_open())
    {
        for (const auto &objData : objectDatabase)
        {
            file << objData.label << " ";

            for (const auto &feature : objData.features)
            {
                file << feature << " ";
            }

            file << std::endl;
        }

        file.close();
    }
    else
    {
        std::cerr << "Error: Unable to open file for writing." << std::endl;
    }
}

void loadObjectDatabase(std::vector<ObjectData> &objectDatabase, const std::string &filename)
{
    std::ifstream file(filename, std::ios::in | std::ios::binary);

    if (file.is_open())
    {
        objectDatabase.clear();

        while (file)
        {
            std::string label;
            file >> label;

            if (label.empty())
                break;

            ObjectData objData;
            objData.label = label;

            while (true)
            {
                double feature;
                file >> feature;

                if (file.fail())
                    break;

                objData.features.push_back(feature);
            }

            objectDatabase.push_back(objData);
        }

        file.close();
    }
    // No need to display an error if the file doesn't exist; training will create the file
}

void customGaussianBlur(const cv::Mat &input, cv::Mat &output, int kernel_size)
{
    // Create a Gaussian kernel
    cv::Mat kernel = cv::getGaussianKernel(kernel_size, -1, CV_32F);
    cv::Mat kernel_transpose = kernel.t();
    cv::Mat gaussian_kernel = kernel * kernel_transpose;

    // Convolve the image with the Gaussian kernel
    cv::filter2D(input, output, -1, gaussian_kernel, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
}

void displayRegions(cv::Mat &frame, cv::Mat &region_map, int largestN);

std::vector<double> computeFeatures(const cv::Mat &region_map, int region_id)
{
    cv::Mat region_mask = (region_map == region_id);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(region_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty() || contours[0].size() < 3)
    {
        return {};
    }

    std::vector<cv::Point> hull;
    cv::convexHull(contours[0], hull);

    double percentFilled = static_cast<double>(contourArea(hull)) / region_mask.total() * 100.0;

    cv::RotatedRect boundingBox = cv::minAreaRect(hull);
    double aspectRatio = boundingBox.size.height / boundingBox.size.width;

    std::vector<double> features = {percentFilled, aspectRatio};

    return features;
}

void displayLargestRegion(cv::Mat &frame, cv::Mat &region_map, int largestN, const std::vector<ObjectData> &objectDatabase, int &textPosY, int &largestRegionIdx, int &textPosX, std::vector<cv::Scalar> &colors, const std::string &classificationResult)
{
    // std::vector<cv::Scalar> colors;
    for (int i = 0; i < 256; ++i)
    {
        colors.push_back(cv::Scalar(rand() % 256, rand() % 256, rand() % 256));
    }

    cv::Mat labels, stats, centroids;
    region_map.convertTo(region_map, CV_8U);
    int num_labels = cv::connectedComponentsWithStats(region_map, labels, stats, centroids);

    // int largestRegionIdx = -1;
    int largestRegionArea = 0;

    for (int i = 1; i < num_labels; ++i)
    {
        int currentRegionArea = stats.at<int>(i, cv::CC_STAT_AREA);

        if (currentRegionArea >= largestN && currentRegionArea > largestRegionArea)
        {
            largestRegionIdx = i;
            largestRegionArea = currentRegionArea;
        }
    }

    if (largestRegionIdx != -1)
    {
        cv::rectangle(frame, cv::Rect(stats.at<int>(largestRegionIdx, cv::CC_STAT_LEFT), stats.at<int>(largestRegionIdx, cv::CC_STAT_TOP), stats.at<int>(largestRegionIdx, cv::CC_STAT_WIDTH), stats.at<int>(largestRegionIdx, cv::CC_STAT_HEIGHT)),
                      colors[largestRegionIdx % colors.size()], 2);

        std::string label = "Unknown"; // Default label if not found in database

        if (largestRegionIdx >= 0 && largestRegionIdx < num_labels)
        {
            for (const auto &objData : objectDatabase)
            {
                if (objData.label == std::to_string(largestRegionIdx + 1))
                {
                    label = objData.label;
                    break;
                }
            }
        }

        // Calculate the position to center the text
        textPosY = frame.rows / 2;
        textPosX = (frame.cols - label.length() * 10) / 2;

        cv::putText(frame, "Label: " + classificationResult, cv::Point(textPosX, textPosY),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, colors[largestRegionIdx % colors.size()], 2);

        std::cout << "Label: " << label << std::endl;
    }
}

double scaledEuclideanDistance(const std::vector<double> &vec1, const std::vector<double> &vec2)
{
    if (vec1.size() != vec2.size())
    {
        return std::numeric_limits<double>::infinity(); // Return infinity for vectors of different sizes
    }

    double distance = 0.0;
    for (size_t i = 0; i < vec1.size(); ++i)
    {
        distance += std::pow((vec1[i] - vec2[i]), 2) / std::pow(1.0, 2); // Assuming stdev_x = 1.0, you may adjust this
    }

    return std::sqrt(distance);
}

// Function to classify an object using the object database
std::string classifyObject(const std::vector<ObjectData> &objectDatabase, const std::vector<double> &newFeatures, double distanceThreshold)
{
    double minDistance = std::numeric_limits<double>::infinity();
    std::string minLabel = ""; // Change the default value to an empty string

    // Print the newFeatures vector for debugging
    std::cout << "New Features: ";
    for (const auto &feature : newFeatures)
    {
        std::cout << feature << " ";
    }
    std::cout << std::endl;

    for (const auto &objData : objectDatabase)
    {
        double distance = scaledEuclideanDistance(newFeatures, objData.features);

        std::cout << "Distance to label " << objData.label << ": " << distance << std::endl;

        if (distance < minDistance)
        {
            minDistance = distance;
            minLabel = objData.label;
        }
    }

    // Print additional information for debugging
    std::cout << "Minimum distance: " << minDistance << ", Threshold: " << distanceThreshold << std::endl;
    std::cout << "Chosen Label: " << minLabel << std::endl;

    if (minDistance < distanceThreshold)
    {
        return minLabel;
    }
    else
    {
        return "New Object";
    }
}

int textPosX = 0;
std::vector<cv::Scalar> colors;

int main()
{
    cv::VideoCapture cap(0);

    if (!cap.isOpened())
    {
        std::cerr << "Error: Unable to open webcam." << std::endl;
        return -1;
    }

    cv::namedWindow("Original Video", cv::WINDOW_NORMAL);
    cv::namedWindow("Blurred Video", cv::WINDOW_NORMAL);
    cv::namedWindow("Thresholded Video", cv::WINDOW_NORMAL);
    cv::namedWindow("Cleaned Video", cv::WINDOW_NORMAL);
    cv::namedWindow("Region Map Video", cv::WINDOW_NORMAL);

    int largestN = 500;

    std::vector<ObjectData> objectDatabase;
    bool trainingMode = false;
    std::string currentLabel;
    bool loadedDatabase = false; // Track if the object database is loaded

    // Load object database if not loaded
    // Load object database if not loaded
    if (!loadedDatabase)
    {
        loadObjectDatabase(objectDatabase, "object_database.txt");
        loadedDatabase = true; // Set the flag to indicate that the database is loaded

        // Check if the object database is empty, and create the file if it doesn't exist
        if (objectDatabase.empty())
        {
            std::ofstream createFile("object_database.txt");
            createFile.close();

            std::cout << "Object database file created." << std::endl;
        }

        // Add a delay to allow time for the database to load
        cv::waitKey(1000); // 1-second delay (adjust as needed)
    }
    int largestRegionIdx = -1;
    int textPosY = 0;

    while (true)
    {
        cv::Mat frame;
        cap >> frame;

        if (frame.empty())
        {
            std::cerr << "Error: End of webcam stream." << std::endl;
            break;
        }

        int kernel_size = 5;
        cv::Mat blurred;
        customGaussianBlur(frame, blurred, kernel_size);

        cv::Mat gray;
        cv::cvtColor(blurred, gray, cv::COLOR_BGR2GRAY);

        cv::Mat samples = gray.reshape(1, gray.rows * gray.cols / 16);
        samples.convertTo(samples, CV_32F);

        cv::Mat labels, centers;
        cv::kmeans(samples, 2, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0), 3, cv::KMEANS_RANDOM_CENTERS, centers);

        int threshold_value = static_cast<int>((centers.at<float>(0) + centers.at<float>(1)) / 2);

        cv::Mat thresholded;
        cv::threshold(gray, thresholded, threshold_value, 255, cv::THRESH_BINARY);
        thresholded.convertTo(thresholded, CV_8U);

        int morph_size = 5;
        cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * morph_size + 1, 2 * morph_size + 1),
                                                    cv::Point(morph_size, morph_size));

        cv::morphologyEx(thresholded, thresholded, cv::MORPH_OPEN, element);
        cv::morphologyEx(thresholded, thresholded, cv::MORPH_CLOSE, element);

        cv::imshow("Original Video", frame);
        cv::imshow("Blurred Video", blurred);
        cv::imshow("Thresholded Video", thresholded);

        cv::Mat cleaned;
        cv::morphologyEx(thresholded, cleaned, cv::MORPH_CLOSE, element);
        cv::imshow("Cleaned Video", cleaned);

        cv::Mat region_map;
        cv::connectedComponents(thresholded, region_map);

        cv::Mat region_display = frame.clone();

        // Display the largest region in classification mode
        if (!trainingMode && loadedDatabase)
        {
            double distanceThreshold = 50.0;

            // displayLargestRegion(region_display, region_map, largestN, objectDatabase, textPosY, largestRegionIdx, textPosX, colors);

            // Classify the object
            std::vector<double> features = computeFeatures(region_map, 1); // Use a specific region_id or modify as needed

            // Classify the object
            std::string classificationResult = classifyObject(objectDatabase, features, distanceThreshold);
            displayLargestRegion(region_display, region_map, largestN, objectDatabase, textPosY, largestRegionIdx, textPosX, colors, classificationResult);

            // for (int i = 0; i < classificationResults.size(); ++i)
            // {
            //     displayLargestRegion(region_display, region_map, largestN, objectDatabase, textPosY, largestRegionIdx, textPosX, colors, classificationResults[i]);
            // }

            // Display the classification result on the frame
            cv::putText(frame, "Classification: " + classificationResult, cv::Point(10, 70), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);

            // Correct the label display
            cv::putText(frame, "Label: " + classificationResult, cv::Point(textPosX, textPosY),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, colors[largestRegionIdx % colors.size()], 2);
        }
        else
        {
            // Training mode
            std::vector<double> features = computeFeatures(region_map, 1);

            // Save the feature vector along with the label to the object database
            if (!currentLabel.empty())
            {
                ObjectData objData;
                objData.features = features;
                objData.label = currentLabel;

                objectDatabase.push_back(objData);

                std::cout << "Object labeled as '" << currentLabel << "' added to the database." << std::endl;
            }

            trainingMode = false; // Exit training mode after collecting one set of features
        }

        char key = cv::waitKey(30);

        if (key == 27)
        {
            break;
        }
        else if (key == 'N' || key == 'n')
        {
            // Enter training mode
            trainingMode = true;

            // Prompt user for label
            std::cout << "Enter label for the current object: ";
            std::cin >> currentLabel;
        }
        else if (key == 'S' || key == 's')
        {
            // Save object database to file
            saveObjectDatabase(objectDatabase, "object_database.txt");
            std::cout << "Object database saved to 'object_database.txt'" << std::endl;
        }
        else
        {
            // Normal mode
            trainingMode = false;
        }

        cv::imshow("Region Map Video", region_display); // Display the region map
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}

void displayRegions(cv::Mat &frame, cv::Mat &region_map, int largestN)
{
    // Random color palette for visualization
    std::vector<cv::Scalar> colors;
    for (int i = 0; i < 256; ++i)
    {
        colors.push_back(cv::Scalar(rand() % 256, rand() % 256, rand() % 256));
    }

    // Get connected components with stats
    cv::Mat labels, stats, centroids;
    region_map.convertTo(region_map, CV_8U); // Ensure the region_map is of type CV_8U
    int num_labels = cv::connectedComponentsWithStats(region_map, labels, stats, centroids);

    // Filter regions based on size
    for (int i = 1; i < num_labels; ++i)
    { // Start from 1 to exclude the background
        if (stats.at<int>(i, cv::CC_STAT_AREA) >= largestN)
        {
            // Draw bounding box and label
            cv::rectangle(frame, cv::Rect(stats.at<int>(i, cv::CC_STAT_LEFT), stats.at<int>(i, cv::CC_STAT_TOP), stats.at<int>(i, cv::CC_STAT_WIDTH), stats.at<int>(i, cv::CC_STAT_HEIGHT)),
                          colors[i % colors.size()], 2);
            cv::putText(frame, std::to_string(i), cv::Point(centroids.at<double>(i, 0), centroids.at<double>(i, 1)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, colors[i % colors.size()], 2);

            // Display features
            std::vector<double> features = computeFeatures(region_map, i);
            cv::putText(frame, "Percent Filled: " + std::to_string(features[0]), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
            cv::putText(frame, "Aspect Ratio: " + std::to_string(features[1]), cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
        }
    }
}
