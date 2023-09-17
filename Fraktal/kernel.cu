#include <iostream>
#include <cmath>
#include <complex>
#include <fstream>
#include <cuda_runtime.h>
//#include <GLFW/glfw3.h>
//#include <GL/glew.h>
//#include <GL/freeglut.h>
//#include <cuda_gl_interop.h>
#include <opencv2/opencv.hpp>
#define WIDTH 800
#define HEIGHT 800



__global__
void mandelbrotGPU(float* output, int width, int height, float xmin, float xmax, float ymin, float ymax, int max_iter, float chaos_cr, float chaos_ci) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float real = xmin + (xmax - xmin) * x / width;
        float imag = ymin + (ymax - ymin) * y / height;

        float2 c = make_float2(real * chaos_cr, imag * chaos_ci);
        float2 z = c;

        int color = max_iter;

        for (int i = 0; i < max_iter; ++i) {
            if (z.x * z.x + z.y * z.y > 4.0f) {
                color = i;
                break;
            }
            float temp = z.x;
            z.x = z.x * z.x - z.y * z.y + c.x;
            z.y = 2.0f * temp * z.y + c.y;
        }

        output[y * width + x] = static_cast<float>(color) / max_iter;
    }
}

__host__
void mandelbrotCPU(float* output, int width, int height, float xmin, float xmax, float ymin, float ymax, int max_iter, float chaos_cr, float chaos_ci) {

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float real = xmin + (xmax - xmin) * x / (width - 1);
            float imag = ymin + (ymax - ymin) * y / (height - 1);



            float2 c = make_float2(real * chaos_cr, imag * chaos_ci);
            float2 z = c;

            int color = max_iter;

            for (int i = 0; i < max_iter; ++i) {
                if (z.x * z.x + z.y * z.y > 4.0f) {
                    color = i;
                    break;
                }
                float temp = z.x;
                z.x = z.x * z.x - z.y * z.y + c.x;
                z.y = 2.0f * temp * z.y + c.y;
            }

            output[y * width + x] = static_cast<float>(color) / max_iter;
        }
    }
}




int main(int argc, char** argv) {

    const int width = WIDTH;
    const int height = HEIGHT;

    float var_cr = 0.89f;
    float var_ci = 0.89f;
    const int max_iter = 120;


    const float xmin = -2.0f;     const float xmax = 1.0f;
    const float ymin = -1.5f;     const float ymax = 1.5f;

    

    //const float xmin = -2.0f;       const float xmax = -1.54f;
    //const float ymax = 0.25f;       const float ymin = -0.25f;



    float* outputGPU; // Dane w pamięci GPU
    float* outputCPU = new float[width * height]; // Dane w pamięci CPU
    cudaMallocManaged(&outputGPU, width * height * sizeof(float));


    //Alokacja wątków na blok
    dim3 blockSize(32, 32);
    //Alokacja bloku na siatkę
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    do
    {
        //std::cout << "Podaj 2 zmienne cr (real) i ci (imaginary) z przedziału 0-1 (zalecane jest podawanie wartości większych od 0.5) np. 0.75: ";
        //std::cin >> var_cr >> var_ci;
    } while (var_cr > 1.00 && var_cr < 0.0 && var_ci > 1.00 && var_ci < 0.0);




    

    // Pomiar czasu dla CPU
    cudaEvent_t startCPU, stopCPU;
    cudaEventCreate(&startCPU);
    cudaEventCreate(&stopCPU);

    cudaEventRecord(startCPU);

    mandelbrotCPU(outputCPU, width, height, xmin, xmax, ymin, ymax, max_iter, var_cr, var_ci);
    cudaEventRecord(stopCPU);
    cudaEventSynchronize(stopCPU);

    float elapsedTimeCPU;
    cudaEventElapsedTime(&elapsedTimeCPU, startCPU, stopCPU);
    std::cout << "Czas CPU: " << elapsedTimeCPU << " ms" << std::endl;


    cudaEventDestroy(startCPU);
    cudaEventDestroy(stopCPU);


    // Pomiar czasu dla GPU
    cudaEvent_t startGPU, stopGPU;
    cudaEventCreate(&startGPU);
    cudaEventCreate(&stopGPU);

    cudaEventRecord(startGPU);
    mandelbrotGPU < <<gridSize, blockSize>> > (outputGPU, width, height, xmin, xmax, ymin, ymax, max_iter, var_cr, var_ci);
    // Przekonwertuj dane na obraz na CPU
    float* gpuImage = new float[width * height];
    cudaMemcpy(gpuImage, outputGPU, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stopGPU);
    cudaEventSynchronize(stopGPU);

    cudaDeviceSynchronize();



    float elapsedTimeGPU;
    cudaEventElapsedTime(&elapsedTimeGPU, startGPU, stopGPU);
    std::cout << "Czas GPU: " << elapsedTimeGPU << " ms" << std::endl;


    cudaEventDestroy(startGPU);
    cudaEventDestroy(stopGPU);
    

    // Twórz obiekt OpenCV Mat z wynikowego obrazu
    cv::Mat FraktalOutputImageGPU(height, width, CV_32F, gpuImage);
    cv::Mat FraktalOutputImageCPU(height, width, CV_32F, outputCPU);

    // Pozycja tekstu (lewy górny róg)
    cv::Point textPosition(10, 30);
    // Kolor tekstu (BGR)
    cv::Scalar textColor(255, 255, 255);

    std::string CPUTimeString = "Time CPU: " + std::to_string(elapsedTimeCPU) + "ms";
    std::string GPUTimeString = "Time GPU: " + std::to_string(elapsedTimeGPU) + "ms";

    // Dodaj tekst do obrazu
    cv::putText(FraktalOutputImageCPU, CPUTimeString, textPosition, cv::FONT_HERSHEY_SIMPLEX, 1, textColor, 2);
    cv::putText(FraktalOutputImageGPU, GPUTimeString, textPosition, cv::FONT_HERSHEY_SIMPLEX, 1, textColor, 2);

    cv::Mat colorImageCPU;
    cv::Mat colorImageGPU;

    FraktalOutputImageCPU.convertTo(colorImageCPU, CV_8UC3, 255);
    FraktalOutputImageGPU.convertTo(colorImageGPU, CV_8UC3, 255);
    cv::applyColorMap(colorImageCPU, colorImageCPU, cv::COLORMAP_HOT);
    cv::applyColorMap(colorImageGPU, colorImageGPU, cv::COLORMAP_HOT);

    // Wyświetl obrazy
    cv::imshow("CPU Fraktal Mandelbrot", colorImageCPU);
    cv::imshow("GPU Fraktal Mandelbrot", colorImageGPU);

    cv::waitKey(0);


    cudaFree(outputGPU);
    
    delete[] outputCPU;

    return 0;
}