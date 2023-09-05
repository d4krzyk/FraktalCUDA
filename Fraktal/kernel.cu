#include <iostream>
#include <cmath>
#include <complex>
#include <fstream>
#include <cuda_runtime.h>
//#include <GLFW/glfw3.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>

#define WIDTH 800
#define HEIGHT 800


GLuint texture; // ID tekstury OpenGL
cudaGraphicsResource* cudaTextureResource; // CUDA resource dla tekstury


// Funkcja inicjalizująca OpenGL
void initOpenGL(int argc, char** argv) {
    // Inicjalizacja GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(WIDTH, HEIGHT); // Ustaw rozmiar okna
    glutInitWindowPosition(0, 0);
    glClearColor(1.0, 1.0, 1.0, 1.0);
    /*glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-2, 2, 2, -2, 1, -1);*/
    // Stwórz okno GLUT
    glutCreateWindow("CUDA-Mandelbrot");

    // Inicjalizacja GLEW dla obsługi rozszerzeń OpenGL
    glewInit();

    // Utwórz teksturę OpenGL
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, WIDTH, HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Zarejestruj teksturę w CUDA
    cudaGraphicsGLRegisterImage(&cudaTextureResource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
}


__global__
void mandelbrotGPU(float* output, int width, int height, float xmin, float xmax, float ymin, float ymax, int max_iter, float chaos_c) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float real = xmin + (xmax - xmin) * x / width;
        float imag = ymin + (ymax - ymin) * y / height;

        float2 c = make_float2(real * chaos_c, imag * chaos_c);
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
void mandelbrotCPU(float* output, int width, int height, float xmin, float xmax, float ymin, float ymax, int max_iter, float chaos_c) {

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float real = xmin + (xmax - xmin) * x / (width - 1);
            float imag = ymin + (ymax - ymin) * y / (height - 1);



            float2 c = make_float2(real * chaos_c, imag * chaos_c);
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
__host__
void saveArrayToFile(const char* filename, float* data, int size) {
    std::ofstream file(filename, std::ios::binary);
    file.write(reinterpret_cast<char*>(data), size * sizeof(float));
    file.close();
}
__host__
void saveParameters(float xmin, float xmax, float ymin, float ymax) {
    std::ofstream file("parameters_XY.txt");
    if (file.is_open()) {
        file << xmin << " " << xmax << " " << ymin << " " << ymax;
        file.close();
        std::cout << "Parametry zapisane do pliku." << std::endl;

    }
    else {
        std::cerr << "Nie można otworzyć pliku do zapisu." << std::endl;
    }

}



__global__ void processImage(float* input, uchar4* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float value = input[y * width + x];
        uchar4 color = make_uchar4(value * 255, value * 255, value * 255, 255);
        output[y * width + x] = color;
    }
}

void Display() {
    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_POINTS);

    

    glEnd();
    glutSwapBuffers();
}





int main(int argc, char** argv) {

    // Inicjalizacja OpenGL
    initOpenGL(argc, argv);


    const int width = 1600;
    const int height = 1600;
    float var_c = 0.89;

    //const float xmin = -2.0f;     const float xmax = 1.0f;
    //const float ymin = -1.5f;     const float ymax = 1.5f;

    const int max_iter = 50;

    const float xmin = -2.0f;       const float xmax = -1.54f;
    const float ymax = 0.25f;       const float ymin = -0.25f;


    saveParameters(xmin, xmax, ymin, ymax);

    float* outputCPU = new float[width * height];
    float* outputGPU;
    cudaMallocManaged(&outputGPU, width * height * sizeof(float));


    //Alokacja wątków na blok
    dim3 blockSize(32, 32);
    //Alokacja bloku na siatkę
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    do
    {
        //std::cout << "Podaj zmienną c z przedziału 0-1 np. 0.45: ";
        //std::cin >> var_c;
    } while (var_c > 1.00 && var_c < 0.0);




    // Pomiar czasu dla GPU
    cudaEvent_t startGPU, stopGPU;
    cudaEventCreate(&startGPU);
    cudaEventCreate(&stopGPU);

    cudaEventRecord(startGPU);
    mandelbrotGPU << <gridSize, blockSize >> > (outputGPU, width, height, xmin, xmax, ymin, ymax, max_iter, var_c);
    cudaEventRecord(stopGPU);
    cudaEventSynchronize(stopGPU);

    float elapsedTimeGPU;
    cudaEventElapsedTime(&elapsedTimeGPU, startGPU, stopGPU);
    std::cout << "Czas GPU: " << elapsedTimeGPU << " ms" << std::endl;


    cudaEventDestroy(startGPU);
    cudaEventDestroy(stopGPU);

    // Pomiar czasu dla CPU
    cudaEvent_t startCPU, stopCPU;
    cudaEventCreate(&startCPU);
    cudaEventCreate(&stopCPU);

    cudaEventRecord(startCPU);

    mandelbrotCPU(outputCPU, width, height, xmin, xmax, ymin, ymax, max_iter, var_c);
    cudaEventRecord(stopCPU);
    cudaEventSynchronize(stopCPU);

    float elapsedTimeCPU;
    cudaEventElapsedTime(&elapsedTimeCPU, startCPU, stopCPU);
    std::cout << "Czas CPU: " << elapsedTimeCPU << " ms" << std::endl;

    cudaEventDestroy(startCPU);
    cudaEventDestroy(stopCPU);


    cudaDeviceSynchronize();


    glutDisplayFunc(Display);
    glutMainLoop();



    // Zapisz tablicę pixels do pliku
    saveArrayToFile("output_mandela_arrayGPU.bin", outputGPU, width * height);
    saveArrayToFile("output_mandela_arrayCPU.bin", outputCPU, width * height);
    cudaFree(outputGPU);
    delete[] outputCPU;

    return 0;
}