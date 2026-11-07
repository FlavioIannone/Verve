#include <iostream>
#include <vector>
#include <chrono>
// SDL2 Include
#define SDL_MAIN_HANDLED
#include <SDL.h>

int main(int argc, char *argv[])
{
    std::cout << "Initializing SDL:\n";

    // Initializes the SDL library
    if (SDL_InitSubSystem(SDL_INIT_VIDEO | SDL_INIT_EVENTS) < 0)
    {
        std::cerr << "Error SDL2 Initialization : " << SDL_GetError();
        return 1;
    }

    const int width = 800;
    const int height = 600;

    // Initializes a window
    SDL_Window *window = SDL_CreateWindow("Test", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width, height, SDL_WINDOW_MAXIMIZED);
    if (window == nullptr)
    {
        std::cout << "Error window creation: " << SDL_GetError();
        return 3;
    }

    // Initializes a renderer
    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (renderer == nullptr)
    {
        std::cout << "Error renderer creation";
        return 4;
    }

    // Initializes a texture
    SDL_Texture *streamingTexture = SDL_CreateTexture(renderer, SDL_GetWindowPixelFormat(window), SDL_TEXTUREACCESS_STREAMING, width, height);

    if (streamingTexture == nullptr)
    {
        std::cout << "Error while creating the streaming texture: " << SDL_GetError();
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();

        return -1;
    }
    SDL_Event e;

    // Color buffer (32-bit per pixel: A-R-G-B)
    std::vector<uint32_t> colorBuffer = std::vector(width * height, 0xFF000000);
    uint8_t frameColorCounter = 0;
    int frameCount = 0;
    auto startTime = std::chrono::high_resolution_clock::now();
    bool isRunning = true;

    // Event loop
    while (isRunning)
    {
        while (SDL_PollEvent(&e))
        {
            if (e.type == SDL_QUIT)
                isRunning = false;
            break;
        }

        frameColorCounter++;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                // RGB values (A will be 0xFF)
                uint8_t r = static_cast<uint8_t>((y * 255) / height);
                uint8_t g = static_cast<uint8_t>((x * 255) / width);
                uint8_t b = frameColorCounter;

                // Bit-packing ARGB32 Alpha[31-24], Red[23-16], Green[15-8], Blue[7-0]
                colorBuffer[y * width + x] = (0xFF << 24) | (r << 16) | (g << 8) | b;
            }
        }

        void *texturePixels = nullptr;
        int pitch = 0;
        // Lock the texture to write the pixels
        SDL_LockTexture(streamingTexture, nullptr, &texturePixels, &pitch);

        // Copying the pixels from the RAM to the VRAM
        std::memcpy(texturePixels, colorBuffer.data(), width * height * sizeof(uint32_t));

        // Unlock the texture
        SDL_UnlockTexture(streamingTexture);

        SDL_RenderClear(renderer); // Clears the screen
        SDL_RenderCopy(renderer, streamingTexture, nullptr, nullptr);
        SDL_RenderPresent(renderer);

        frameCount++;
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    double elapsedSeconds = std::chrono::duration<double>(endTime - startTime).count();

    std::cout << " Elapsed time: " << elapsedSeconds << "\n";
    std::cout << " Average FPS: " << static_cast<long int>(frameCount / elapsedSeconds) << "\n";

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}