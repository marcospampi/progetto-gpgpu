#include "image.hpp"
#include <string>
#include <stdexcept>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

Image::Image( std::string path ) {
    data = stbi_load( path.c_str(), &width, &height, &channels, 4 );
    if ( data == nullptr ) {
        throw std::runtime_error("Cannot load image.");
    }
}
Image::~Image( ) {
    stbi_image_free(data);
}

std::span<int> Image::operator[]( int i ) {
    auto data = reinterpret_cast<int*>(this->data);
    return std::span<int>( data + (i*width), width );
}