#include <string>
#include <array>
#include <span>

class Image {
public:
    Image ( std::string );
    ~Image();
    int getHeight() const { return height; }
    int getWidth() const { return width; }
    unsigned char *getData() { return data; } 
    std::span<int> operator[]( int rowIndex );

private:
    int height, width, channels;
    unsigned char *data;
};