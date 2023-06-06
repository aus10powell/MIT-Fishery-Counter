#include <iostream>
#include <string>

extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/timestamp.h>
}

int main(int argc, char* argv[]) {
    // Check if the input file path is provided
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }

    const std::string inputFilePath = argv[1];

    // Register all formats and codecs with FFmpeg
    av_register_all();

    // Open the input file
    AVFormatContext* formatContext = nullptr;
    if (avformat_open_input(&formatContext, inputFilePath.c_str(), nullptr, nullptr) < 0) {
        std::cerr << "Error opening input file" << std::endl;
        return 1;
    }

    // Retrieve stream information
    if (avformat_find_stream_info(formatContext, nullptr) < 0) {
        std::cerr << "Error finding stream information" << std::endl;
        avformat_close_input(&formatContext);
        return 1;
    }

    // Check the format of the input file
    const std::string formatName = av_guess_format(nullptr, inputFilePath.c_str(), nullptr)->name;
    if (formatName != "avi" && formatName != "mkv" && formatName != "mp4") {
        std::cerr << "Invalid input file format. Only AVI, MKV, and MP4 formats are supported." << std::endl;
        avformat_close_input(&formatContext);
        return 1;
    }

    // Iterate through the streams and find video stream
    AVStream* videoStream = nullptr;
    for (unsigned int i = 0; i < formatContext->nb_streams; ++i) {
        if (formatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            videoStream = formatContext->streams[i];
            break;
        }
    }

    if (videoStream == nullptr) {
        std::cerr << "No video stream found in the input file" << std::endl;
        avformat_close_input(&formatContext);
        return 1;
    }

    // Segment duration in seconds (1 minute = 60 seconds)
    const int segmentDuration = 60;

    // Start time of the current segment
    int64_t segmentStartTime = 0;

    // Read packets and split video into segments
    AVPacket packet;
    while (av_read_frame(formatContext, &packet) >= 0) {
        // Check if the packet belongs to the video stream
        if (packet.stream_index == videoStream->index) {
            // Compute the current packet timestamp in seconds
            int64_t packetTimestamp = av_rescale_q(packet.pts, videoStream->time_base, AV_TIME_BASE_Q);
            int64_t segmentEndTime = segmentStartTime + (segmentDuration * AV_TIME_BASE);

            // Check if the packet belongs to the next segment
            if (packetTimestamp > segmentEndTime) {
                // Close the current segment
                std::cout << "Segment: " << segmentStartTime / AV_TIME_BASE << " - " << segmentEndTime / AV_TIME_BASE
                          << std::endl;

                // Start a new segment
                segmentStartTime = packetTimestamp;
            }
        }

        // Free the packet resources
        av_packet_unref(&packet);
    }

   
