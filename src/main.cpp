#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <torch/script.h>
#include <dlfcn.h>

int main(int argc, const char *argv[])
{
    if (argc != 3)
    {
        std::cout << argv[0] << "image.jpg end_to_end_model.pt" << std::endl;
        return 1;
    }

    void *custom_op_lib = dlopen(argv[3], RTLD_NOW | RTLD_GLOBAL);
    if (custom_op_lib == NULL)
    {
        std::cerr << "could not open custom op library: " << dlerror() << std::endl;
        return 1;
    }

    auto img_ = cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::Mat img(480, 640, CV_8UC3);
    cv::resize(img_, img, img.size(), 0, 0, cv::INTER_AREA);
    auto input_ = torch::tensor(at::ArrayRef<uint8_t>(img.data, img.rows * img.cols * 3)).view({img.rows, img.cols, 3});

    torch::NoGradGuard no_grad;
    torch::jit::script::Module module;
    try
    {
        module = torch::jit::load(argv[2]);
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_);

        at::Tensor output = module.forward(inputs).toTensor();
        std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
    }
    catch (const c10::Error &e)
    {
        std::cerr << "error loading the model\n";
        return -1;
    }

    std::cout << "ok\n";
}