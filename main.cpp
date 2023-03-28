#include <torch/torch.h>
#include <iostream>
#include <typeinfo>
#include "ProgressBar.h"

struct DCGANGeneratorImpl : torch::nn::Module {
    DCGANGeneratorImpl(int kNoiseSize)
            : conv1(torch::nn::ConvTranspose2dOptions(kNoiseSize, 256, 4)
                            .bias(false)),
              batch_norm1(256),
              conv2(torch::nn::ConvTranspose2dOptions(256, 128, 3)
                            .stride(2)
                            .padding(1)
                            .bias(false)),
              batch_norm2(128),
              conv3(torch::nn::ConvTranspose2dOptions(128, 64, 4)
                            .stride(2)
                            .padding(1)
                            .bias(false)),
              batch_norm3(64),
              conv4(torch::nn::ConvTranspose2dOptions(64, 1, 4)
                            .stride(2)
                            .padding(1)
                            .bias(false)) {
        // register_module() is needed if we want to use the parameters() method later on
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("conv3", conv3);
        register_module("conv4", conv4);
        register_module("batch_norm1", batch_norm1);
        register_module("batch_norm2", batch_norm2);
        register_module("batch_norm3", batch_norm3);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(batch_norm1(conv1(x)));
        x = torch::relu(batch_norm2(conv2(x)));
        x = torch::relu(batch_norm3(conv3(x)));
        x = torch::tanh(conv4(x));
        return x;
    }

    torch::nn::ConvTranspose2d conv1, conv2, conv3, conv4;
    torch::nn::BatchNorm2d batch_norm1, batch_norm2, batch_norm3;
};

TORCH_MODULE(DCGANGenerator);

const int kNoiseSize = 100;

DCGANGenerator generator(kNoiseSize);

torch::nn::Sequential discriminator(
        // Layer 1
        torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 64, 4).stride(2).padding(1).bias(false)),
        torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
        // Layer 2
        torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 4).stride(2).padding(1).bias(false)),
        torch::nn::BatchNorm2d(128),
        torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
        // Layer 3
        torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 4).stride(2).padding(1).bias(false)),
        torch::nn::BatchNorm2d(256),
        torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
        // Layer 4
        torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 1, 3).stride(1).padding(0).bias(false)),
        torch::nn::Sigmoid());

int main() {
    int batch_size = 32;
    torch::Device device(torch::kCUDA, 1);
    auto dataset = torch::data::datasets::MNIST("../mnist")
            .map(torch::data::transforms::Normalize<>(0.5, 0.5))
            .map(torch::data::transforms::Stack<>());
    std::cout << dataset.size().value() << std::endl;
    int64_t batches_per_epoch = dataset.size().value() / batch_size;
    discriminator->to(device);
    generator->to(device);
    auto data_loader = torch::data::make_data_loader(std::move(dataset),
                                                     torch::data::DataLoaderOptions().batch_size(batch_size).workers(
                                                             2));
    torch::optim::Adam generator_optimizer(
            generator->parameters(), torch::optim::AdamOptions(2e-4));
    torch::optim::Adam discriminator_optimizer(
            discriminator->parameters(), torch::optim::AdamOptions(5e-4));
    int64_t kNumberOfEpochs = 10;
    for (int64_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {

        int64_t batch_index = 0;
        ProgressBar pbar((int) batches_per_epoch);
        for (torch::data::Example<> &batch: *data_loader) {
            // Train discriminator with real images.
            pbar.set_prefix("Epoch " + std::to_string(epoch));
            discriminator->zero_grad();
            torch::Tensor real_images = batch.data.to(device);
            torch::Tensor real_labels = torch::empty(batch.data.size(0)).uniform_(0.8, 1.0).to(device);
            torch::Tensor real_output = discriminator->forward(real_images).reshape({batch_size});
            torch::Tensor d_loss_real = torch::binary_cross_entropy(real_output, real_labels).to(device);
            d_loss_real.backward();

            // Train discriminator with fake images.
            torch::Tensor noise = torch::randn({batch.data.size(0), kNoiseSize, 1, 1}).to(device);
            torch::Tensor fake_images = generator->forward(noise);
            torch::Tensor fake_labels = torch::zeros(batch.data.size(0)).to(device);
            torch::Tensor fake_output = discriminator->forward(fake_images.detach()).reshape({batch_size});
            torch::Tensor d_loss_fake = torch::binary_cross_entropy(fake_output, fake_labels);
            d_loss_fake.backward();

            torch::Tensor d_loss = d_loss_real + d_loss_fake;
            discriminator_optimizer.step();

            // Train generator.
            generator->zero_grad();
            fake_labels.fill_(1);
            fake_output = discriminator->forward(fake_images).reshape({batch_size});
            torch::Tensor g_loss = torch::binary_cross_entropy(fake_output, fake_labels);
            g_loss.backward();
            generator_optimizer.step();
            pbar.update();
            pbar.set_postfix("D_loss " + std::to_string(d_loss.item<float>()) + "|G_loss " + std::to_string(g_loss.item<float>()));
            /*std::printf(
                    "\r[%2ld/%2ld][%3ld/%3ld] D_loss: %.4f | G_loss: %.4f",
                    epoch,
                    kNumberOfEpochs,
                    ++batch_index,
                    batches_per_epoch,
                    d_loss.item<float>(),
                    g_loss.item<float>());*/
        }
        pbar.close();
    }
    return 0;
}