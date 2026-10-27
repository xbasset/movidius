// Host transport only: every graph is loaded exclusively on MYRIAD.
#include <inference_engine.hpp>
#include <ie_core.hpp>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <cstdlib>

namespace ie = InferenceEngine;
struct Context {
    explicit Context(const std::string& registry) : core(registry) {}
    ie::Core core;
    ie::CNNNetwork network;
    ie::ExecutableNetwork executable;
    ie::InferRequest request;
    std::string output;
};

static void error_message(char* error, size_t size, const std::exception& exception) {
    std::snprintf(error, size, "%s", exception.what());
}

extern "C" void* crivo_open(const char* registry, const char* xml, char* error, size_t size) {
    try {
        if (std::getenv("MOVI_DEBUG")) std::fprintf(stderr, "OpenVINO Core\n");
        std::unique_ptr<Context> c(new Context(registry));
        if (std::getenv("MOVI_DEBUG")) std::fprintf(stderr, "ReadNetwork\n");
        c->network = c->core.ReadNetwork(xml);
        for (auto& input : c->network.getInputsInfo()) {
            input.second->setPrecision(input.first == "input_ids" || input.first == "opt_positions"
                                       ? ie::Precision::I32 : ie::Precision::FP16);
        }
        for (auto& output : c->network.getOutputsInfo()) {
            output.second->setPrecision(ie::Precision::FP32);
            c->output = output.first;
        }
        if (std::getenv("MOVI_DEBUG")) std::fprintf(stderr, "LoadNetwork MYRIAD\n");
        c->executable = c->core.LoadNetwork(c->network, "MYRIAD", {{"PERF_COUNT", "YES"}});
        c->request = c->executable.CreateInferRequest();
        return c.release();
    } catch (const std::exception& e) {
        error_message(error, size, e);
        return nullptr;
    }
}

extern "C" int crivo_input(void* context, const char* name, const void* data, size_t bytes,
                            char* error, size_t size) {
    try {
        auto* c = static_cast<Context*>(context);
        auto blob = c->request.GetBlob(name);
        if (blob->byteSize() != bytes) throw std::runtime_error(std::string(name) + ": input byte count " +
            std::to_string(bytes) + " differs from graph's " + std::to_string(blob->byteSize()));
        std::memcpy(blob->buffer().as<void*>(), data, bytes);
        return 0;
    } catch (const std::exception& e) {
        if (std::getenv("MOVI_DEBUG")) std::fprintf(stderr, "crivo_input: %s\n", e.what());
        error_message(error, size, e);
        return -1;
    }
}

extern "C" int crivo_infer(void* context, float* output, size_t elements, double* ms,
                            char* error, size_t size) {
    try {
        auto* c = static_cast<Context*>(context);
        auto start = std::chrono::steady_clock::now();
        c->request.Infer();
        *ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        auto blob = c->request.GetBlob(c->output);
        if (blob->size() != elements) throw std::runtime_error("Output element count differs from graph");
        std::memcpy(output, blob->cbuffer().as<const void*>(), elements * sizeof(float));
        return 0;
    } catch (const std::exception& e) {
        error_message(error, size, e);
        return -1;
    }
}

extern "C" int crivo_profile(void* context, const char* path, char* error, size_t size) {
    try {
        auto* c = static_cast<Context*>(context);
        std::ofstream file(path);
        file << "layer\texecution_type\tstatus\treal_us\tcpu_us\n";
        for (const auto& stage : c->request.GetPerformanceCounts()) {
            file << stage.first << '\t' << stage.second.exec_type << '\t' << stage.second.status
                 << '\t' << stage.second.realTime_uSec << '\t' << stage.second.cpu_uSec << '\n';
        }
        return 0;
    } catch (const std::exception& e) {
        error_message(error, size, e);
        return -1;
    }
}

extern "C" void crivo_close(void* context) {
    delete static_cast<Context*>(context);
}
