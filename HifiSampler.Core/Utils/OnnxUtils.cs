using Microsoft.ML.OnnxRuntime;

namespace HifiSampler.Core.Utils;

public static class OnnxUtils
{
    public static InferenceSession CreateSession(string modelPath, string device, int deviceId = 0)
    {
        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException($"ONNX model file not found: {modelPath}", modelPath);
        }

        if (String.IsNullOrWhiteSpace(device))
        {
            throw new ArgumentException("Onnx device cannot be null or empty.");
        }

        var sessionOptions = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL
        };

        switch (device.Trim().ToLowerInvariant())
        {
            case "cpu":
                // 默认支持 CPU
                break;
            case "cuda":
                sessionOptions.AppendExecutionProvider_CUDA(deviceId);
                break;
            case "directml":
            case "dml":
                sessionOptions.AppendExecutionProvider_DML(deviceId);
                break;
            default:
                throw new ArgumentException($"Unknown device: {device}.");
        }

        return new InferenceSession(modelPath, sessionOptions);
    }
}