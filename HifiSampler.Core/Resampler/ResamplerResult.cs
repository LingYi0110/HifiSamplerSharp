namespace HifiSampler.Core.Resampler;

public sealed record ResamplerResult(
    int StatusCode,
    string Message,
    string? Traceback = null);
