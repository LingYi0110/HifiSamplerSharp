namespace HifiSampler.Core.Resampler;

public sealed record ResampleResult(
    int StatusCode,
    string Message,
    string? Traceback = null);
