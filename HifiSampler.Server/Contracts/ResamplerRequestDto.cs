using HifiSampler.Core.Resampler;

namespace HifiSampler.Server.Contracts;

public sealed record ResamplerRequestDto(
    string InputFile,
    string OutputFile,
    int PitchMidi,
    double Velocity,
    ResamplerFlags Flags,
    double Offset,
    int Length,
    double Consonant,
    double Cutoff,
    double Volume,
    double Modulation,
    double Tempo,
    double[] PitchBendCents);
