using HifiSampler.Core.Resampler;
using HifiSampler.Server.Contracts;

namespace HifiSampler.Server;

public sealed class ResamplerService : IDisposable
{
    private readonly Resampler _resampler;
    private readonly SemaphoreSlim _limiter;

    public ResamplerService(IConfiguration configuration)
    {
        var config = new ResamplerConfig();
        configuration.GetSection(ResamplerConfig.SectionName).Bind(config);

        _resampler = ResamplerFactory.CreateResampler(config);
        _limiter = new SemaphoreSlim(Math.Max(1, config.MaxWorkers));
        IsReady = true;
    }

    public bool IsReady { get; }

    public async Task<ResampleResult> RenderAsync(
        ResamplerRequestDto request,
        CancellationToken cancellationToken = default)
    {
        await _limiter.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            return await _resampler.RenderAsync(
                    request.InputFile,
                    request.OutputFile,
                    request.PitchMidi,
                    request.Velocity,
                    request.Flags,
                    request.Offset,
                    request.Length,
                    request.Consonant,
                    request.Cutoff,
                    request.Volume,
                    request.Modulation,
                    request.Tempo,
                    request.PitchBendCents,
                    cancellationToken)
                .ConfigureAwait(false);
        }
        finally
        {
            _limiter.Release();
        }
    }

    public void Dispose()
    {
        _limiter.Dispose();
    }
}
