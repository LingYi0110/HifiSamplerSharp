using HifiSampler.Core.Resampler;
using HifiSampler.Server.Contracts;
using System.Globalization;

namespace HifiSampler.Server;

public sealed class ResamplerService : IDisposable
{
    private readonly Resampler _resampler;
    private readonly SemaphoreSlim _limiter;
    private readonly ILogger<ResamplerService> _logger;

    public ResamplerService(
        IConfiguration configuration,
        ILogger<ResamplerService> logger)
    {
        var config = LoadConfig(configuration);

        _resampler = ResamplerFactory.CreateResampler(config);
        _limiter = new SemaphoreSlim(Math.Max(1, config.MaxWorkers));
        _logger = logger;
        IsReady = true;
    }

    public bool IsReady { get; }

    public async Task<ResampleResult> RenderAsync(
        ResamplerRequestDto request,
        CancellationToken cancellationToken = default)
    {
        _logger.LogInformation(
            "Resample request: input={InputFile}, output={OutputFile}, pitch={PitchMidi}, velocity={Velocity}, flags={Flags}",
            request.InputFile,
            request.OutputFile,
            request.PitchMidi,
            request.Velocity,
            request.Flags);

        await _limiter.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            var startedAt = DateTime.UtcNow;
            var result = await _resampler.RenderAsync(
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

            _logger.LogInformation(
                "Resample finished: status={StatusCode}, elapsedMs={ElapsedMs}, message={Message}",
                result.StatusCode,
                (DateTime.UtcNow - startedAt).TotalMilliseconds,
                result.Message);
            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Resample failed for input={InputFile}", request.InputFile);
            throw;
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

    private static ResamplerConfig LoadConfig(IConfiguration configuration)
    {
        var section = configuration.GetSection(ResamplerConfig.SectionName);
        var vocoderSection = section.GetSection(nameof(ResamplerConfig.VocoderConfig));
        var hnSepSection = section.GetSection(nameof(ResamplerConfig.HnSepConfig));

        return new ResamplerConfig
        {
            Port = GetInt(section, nameof(ResamplerConfig.Port), 8572),
            CachePath = section[nameof(ResamplerConfig.CachePath)],
            MaxWorkers = GetInt(section, nameof(ResamplerConfig.MaxWorkers), 2),
            SampleRate = GetInt(section, nameof(ResamplerConfig.SampleRate), 44100),
            HopSize = GetInt(section, nameof(ResamplerConfig.HopSize), 512),
            OriginHopSize = GetInt(section, nameof(ResamplerConfig.OriginHopSize), 128),
            NFft = GetInt(section, nameof(ResamplerConfig.NFft), 2048),
            WinSize = GetInt(section, nameof(ResamplerConfig.WinSize), 2048),
            NumMels = GetInt(section, nameof(ResamplerConfig.NumMels), 128),
            MelFMin = GetFloat(section, nameof(ResamplerConfig.MelFMin), 40f),
            MelFMax = GetFloat(section, nameof(ResamplerConfig.MelFMax), 16000f),
            Fill = GetInt(section, nameof(ResamplerConfig.Fill), 8),
            PeakLimit = GetDouble(section, nameof(ResamplerConfig.PeakLimit), 0.9),
            WaveNorm = GetBool(section, nameof(ResamplerConfig.WaveNorm), true),
            LoopMode = GetBool(section, nameof(ResamplerConfig.LoopMode), false),
            VocoderConfig = new VocoderConfig
            {
                ModelType = vocoderSection[nameof(VocoderConfig.ModelType)] ?? "onnx",
                ModelPath = vocoderSection[nameof(VocoderConfig.ModelPath)] ?? string.Empty,
                Device = vocoderSection[nameof(VocoderConfig.Device)] ?? "cpu",
                DeviceId = GetInt(vocoderSection, nameof(VocoderConfig.DeviceId), 0),
                NumMels = GetInt(vocoderSection, nameof(VocoderConfig.NumMels), 128)
            },
            HnSepConfig = new HnSepConfig
            {
                ModelType = hnSepSection[nameof(HnSepConfig.ModelType)] ?? "onnx",
                ModelPath = hnSepSection[nameof(HnSepConfig.ModelPath)] ?? string.Empty,
                Device = hnSepSection[nameof(HnSepConfig.Device)] ?? "cpu",
                DeviceId = GetInt(hnSepSection, nameof(HnSepConfig.DeviceId), 0),
                HopLength = GetInt(hnSepSection, nameof(HnSepConfig.HopLength), 512),
                NFft = GetInt(hnSepSection, nameof(HnSepConfig.NFft), 2048)
            }
        };
    }

    private static int GetInt(IConfiguration section, string key, int defaultValue) =>
        int.TryParse(section[key], NumberStyles.Integer, CultureInfo.InvariantCulture, out var value)
            ? value
            : defaultValue;

    private static bool GetBool(IConfiguration section, string key, bool defaultValue) =>
        bool.TryParse(section[key], out var value) ? value : defaultValue;

    private static float GetFloat(IConfiguration section, string key, float defaultValue) =>
        float.TryParse(section[key], NumberStyles.Float, CultureInfo.InvariantCulture, out var value)
            ? value
            : defaultValue;

    private static double GetDouble(IConfiguration section, string key, double defaultValue) =>
        double.TryParse(section[key], NumberStyles.Float, CultureInfo.InvariantCulture, out var value)
            ? value
            : defaultValue;
}
