using System.Security.Cryptography;
using System.Text;
using TorchSharp;
using static TorchSharp.torch;

namespace HifiSampler.Core.Resampler;

public sealed class ResamplerCacheManager
{
    public bool ShouldBypassCache(ResamplerFlags flags) => flags.G;

    public async Task<(float[,] mel, float scale)?> TryLoadMelAsync(
        string inputFile,
        ResamplerFlags flags,
        CancellationToken cancellationToken = default)
    {
        var melPath = BuildMelCacheFilePath(inputFile, flags);
        var scalePath = BuildScaleCacheFilePath(inputFile, flags);
        if (!File.Exists(melPath) || !File.Exists(scalePath))
        {
            return null;
        }

        try
        {
            await using var _ = File.OpenRead(melPath);
            cancellationToken.ThrowIfCancellationRequested();
            var melTensor = Tensor.Load(melPath);
            var scaleTensor = Tensor.Load(scalePath);
            var mel = TensorTo2D(melTensor);
            var scale = scaleTensor.cpu().data<float>().FirstOrDefault(1f);
            melTensor.Dispose();
            scaleTensor.Dispose();
            return (mel, scale);
        }
        catch
        {
            return null;
        }
    }

    public Task<float[]?> TryLoadHnSepAsync(
        string inputFile,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var hnsepPath = BuildHnSepCacheFilePath(inputFile);
        if (!File.Exists(hnsepPath))
        {
            return Task.FromResult<float[]?>(null);
        }

        try
        {
            var tensor = Tensor.Load(hnsepPath);
            var result = tensor.cpu().data<float>().ToArray();
            tensor.Dispose();
            return Task.FromResult<float[]?>(result);
        }
        catch
        {
            return Task.FromResult<float[]?>(null);
        }
    }

    public Task SaveMelAsync(
        string inputFile,
        ResamplerFlags flags,
        (float[,] mel, float scale) feature,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var melPath = BuildMelCacheFilePath(inputFile, flags);
        var scalePath = BuildScaleCacheFilePath(inputFile, flags);
        EnsureDirectory(melPath);
        EnsureDirectory(scalePath);

        using var melTensor = TensorFrom2D(feature.mel);
        using var scaleTensor = tensor(new[] { feature.scale }, dtype: ScalarType.Float32);
        melTensor.save(melPath);
        scaleTensor.save(scalePath);
        return Task.CompletedTask;
    }

    public Task SaveHnSepAsync(
        string inputFile,
        float[] hnsep,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var hnsepPath = BuildHnSepCacheFilePath(inputFile);
        EnsureDirectory(hnsepPath);
        using var tensor = torch.tensor(hnsep, dtype: ScalarType.Float32);
        tensor.save(hnsepPath);
        return Task.CompletedTask;
    }

    private string BuildMelCacheFilePath(string inputFile, ResamplerFlags flags)
    {
        var cacheDirectory = ResolveCacheDirectory(inputFile);
        var fileStem = Path.GetFileNameWithoutExtension(inputFile);
        var signature = BuildFeatureSignature(flags);
        return Path.Combine(cacheDirectory, $"{fileStem}_{signature}.mel.pt");
    }

    private string BuildScaleCacheFilePath(string inputFile, ResamplerFlags flags)
    {
        var cacheDirectory = ResolveCacheDirectory(inputFile);
        var fileStem = Path.GetFileNameWithoutExtension(inputFile);
        var signature = BuildFeatureSignature(flags);
        return Path.Combine(cacheDirectory, $"{fileStem}_{signature}.scale.pt");
    }

    private string BuildHnSepCacheFilePath(string inputFile)
    {
        var cacheDirectory = ResolveCacheDirectory(inputFile);
        var fileStem = Path.GetFileNameWithoutExtension(inputFile);
        return Path.Combine(cacheDirectory, $"{fileStem}.hnsep.pt");
    }

    private static string ResolveCacheDirectory(string inputFile)
    {
        var inputDirectory = Path.GetDirectoryName(inputFile);
        return string.IsNullOrWhiteSpace(inputDirectory)
            ? Directory.GetCurrentDirectory()
            : inputDirectory;
    }

    private static void EnsureDirectory(string path)
    {
        var dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrWhiteSpace(dir))
        {
            Directory.CreateDirectory(dir);
        }
    }

    private static string BuildFeatureSignature(ResamplerFlags flags)
    {
        var key = $"g={flags.g};Hb={flags.Hb};Hv={flags.Hv};Ht={flags.Ht}";
        var hash = SHA256.HashData(Encoding.UTF8.GetBytes(key));
        return Convert.ToHexString(hash.AsSpan(0, 6)).ToLowerInvariant();
    }

    private static Tensor TensorFrom2D(float[,] source)
    {
        var rows = source.GetLength(0);
        var cols = source.GetLength(1);
        var flat = new float[rows * cols];
        var idx = 0;
        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < cols; c++)
            {
                flat[idx++] = source[r, c];
            }
        }

        return tensor(flat, dtype: ScalarType.Float32).reshape(rows, cols);
    }

    private static float[,] TensorTo2D(Tensor tensor2d)
    {
        var rows = (int)tensor2d.shape[0];
        var cols = (int)tensor2d.shape[1];
        var flat = tensor2d.contiguous().cpu().data<float>().ToArray();
        var result = new float[rows, cols];
        var idx = 0;
        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < cols; c++)
            {
                result[r, c] = flat[idx++];
            }
        }

        return result;
    }
}
