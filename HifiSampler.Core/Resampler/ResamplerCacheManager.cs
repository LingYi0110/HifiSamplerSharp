using System.Security.Cryptography;
using System.Text;
using System.Runtime.InteropServices;

namespace HifiSampler.Core.Resampler;

public sealed class ResamplerCacheManager
{
    private const uint MelMagic = 0x314C454D; // MEL1
    private const uint ScaleMagic = 0x314C4353; // SCL1
    private const uint HnSepMagic = 0x31504E48; // HNP1

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
            cancellationToken.ThrowIfCancellationRequested();
            var mel = ReadMel(melPath);
            var scale = ReadScale(scalePath);
            if (mel is null || !scale.HasValue)
            {
                return null;
            }

            return (mel, scale.Value);
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
            return Task.FromResult<float[]?>(ReadHnSep(hnsepPath));
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
        WriteMel(melPath, feature.mel);
        WriteScale(scalePath, feature.scale);
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
        WriteHnSep(hnsepPath, hnsep);
        return Task.CompletedTask;
    }

    private string BuildMelCacheFilePath(string inputFile, ResamplerFlags flags)
    {
        var cacheDirectory = ResolveCacheDirectory(inputFile);
        var fileStem = Path.GetFileNameWithoutExtension(inputFile);
        var signature = BuildFeatureSignature(flags);
        return Path.Combine(cacheDirectory, $"{fileStem}_{signature}.mel.bin");
    }

    private string BuildScaleCacheFilePath(string inputFile, ResamplerFlags flags)
    {
        var cacheDirectory = ResolveCacheDirectory(inputFile);
        var fileStem = Path.GetFileNameWithoutExtension(inputFile);
        var signature = BuildFeatureSignature(flags);
        return Path.Combine(cacheDirectory, $"{fileStem}_{signature}.scale.bin");
    }

    private string BuildHnSepCacheFilePath(string inputFile)
    {
        var cacheDirectory = ResolveCacheDirectory(inputFile);
        var fileStem = Path.GetFileNameWithoutExtension(inputFile);
        return Path.Combine(cacheDirectory, $"{fileStem}.hnsep.bin");
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

    private static float[,]? ReadMel(string path)
    {
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 4096, useAsync: false);
        using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);
        if (reader.ReadUInt32() != MelMagic)
        {
            return null;
        }

        var rows = reader.ReadInt32();
        var cols = reader.ReadInt32();
        if (rows <= 0 || cols <= 0)
        {
            return null;
        }

        var flat = new float[rows * cols];
        stream.ReadExactly(MemoryMarshal.AsBytes(flat.AsSpan()));
        var mel = new float[rows, cols];
        var offset = 0;
        for (var row = 0; row < rows; row++)
        {
            for (var col = 0; col < cols; col++)
            {
                mel[row, col] = flat[offset++];
            }
        }

        return mel;
    }

    private static float? ReadScale(string path)
    {
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 4096, useAsync: false);
        using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);
        if (reader.ReadUInt32() != ScaleMagic)
        {
            return null;
        }

        return reader.ReadSingle();
    }

    private static float[]? ReadHnSep(string path)
    {
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 4096, useAsync: false);
        using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);
        if (reader.ReadUInt32() != HnSepMagic)
        {
            return null;
        }

        var length = reader.ReadInt32();
        if (length <= 0)
        {
            return [];
        }

        var result = new float[length];
        stream.ReadExactly(MemoryMarshal.AsBytes(result.AsSpan()));
        return result;
    }

    private static void WriteMel(string path, float[,] mel)
    {
        var rows = mel.GetLength(0);
        var cols = mel.GetLength(1);
        var flat = new float[rows * cols];
        var offset = 0;
        for (var row = 0; row < rows; row++)
        {
            for (var col = 0; col < cols; col++)
            {
                flat[offset++] = mel[row, col];
            }
        }

        using var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None, 4096, useAsync: false);
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);
        writer.Write(MelMagic);
        writer.Write(rows);
        writer.Write(cols);
        writer.Flush();
        stream.Write(MemoryMarshal.AsBytes(flat.AsSpan()));
    }

    private static void WriteScale(string path, float scale)
    {
        using var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None, 4096, useAsync: false);
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);
        writer.Write(ScaleMagic);
        writer.Write(scale);
        writer.Flush();
    }

    private static void WriteHnSep(string path, ReadOnlySpan<float> values)
    {
        using var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None, 4096, useAsync: false);
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);
        writer.Write(HnSepMagic);
        writer.Write(values.Length);
        writer.Flush();
        stream.Write(MemoryMarshal.AsBytes(values));
    }
}
