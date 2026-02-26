using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Runtime.InteropServices;
using System.Text;

namespace HifiSampler.Core.Resampler;

public sealed class ResamplerCacheManager
{
    private const int IoBufferSize = 64 * 1024;
    private const int MelHeaderBytes = sizeof(int) + sizeof(int) + sizeof(float);
    private const int HnSepHeaderBytes = sizeof(int);

    public bool ShouldBypassCache(ResamplerFlags flags) => flags.G;

    public Task<(float[,] mel, float scale)?> TryLoadMelAsync(
        string inputFile,
        ResamplerFlags flags,
        CancellationToken cancellationToken = default)
    {
        var melPath = BuildMelCacheFilePath(inputFile, flags);
        if (!File.Exists(melPath))
        {
            return Task.FromResult<(float[,] mel, float scale)?>(null);
        }

        try
        {
            cancellationToken.ThrowIfCancellationRequested();
            return Task.FromResult(ReadMel(melPath));
        }
        catch
        {
            return Task.FromResult<(float[,] mel, float scale)?>(null);
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
            return Task.FromResult(ReadHnSep(hnsepPath));
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
        EnsureDirectory(melPath);
        WriteMel(melPath, feature.mel, feature.scale);
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
        return Path.Combine(cacheDirectory, $"{fileStem}_{signature}.mel.cache");
    }

    private string BuildHnSepCacheFilePath(string inputFile)
    {
        var cacheDirectory = ResolveCacheDirectory(inputFile);
        var fileStem = Path.GetFileNameWithoutExtension(inputFile);
        return Path.Combine(cacheDirectory, $"{fileStem}.hnsep.cache");
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

    private static (float[,] mel, float scale)? ReadMel(string path)
    {
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, IoBufferSize, FileOptions.SequentialScan);
        if (stream.Length < MelHeaderBytes)
        {
            return null;
        }

        Span<byte> header = stackalloc byte[MelHeaderBytes];
        stream.ReadExactly(header);

        var rows = BinaryPrimitives.ReadInt32LittleEndian(header[..sizeof(int)]);
        var cols = BinaryPrimitives.ReadInt32LittleEndian(header.Slice(sizeof(int), sizeof(int)));
        if (rows <= 0 || cols <= 0)
        {
            return null;
        }

        if (!TryGetElementCount(rows, cols, out var elementCount))
        {
            return null;
        }

        var expectedLength = MelHeaderBytes + (long)elementCount * sizeof(float);
        if (stream.Length != expectedLength)
        {
            return null;
        }

        var scaleBits = BinaryPrimitives.ReadInt32LittleEndian(header.Slice(sizeof(int) * 2, sizeof(float)));
        var scale = BitConverter.Int32BitsToSingle(scaleBits);
        var mel = new float[rows, cols];
        stream.ReadExactly(MemoryMarshal.AsBytes(GetMatrixSpan(mel)));

        return (mel, scale);
    }

    private static float[]? ReadHnSep(string path)
    {
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, IoBufferSize, FileOptions.SequentialScan);
        if (stream.Length < HnSepHeaderBytes)
        {
            return null;
        }

        Span<byte> header = stackalloc byte[HnSepHeaderBytes];
        stream.ReadExactly(header);

        var length = BinaryPrimitives.ReadInt32LittleEndian(header);
        if (length <= 0)
        {
            return [];
        }

        var expectedLength = HnSepHeaderBytes + (long)length * sizeof(float);
        if (stream.Length != expectedLength)
        {
            return null;
        }

        var result = new float[length];
        stream.ReadExactly(MemoryMarshal.AsBytes(result.AsSpan()));
        return result;
    }

    private static void WriteMel(string path, float[,] mel, float scale)
    {
        var rows = mel.GetLength(0);
        var cols = mel.GetLength(1);
        if (!TryGetElementCount(rows, cols, out _))
        {
            throw new InvalidDataException("Mel shape is invalid for cache serialization.");
        }

        var header = new byte[MelHeaderBytes];
        BinaryPrimitives.WriteInt32LittleEndian(header.AsSpan(0, sizeof(int)), rows);
        BinaryPrimitives.WriteInt32LittleEndian(header.AsSpan(sizeof(int), sizeof(int)), cols);
        BinaryPrimitives.WriteInt32LittleEndian(
            header.AsSpan(sizeof(int) * 2, sizeof(float)),
            BitConverter.SingleToInt32Bits(scale));

        WriteAtomically(path, stream =>
        {
            stream.Write(header);
            stream.Write(MemoryMarshal.AsBytes(GetMatrixSpan(mel)));
        });
    }

    private static void WriteHnSep(string path, float[] values)
    {
        var header = new byte[HnSepHeaderBytes];
        BinaryPrimitives.WriteInt32LittleEndian(header, values.Length);

        WriteAtomically(path, stream =>
        {
            stream.Write(header);
            stream.Write(MemoryMarshal.AsBytes(values.AsSpan()));
        });
    }

    private static bool TryGetElementCount(int rows, int cols, out int count)
    {
        var total = (long)rows * cols;
        if (total <= 0 || total > int.MaxValue)
        {
            count = 0;
            return false;
        }

        count = (int)total;
        return true;
    }

    private static Span<float> GetMatrixSpan(float[,] matrix)
    {
        if (matrix.Length == 0)
        {
            return [];
        }

        return MemoryMarshal.CreateSpan(ref matrix[0, 0], matrix.Length);
    }

    private static void WriteAtomically(string path, Action<FileStream> write)
    {
        var directory = Path.GetDirectoryName(path);
        if (string.IsNullOrWhiteSpace(directory))
        {
            directory = Directory.GetCurrentDirectory();
        }

        var tempPath = Path.Combine(directory, $".{Path.GetFileName(path)}.{Guid.NewGuid():N}.tmp");
        try
        {
            using (var stream = new FileStream(tempPath, FileMode.CreateNew, FileAccess.Write, FileShare.None, IoBufferSize, FileOptions.None))
            {
                write(stream);
                stream.Flush(flushToDisk: true);
            }

            File.Move(tempPath, path, overwrite: true);
        }
        finally
        {
            try
            {
                if (File.Exists(tempPath))
                {
                    File.Delete(tempPath);
                }
            }
            catch
            {
                // Best effort cleanup.
            }
        }
    }
}
