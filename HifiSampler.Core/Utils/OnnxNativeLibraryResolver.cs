using System.Reflection;
using System.Runtime.InteropServices;
using System.Threading;
using Microsoft.ML.OnnxRuntime;

namespace HifiSampler.Core.Utils;

// Due to https://github.com/microsoft/onnxruntime/pull/27266,
// wait for an official release with the fix before removing this workaround.
internal static class OnnxNativeLibraryResolver
{
    private static int _resolverConfigured;

    public static void EnsureConfigured()
    {
        if (Interlocked.Exchange(ref _resolverConfigured, 1) == 1)
        {
            return;
        }

        try
        {
            NativeLibrary.SetDllImportResolver(typeof(InferenceSession).Assembly, Resolve);
        }
        catch (InvalidOperationException)
        {
            // Resolver may already be configured by another component.
        }
    }

    private static IntPtr Resolve(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (!libraryName.Equals("onnxruntime.dll", StringComparison.OrdinalIgnoreCase))
        {
            return IntPtr.Zero;
        }

        if (OperatingSystem.IsLinux())
        {
            if (TryLoadInKnownLocations("libonnxruntime.so", out var handle))
            {
                return handle;
            }

            if (NativeLibrary.TryLoad("libonnxruntime.so", assembly, searchPath, out handle))
            {
                return handle;
            }
        }

        if (OperatingSystem.IsMacOS())
        {
            if (TryLoadInKnownLocations("libonnxruntime.dylib", out var handle))
            {
                return handle;
            }

            if (NativeLibrary.TryLoad("libonnxruntime.dylib", assembly, searchPath, out handle))
            {
                return handle;
            }
        }

        return IntPtr.Zero;
    }

    private static bool TryLoadInKnownLocations(string fileName, out IntPtr handle)
    {
        handle = IntPtr.Zero;
        var baseDir = AppContext.BaseDirectory;
        var rid = GetCurrentRuntimeIdentifier();
        if (String.IsNullOrEmpty(rid))
        {
            return false;
        }

        var runtimeNativeDir = Path.Combine(baseDir, "runtimes", rid, "native");
        var candidates = new[]
        {
            Path.Combine(baseDir, fileName),
            Path.Combine(runtimeNativeDir, fileName)
        };

        foreach (var candidate in candidates)
        {
            if (!File.Exists(candidate))
            {
                continue;
            }

            if (NativeLibrary.TryLoad(candidate, out handle))
            {
                return true;
            }
        }

        return false;
    }

    private static string GetCurrentRuntimeIdentifier()
    {
        var architecture = RuntimeInformation.ProcessArchitecture;

        if (OperatingSystem.IsLinux())
        {
            return architecture switch
            {
                Architecture.X64 => "linux-x64",
                Architecture.Arm64 => "linux-arm64",
                _ => String.Empty
            };
        }

        if (OperatingSystem.IsMacOS())
        {
            return architecture switch
            {
                Architecture.X64 => "osx-x64",
                Architecture.Arm64 => "osx-arm64",
                _ => String.Empty
            };
        }

        return String.Empty;
    }
}
