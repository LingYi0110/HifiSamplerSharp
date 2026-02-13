using System.Globalization;
using System.Net;
using System.Net.Http.Json;
using System.Net.Sockets;
using System.Text.RegularExpressions;
using System.Diagnostics;

namespace HifiSampler.Client;

internal static partial class Program
{
    private static readonly HttpClient PostClient = new() { Timeout = TimeSpan.FromSeconds(180) };
    private static readonly Func<HttpClient> CheckClientFactory = () => new HttpClient { Timeout = TimeSpan.FromSeconds(2) };
    private const int TargetPort = 8572;
    private static readonly string LauncherScriptName = OperatingSystem.IsWindows() ? "start.bat" : "start.sh";
    private const string ServerStartupMutexName = "Global\\HifiSamplerServerStartupMutex_DCL_8572";

    private static async Task Main(string[] args)
    {
        if (args.Length < 1)
        {
            Environment.Exit(1);
            return;
        }

        if (!TryBuildRequest(args, out var request, out var parseError) || request is null)
        {
            Console.Error.WriteLine(parseError);
            Environment.Exit(1);
            return;
        }

        var proceedToCommunication = false;
        var errorOccurred = false;
        var errorMessage = string.Empty;

        if (await IsServerReady(TargetPort))
        {
            proceedToCommunication = true;
        }
        else
        {
            using var mutex = new Mutex(false, ServerStartupMutexName);
            var mutexAcquired = false;
            var mutexOwned = false;
            try
            {
                try
                {
                    mutexAcquired = mutex.WaitOne(TimeSpan.FromSeconds(5));
                    mutexOwned = mutexAcquired;
                }
                catch (AbandonedMutexException)
                {
                    mutexAcquired = true;
                    mutexOwned = true;
                }

                if (mutexAcquired)
                {
                    if (await IsServerReady(TargetPort))
                    {
                        proceedToCommunication = true;
                    }
                    else if (!IsPortInUse(TargetPort))
                    {
                        var exeDir = AppContext.BaseDirectory;
                        var scriptPath = Path.Combine(exeDir, LauncherScriptName);
                        if (!File.Exists(scriptPath))
                        {
                            errorMessage = $"Error: {LauncherScriptName} not found at '{scriptPath}'.";
                            errorOccurred = true;
                        }
                        else
                        {
                            LaunchServerLauncher(scriptPath);
                            if (await WaitForServerToStart(TargetPort, 60))
                            {
                                proceedToCommunication = true;
                            }
                            else
                            {
                                errorMessage = "Server launched by this instance did not become ready.";
                                errorOccurred = true;
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                errorMessage = ex.Message;
                errorOccurred = true;
            }
            finally
            {
                if (mutexOwned)
                {
                    try
                    {
                        mutex.ReleaseMutex();
                    }
                    catch
                    {
                        // no-op
                    }
                }
            }
        }

        if (errorOccurred)
        {
            Console.Error.WriteLine(errorMessage);
            Environment.Exit(1);
            return;
        }

        if (!proceedToCommunication)
        {
            proceedToCommunication = await WaitForServerToStart(TargetPort, 95);
        }

        if (!proceedToCommunication)
        {
            Console.Error.WriteLine("Server failed readiness check.");
            Environment.Exit(1);
            return;
        }

        await CommunicateWithServer(request);
    }

    private static async Task CommunicateWithServer(ResamplerRequest request)
    {
        const int maxRetries = 2;
        const int retryDelayMs = 800;

        for (var attempt = 1; attempt <= maxRetries + 1; attempt++)
        {
            try
            {
                var response = await PostClient.PostAsJsonAsync(
                    $"http://127.0.0.1:{TargetPort}/",
                    request,
                    ClientJsonContext.Default.ResamplerRequest);
                var body = response.Content is null ? string.Empty : await response.Content.ReadAsStringAsync();
                if (response.IsSuccessStatusCode)
                {
                    Console.WriteLine(body);
                    return;
                }

                if ((response.StatusCode == HttpStatusCode.ServiceUnavailable ||
                     response.StatusCode == HttpStatusCode.InternalServerError) &&
                    attempt <= maxRetries)
                {
                    await Task.Delay(retryDelayMs);
                    continue;
                }

                Console.Error.WriteLine(body);
                Environment.Exit(1);
                return;
            }
            catch (Exception ex)
            {
                if (attempt <= maxRetries)
                {
                    await Task.Delay(retryDelayMs);
                    continue;
                }

                Console.Error.WriteLine(ex.Message);
                Environment.Exit(1);
                return;
            }
        }
    }

    private static async Task<bool> IsServerReady(int port)
    {
        HttpClient? checkClient = null;
        try
        {
            checkClient = CheckClientFactory();
            var response = await checkClient.GetAsync($"http://127.0.0.1:{port}/");
            return response.StatusCode == HttpStatusCode.OK;
        }
        catch
        {
            return false;
        }
        finally
        {
            checkClient?.Dispose();
        }
    }

    private static async Task<bool> WaitForServerToStart(int port, int timeoutSeconds)
    {
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(timeoutSeconds));
        while (!cts.IsCancellationRequested)
        {
            if (await IsServerReady(port))
            {
                return true;
            }

            try
            {
                await Task.Delay(TimeSpan.FromMilliseconds(500), cts.Token);
            }
            catch (TaskCanceledException)
            {
                break;
            }
        }

        return false;
    }

    private static bool IsPortInUse(int port)
    {
        const int timeoutMs = 150;
        Socket? socket = null;
        try
        {
            socket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
            var result = socket.BeginConnect("127.0.0.1", port, null, null);
            var success = result.AsyncWaitHandle.WaitOne(timeoutMs);
            if (!success)
            {
                return false;
            }

            socket.EndConnect(result);
            return true;
        }
        catch
        {
            return false;
        }
        finally
        {
            socket?.Close();
            socket?.Dispose();
        }
    }

    private static void LaunchServerLauncher(string launcherScriptPath)
    {
        var workingDir = Path.GetDirectoryName(launcherScriptPath) ?? ".";
        if (OperatingSystem.IsWindows())
        {
            var arguments =
                $"/C start \"HifiSampler Server\" /D \"{workingDir}\" cmd /C \"title HifiSampler Server & \"{launcherScriptPath}\"\"";
            Process.Start(new ProcessStartInfo
            {
                FileName = "cmd.exe",
                Arguments = arguments,
                WorkingDirectory = workingDir,
                UseShellExecute = false,
                CreateNoWindow = true
            });
            return;
        }

        var scriptCmd = $"\\\"{launcherScriptPath}\\\"";
        var bashCmd = $"bash -c '{scriptCmd}'";
        var terminals = new[]
        {
            new { Name = "gnome-terminal", Arg = "--", Cmd = bashCmd },
            new { Name = "konsole", Arg = "-e", Cmd = bashCmd },
            new { Name = "xfce4-terminal", Arg = "--command", Cmd = bashCmd },
            new { Name = "mate-terminal", Arg = "-e", Cmd = bashCmd },
            new { Name = "terminator", Arg = "-e", Cmd = bashCmd },
            new { Name = "lxterminal", Arg = "-e", Cmd = bashCmd },
            new { Name = "xterm", Arg = "-e", Cmd = bashCmd }
        };

        foreach (var term in terminals)
        {
            try
            {
                using var check = Process.Start(new ProcessStartInfo("which", term.Name)
                {
                    RedirectStandardOutput = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                });
                check?.WaitForExit(500);
                if (check?.ExitCode != 0)
                {
                    continue;
                }

                Process.Start(new ProcessStartInfo
                {
                    FileName = term.Name,
                    Arguments = $"{term.Arg} {term.Cmd}",
                    WorkingDirectory = workingDir,
                    UseShellExecute = true,
                    CreateNoWindow = false
                });
                return;
            }
            catch
            {
                // try next terminal
            }
        }

        throw new NotSupportedException("Could not find supported graphical terminal.");
    }

    private static bool TryBuildRequest(string[] args, out ResamplerRequest? request, out string error)
    {
        request = null;
        error = string.Empty;

        var raw = string.Join(" ", args);
        if (!TrySplitUtauArguments(raw, out var tokens, out error))
        {
            return false;
        }

        if (!double.TryParse(tokens[3], NumberStyles.Float, CultureInfo.InvariantCulture, out var velocity) ||
            !double.TryParse(tokens[5], NumberStyles.Float, CultureInfo.InvariantCulture, out var offset) ||
            !int.TryParse(tokens[6], NumberStyles.Integer, CultureInfo.InvariantCulture, out var length) ||
            !double.TryParse(tokens[7], NumberStyles.Float, CultureInfo.InvariantCulture, out var consonant) ||
            !double.TryParse(tokens[8], NumberStyles.Float, CultureInfo.InvariantCulture, out var cutoff) ||
            !double.TryParse(tokens[9], NumberStyles.Float, CultureInfo.InvariantCulture, out var volume) ||
            !double.TryParse(tokens[10], NumberStyles.Float, CultureInfo.InvariantCulture, out var modulation))
        {
            error = "Invalid numeric parameter.";
            return false;
        }

        if (tokens[11].Length < 2 || tokens[11][0] != '!' ||
            !double.TryParse(tokens[11][1..], NumberStyles.Float, CultureInfo.InvariantCulture, out var tempo))
        {
            error = "Invalid tempo parameter.";
            return false;
        }

        try
        {
            var pitchMidi = NoteToMidi(tokens[2]);
            var pitchBendCents = ParsePitchStringToCents(tokens[12]);
            var flags = ParseFlags(tokens[4]);
            request = new ResamplerRequest(
                tokens[0],
                tokens[1],
                pitchMidi,
                velocity,
                flags,
                offset,
                length,
                consonant,
                cutoff,
                volume,
                modulation,
                tempo,
                pitchBendCents);
            return true;
        }
        catch (Exception ex)
        {
            error = ex.Message;
            return false;
        }
    }

    private static bool TrySplitUtauArguments(string raw, out string[] tokens, out string error)
    {
        tokens = [];
        error = string.Empty;
        var parts = raw.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (parts.Length < 12)
        {
            error = $"Expected at least 12 parameters, got {parts.Length}.";
            return false;
        }

        if (parts.Length == 12)
        {
            tokens =
            [
                parts[0], parts[1], parts[2], parts[3], string.Empty,
                parts[4], parts[5], parts[6], parts[7], parts[8], parts[9], parts[10], parts[11]
            ];
            return true;
        }

        var otherArgs = parts.Skip(parts.Length - 11).ToArray();
        var filePathTokens = parts.Take(parts.Length - 11).ToArray();
        var filePathString = string.Join(" ", filePathTokens);
        var idx = filePathString.IndexOf(".wav ", StringComparison.OrdinalIgnoreCase);
        if (idx < 0)
        {
            error = "Cannot split input/output path segment.";
            return false;
        }

        var inFile = filePathString[..(idx + 4)];
        var outFile = filePathString[(idx + 5)..];
        tokens = [inFile, outFile, .. otherArgs];
        if (tokens.Length != 13)
        {
            error = $"Expected 13 parameters after split, got {tokens.Length}.";
            return false;
        }

        return true;
    }

    private static int NoteToMidi(string note)
    {
        var map = new Dictionary<char, int>
        {
            ['C'] = 0, ['D'] = 2, ['E'] = 4, ['F'] = 5, ['G'] = 7, ['A'] = 9, ['B'] = 11
        };

        var normalized = note.Trim().ToUpperInvariant();
        if (normalized.Length < 2 || !map.TryGetValue(normalized[0], out var semitone))
        {
            throw new ArgumentException("Invalid note name.");
        }

        var cursor = 1;
        if (cursor < normalized.Length && (normalized[cursor] == '#' || normalized[cursor] == 'B'))
        {
            semitone += normalized[cursor] == '#' ? 1 : -1;
            cursor++;
        }

        if (!int.TryParse(normalized[cursor..], NumberStyles.Integer, CultureInfo.InvariantCulture, out var octave))
        {
            throw new ArgumentException("Invalid note octave.");
        }

        return (octave + 1) * 12 + semitone;
    }

    private static ResamplerFlags ParseFlags(string raw)
    {
        if (string.IsNullOrWhiteSpace(raw) || raw == "0")
        {
            return new ResamplerFlags();
        }

        int? g = null, hb = null, hv = null, ht = null, t = null, a = null, hg = null, p = null;
        var he = false;
        var forceGenerate = false;

        foreach (Match match in FlagRegex().Matches(raw.Replace("/", string.Empty)))
        {
            var key = match.Groups[1].Value;
            var hasValue = !string.IsNullOrEmpty(match.Groups[2].Value);
            var value = hasValue ? int.Parse(match.Groups[2].Value, CultureInfo.InvariantCulture) : 0;

            switch (key)
            {
                case "g" when g is null:
                    g = value;
                    break;
                case "Hb" when hb is null:
                    hb = value;
                    break;
                case "Hv" when hv is null:
                    hv = value;
                    break;
                case "Ht" when ht is null:
                    ht = value;
                    break;
                case "t" when t is null:
                    t = value;
                    break;
                case "A" when a is null:
                    a = value;
                    break;
                case "HG" when hg is null:
                    hg = value;
                    break;
                case "P" when p is null:
                    p = value;
                    break;
                case "He":
                    he = true;
                    break;
                case "G":
                    forceGenerate = true;
                    break;
            }
        }

        return new ResamplerFlags
        {
            g = g ?? 0,
            Hb = hb ?? 100,
            Hv = hv ?? 100,
            Ht = ht ?? 0,
            t = t ?? 0,
            A = a ?? 0,
            HG = hg ?? 0,
            P = p ?? 100,
            He = he,
            G = forceGenerate
        };
    }

    private static double[] ParsePitchStringToCents(string pitchString)
    {
        if (string.IsNullOrWhiteSpace(pitchString))
        {
            return [0];
        }

        var pitch = pitchString.Split('#');
        var result = new List<int>();
        for (var i = 0; i < pitch.Length; i += 2)
        {
            var chunk = pitch[i];
            var values = ToInt12Stream(chunk);
            result.AddRange(values);

            if (i + 1 < pitch.Length && int.TryParse(pitch[i + 1], NumberStyles.Integer, CultureInfo.InvariantCulture, out var rle))
            {
                var last = result.Count > 0 ? result[^1] : 0;
                for (var j = 0; j < rle; j++)
                {
                    result.Add(last);
                }
            }
        }

        result.Add(0);
        return result.Select(static x => (double)x).ToArray();
    }

    private static IEnumerable<int> ToInt12Stream(string b64)
    {
        for (var i = 0; i + 1 < b64.Length; i += 2)
        {
            yield return ToInt12(b64[i], b64[i + 1]);
        }
    }

    private static int ToInt12(char c0, char c1)
    {
        var uint12 = (ToUint6(c0) << 6) | ToUint6(c1);
        return ((uint12 >> 11) & 1) == 1 ? uint12 - 4096 : uint12;
    }

    private static int ToUint6(char b64)
    {
        var c = (int)b64;
        if (c >= 97) return c - 71;
        if (c >= 65) return c - 65;
        if (c >= 48) return c + 4;
        if (c == 43) return 62;
        if (c == 47) return 63;
        throw new ArgumentException("Invalid base64 pitch character.");
    }

    [GeneratedRegex("(HG|Hb|Hv|Ht|He|g|t|A|P|G)(-?\\d+)?", RegexOptions.Compiled)]
    private static partial Regex FlagRegex();
}

internal sealed record ResamplerRequest(
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
